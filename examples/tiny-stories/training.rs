//! Training loop for the character-level TinyStories LM: builds the window
//! dataloaders, runs the train/validate epochs, checkpoints the model and
//! optimizer, and samples a story at every validation point so the text can be
//! watched growing legible.
//!
//! The epoch loops themselves are `burn_stack::examples::tiny_stories::lm`,
//! shared with `burn-mamba`. What is delta-specific here is the [`Wrap`]
//! newtype: it adapts the network to Burn's `TrainStep` / `InferenceStep` via
//! next-character cross-entropy over **every** position of the window, and
//! supplies the `LmModel` seam the shared loops build against — including the
//! two cache-aware halves of it, since the loops train a *run* of windows and
//! carry [`DeltaCaches`] from each window into the next (see that module's
//! "Runs, carried state, and the frontier").

pub use crate::common::{
    cli::AppArgs,
    model::ModelConfigExt,
    tiny_stories::lm::TinyStoriesConfig,
    training::{TrainingConfig, metric_current},
};
use crate::dataset::TinyStoriesBatch;
use burn::prelude::*;
use burn::{
    data::dataloader::Progress,
    module::AutodiffModule,
    optim::{GradientsParams, ModuleOptimizer},
    train::metric::MetricMetadata,
    train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};
use burn_deltanet::prelude::{DeltaCaches, DeltaPath, DeltaVocabNet, DeltaVocabNetConfig};
use burn_stack::examples::tiny_stories::lm::{
    self, Frontier, LmModel, dataloaders, epoch_train, epoch_valid,
};
use burn_stack::modules::CacheStack;

/// The delta-rule path used for both training and prefill: the chunkwise WY
/// algorithm at the default chunk length. `DeltaPath::Recurrent` computes the
/// same thing one token at a time, and is what `step()` decodes with.
pub fn path() -> DeltaPath {
    DeltaPath::default()
}

/// Run the full training routine: load/init the model and optimizer, then train
/// for the configured number of epochs (validating, sampling and checkpointing
/// along the way).
pub fn train(
    config: TinyStoriesConfig,
    model_config: DeltaVocabNetConfig,
    training_device: Device,
    app_args: &AppArgs,
) {
    training_device.seed(config.training.seed);

    // load (or init and save) model and optim
    let model: DeltaVocabNet = app_args.load_or_save_model(&model_config, &training_device);
    println!("Number of parameters: {}", model.num_params());
    let muon_plan = ModelConfigExt::muon_plan(&model_config);
    if config.training.optimizer.muon.is_some() {
        // Which weights Muon took over (and where the fused ones split).
        print!("{}", muon_plan.describe(&model));
    }
    let mut optim = app_args.load_or_save_optim(config.training.optimizer.init(&muon_plan));

    let mut model = Wrap(model);

    // Create the dataloaders (downloading the corpus on the first run).
    let (dataloader_train, dataloader_valid) = dataloaders(&config, &training_device);

    let training_num_items = dataloader_train.num_items();

    let mut metric_meta = MetricMetadata {
        progress: Progress::new(0, training_num_items, None),
        iteration: Some(0),
        lr: Some(config.training.lr.get_lr(0).into()),
    };

    // `--max-batches`: an optional cap on the whole run, spent across epochs.
    let mut batch_budget = app_args.batch_budget();

    // The frontier gate outlives the epochs: its opening-window baseline is a
    // property of the model's current skill, not of where the epoch loop is.
    let mut frontier = Frontier::new(config.frontier.clone());

    println!("running small initial validation...");
    epoch_valid::<Wrap>(
        std::sync::Arc::clone(&dataloader_valid),
        &model.valid(),
        &config,
        0,
        Some(10),
    );

    println!("Starting training...");
    // Iterate over our training for X epochs
    for epoch in 1..config.training.num_epochs + 1 {
        model = epoch_train(
            std::sync::Arc::clone(&dataloader_train),
            std::sync::Arc::clone(&dataloader_valid),
            model,
            &config,
            &mut optim,
            &mut metric_meta,
            &mut frontier,
            epoch,
            &mut batch_budget,
            Some(10),
            app_args,
            training_device.clone().inner(),
        );

        // save assets
        app_args.save_model(&model.0);
        app_args.save_optim(&optim);

        println!("running full validation...");
        epoch_valid::<Wrap>(
            std::sync::Arc::clone(&dataloader_valid),
            &model.valid(),
            &config,
            epoch,
            None,
        );

        if batch_budget.is_exhausted() {
            println!("reached the --max-batches limit; stopping training");
            break;
        }
    }
    println!("Training finished.");
}

/// Wrapper over [`DeltaVocabNet`] for custom implementations.
pub struct Wrap(pub DeltaVocabNet);

impl LmModel for Wrap {
    type Valid = Wrap;
    type Caches = DeltaCaches;

    fn valid(&self) -> Self::Valid {
        Wrap(self.0.valid())
    }

    fn train_window(
        &self,
        batch: TinyStoriesBatch,
        caches: Option<Self::Caches>,
    ) -> (TrainOutput<ClassificationOutput>, Self::Caches) {
        let (pre_metrics, caches) = self.forward_lm(batch.inputs, batch.targets, caches);
        let grads = pre_metrics.loss.backward();
        (TrainOutput::new(&self.0, grads, pre_metrics), caches)
    }

    fn detach_caches(caches: Self::Caches) -> Self::Caches {
        // One cache type for all four families, so the `CacheStack` provided
        // method is the whole implementation.
        caches.detach()
    }

    fn valid_window(
        valid: &Self::Valid,
        batch: TinyStoriesBatch,
        caches: Option<Self::Caches>,
    ) -> (ClassificationOutput, Self::Caches) {
        valid.forward_lm(batch.inputs, batch.targets, caches)
    }

    fn optim_step(self, optim: &mut ModuleOptimizer, lr: f64, grads: GradientsParams) -> Self {
        Wrap(optim.step(lr, self.0, grads))
    }

    fn save(&self, app_args: &AppArgs) {
        app_args.save_model(&self.0);
    }

    fn generate(
        valid: &Self::Valid,
        device: &Device,
        prompt: &str,
        n_chars: usize,
        temperature: f64,
        seed: u64,
    ) -> String {
        crate::inference::generate(&valid.0, device, prompt, n_chars, temperature, seed)
    }
}

impl TrainStep for Wrap {
    type Input = TinyStoriesBatch;
    type Output = ClassificationOutput;

    fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
        LmModel::train_window(self, batch, None).0
    }
}

impl InferenceStep for Wrap {
    type Input = TinyStoriesBatch;
    type Output = ClassificationOutput;

    fn step(&self, batch: Self::Input) -> Self::Output {
        self.forward_lm(batch.inputs, batch.targets, None).0
    }
}

impl Wrap {
    /// Forward the LM from `caches` (`None` ⇒ a zero state) and score **every**
    /// position of the window against its next character (see
    /// [`lm_output`](burn_stack::examples::tiny_stories::lm::lm_output)),
    /// returning the window's final state alongside.
    pub fn forward_lm(
        &self,
        inputs: Tensor<2, Int>,
        targets: Tensor<2, Int>,
        caches: Option<DeltaCaches>,
    ) -> (ClassificationOutput, DeltaCaches) {
        let (logits, caches) = self.0.forward(inputs, caches, path(), None);
        (lm::lm_output(logits, targets), caches)
    }
}
