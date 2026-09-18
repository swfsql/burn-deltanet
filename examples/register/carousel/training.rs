//! Training loop for the register-carousel example: builds the dataloaders,
//! runs the train/validate epochs, and checkpoints the model and optimizer. The
//! [`Wrap`] newtype adapts the network to Burn's `TrainStep` / `InferenceStep`
//! via a cross-entropy head over the **scored** positions (the queries).

pub use crate::common::{
    cli::AppArgs,
    model::ModelConfigExt,
    session::{Cadence, Session},
    training::{TrainingConfig, metric_current},
};
use crate::dataset::{
    CarouselBatch, CarouselBatcher, CarouselDataset, EVAL_SEED, Family, NUM_CLASSES, NUM_EVAL,
    NUM_TRAIN, SEQ_LENGTH, TRAIN_SEED, Turn,
};
use burn::prelude::*;
use burn::{
    data::dataloader::{DataLoader, DataLoaderBuilder, Progress},
    module::AutodiffModule,
    optim::ModuleOptimizer,
    train::metric::{Adaptor, Metric, MetricMetadata, Numeric},
    train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};
use burn_deltanet::prelude::*;

/// The evaluation splits, reported separately: `cycle` is where a model that
/// cannot hold the carousel's period fails, so a single averaged number would
/// hide the whole point (see [`crate::dataset`]).
pub const EVAL_FAMILIES: [(&str, Family); 2] =
    [("random", Family::Random), ("cycle", Family::Cycle)];

/// The delta-rule evaluation used throughout the example.
///
/// Two chunks over a 32-token sequence, so the chunkwise WY path's inter-chunk
/// recurrence is actually exercised rather than degenerating to a single chunk.
/// The micro-step fold makes the delta rule's own sequence `SEQ_LENGTH · u`
/// long; the chunk length is in *those* steps, which is why it is left at a
/// divisor of it. `DeltaPath::Recurrent` computes the identical function.
pub fn path() -> DeltaPath {
    DeltaPath::chunk_len(SEQ_LENGTH / 2)
}

/// Run the full training routine: load/init the model and optimizer, then train
/// for the configured number of epochs (validating and checkpointing along the
/// way).
pub fn train(
    training_config: TrainingConfig,
    model_config: DeltaLatentNetConfig,
    turn: Turn,
    training_device: Device,
    app_args: &AppArgs,
) {
    training_device.seed(training_config.seed);

    let model: DeltaLatentNet = app_args.load_or_save_model(&model_config, &training_device);
    println!("Number of parameters: {}", model.num_params());
    let muon_plan = ModelConfigExt::muon_plan(&model_config);
    if training_config.optimizer.muon.is_some() {
        print!("{}", muon_plan.describe(&model));
    }
    let (mut optim, progress) =
        app_args.load_or_save_optim(training_config.optimizer.init(&muon_plan));

    let mut model = Wrap(model, model_config.clone());
    let batcher = CarouselBatcher::default();

    // Training batches live on the autodiff device (to match the weights);
    // validation runs on the inner backend.
    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(training_config.batch_size)
        .shuffle(progress.shuffle_seed(training_config.seed))
        .num_workers(training_config.num_workers)
        .set_device(training_device.clone())
        .build(CarouselDataset::new(
            NUM_TRAIN,
            SEQ_LENGTH,
            Family::Mixed,
            turn,
            TRAIN_SEED,
        ));
    let valid_loaders: Vec<(&str, Dataloader)> = EVAL_FAMILIES
        .iter()
        .map(|(name, family)| {
            let loader: Dataloader = DataLoaderBuilder::new(batcher.clone())
                .batch_size(training_config.batch_size)
                .num_workers(training_config.num_workers)
                .set_device(training_device.clone().inner())
                .build(CarouselDataset::new(
                    NUM_EVAL, SEQ_LENGTH, *family, turn, EVAL_SEED,
                ));
            (*name, loader)
        })
        .collect();

    // Resume position, `--max-batches` budget, cadence and metrics log: by
    // default a validation every five epochs and no mid-epoch checkpoint.
    let batches = dataloader_train.num_items().div_ceil(training_config.batch_size);
    let cadence = Cadence {
        valid_every: Some(5 * batches),
        ..Cadence::default()
    };
    let mut session = app_args.session(
        progress,
        &training_config,
        cadence,
        dataloader_train.num_items(),
    );

    println!(
        "running initial validation (chance ≈ {:.1}%)...",
        100.0 / NUM_CLASSES as f32
    );
    validate_all(&valid_loaders, model.0.valid(), &model_config, 0, &mut session);

    println!("Starting training...");
    for epoch in session.epochs(training_config.num_epochs) {
        model.0 = epoch_train(
            std::sync::Arc::clone(&dataloader_train),
            model.0,
            &training_config,
            &model_config,
            &mut optim,
            &mut session,
            epoch,
            &valid_loaders,
            app_args,
        );

        app_args.save_model(&model.0);
        app_args.save_optim(&optim, session.progress());

        let last = epoch == training_config.num_epochs || session.is_exhausted();
        if last && !session.validated_now() {
            println!("running final validation...");
            validate_all(&valid_loaders, model.0.valid(), &model_config, epoch, &mut session);
        }

        if session.is_exhausted() {
            println!("reached the --max-batches limit; stopping training");
            break;
        }
    }
    println!("Training finished.");
}

type Dataloader = std::sync::Arc<dyn DataLoader<CarouselBatch> + 'static>;

/// Train for (the rest of) one epoch, stepping the optimizer per batch and
/// checkpointing and validating (on `valid_loaders`) at the `session`'s cadence;
/// returns the updated model. Ends early once the session's budget
/// (`--max-batches`) runs out.
#[allow(clippy::too_many_arguments)]
pub fn epoch_train(
    dataloader_train: Dataloader,
    training_model: DeltaLatentNet,
    training_config: &TrainingConfig,
    model_config: &DeltaLatentNetConfig,
    optim: &mut ModuleOptimizer,
    session: &mut Session,
    epoch: usize,
    valid_loaders: &[(&str, Dataloader)],
    app_args: &AppArgs,
) ->DeltaLatentNet {
    let batches = dataloader_train.num_items().div_ceil(training_config.batch_size);
    let mut loss_metric = burn::train::metric::LossMetric::new();
    let mut acc_metric = burn::train::metric::AccuracyMetric::new();
    let mut iteration_speed_metric = burn::train::metric::IterationSpeedMetric::new();

    let mut training_model = Wrap(training_model, model_config.clone());

    for batch in dataloader_train
        .iter()
        .map(|batch| batch.expect("dataloader batch"))
        .take(session.batch_limit(batches))
    {
        let b = session.begin_batch();
        let [batch_size, _, _] = batch.inputs.dims();
        let (_step, lr) = session.begin_step(batch_size);

        let train_output = TrainStep::step(&training_model, batch);
        let pre_metrics = &train_output.item;

        loss_metric.update(&pre_metrics.adapt(), session.meta());
        acc_metric.update(&pre_metrics.adapt(), session.meta());
        iteration_speed_metric.update(&pre_metrics.adapt(), session.meta());

        training_model.0 = optim.step(lr, training_model.0, train_output.grads);

        let (loss, acc) = (
            metric_current(loss_metric.value()),
            metric_current(acc_metric.value()),
        );
        session.log_train(&[("loss", loss), ("acc", acc)]);
        println!(
            "Epoch {}/{}, Batch {b:0>4}/{batches}, Loss {loss:.4}, Acc {acc:0>6.2}, lr {lr:0>6.2e}, it/s {:.2}",
            epoch,
            training_config.num_epochs,
            metric_current(iteration_speed_metric.value()),
        );

        if session.checkpoint_due() {
            app_args.save_model(&training_model.0);
            app_args.save_optim(optim, session.progress());
        }
        if session.valid_due() {
            println!("running validation...");
            let valid_model = training_model.0.valid();
            validate_all(valid_loaders, valid_model, model_config, epoch, session);
        }
    }

    println!(
        "Epoch {}/{}, Avg Loss {:.4}, Avg Acc: {}",
        epoch,
        training_config.num_epochs,
        metric_current(loss_metric.running_value()),
        metric_current(acc_metric.running_value()),
    );
    session.end_epoch(batches);

    training_model.0
}

/// Validate on every family in turn (each capped at the session's
/// `valid_batches`), one line and one metrics-log entry each.
pub fn validate_all(
    loaders: &[(&str, Dataloader)],
    valid_model: DeltaLatentNet,
    model_config: &DeltaLatentNetConfig,
    epoch: usize,
    session: &mut Session,
) {
    let valid_model = Wrap(valid_model, model_config.clone());
    let limit = session.cadence().valid_batches.unwrap_or(usize::MAX);
    for (name, loader) in loaders {
        let (loss, acc) = evaluate(std::sync::Arc::clone(loader), &valid_model, limit);
        session.log_valid(name, &[("loss", loss), ("acc", acc)]);
        println!("  epoch {epoch}, {name:<7} loss {loss:.4}, acc {acc:6.2}%");
    }
}

/// Average loss and accuracy of `model` over (up to `limit` batches of) one
/// dataloader.
pub fn evaluate(dataloader: Dataloader, model: &Wrap, limit: usize) -> (f64, f64) {
    let metric_meta = MetricMetadata {
        progress: Progress::new(0, dataloader.num_items(), None),
        iteration: Some(0),
        lr: None,
    };
    let mut loss_metric = burn::train::metric::LossMetric::new();
    let mut acc_metric = burn::train::metric::AccuracyMetric::new();

    for batch in dataloader.iter().map(|b| b.expect("dataloader batch")).take(limit) {
        let pre_metrics = InferenceStep::step(model, batch);
        loss_metric.update(&pre_metrics.adapt(), &metric_meta);
        acc_metric.update(&pre_metrics.adapt(), &metric_meta);
    }
    (
        metric_current(loss_metric.running_value()),
        metric_current(acc_metric.running_value()),
    )
}

/// Wrapper over [`DeltaLatentNet`] for custom implementations.
pub struct Wrap(pub DeltaLatentNet, pub DeltaLatentNetConfig);

impl TrainStep for Wrap {
    type Input = CarouselBatch;
    type Output = ClassificationOutput;

    fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
        let pre_metrics = InferenceStep::step(self, batch);
        let grads = pre_metrics.loss.backward();
        TrainOutput::new(&self.0, grads, pre_metrics)
    }
}

impl InferenceStep for Wrap {
    type Input = CarouselBatch;
    type Output = ClassificationOutput;

    fn step(&self, batch: Self::Input) -> Self::Output {
        self.forward_classification(batch.inputs, batch.targets, batch.scored)
    }
}

impl Wrap {
    /// Forward the model and score the read port at every **query** position.
    pub fn forward_classification(
        &self,
        inputs: Tensor<3>,
        targets: Tensor<2, Int>,
        scored: Tensor<1, Int>,
    ) -> ClassificationOutput {
        let model = &self.0;
        let [batch_size, sequence_size, _num_symbols] = inputs.dims();
        assert_eq!([batch_size, sequence_size], targets.dims());

        let (output, _caches) = model.forward(inputs, None, path(), None);
        assert_eq!([batch_size, sequence_size, NUM_CLASSES], output.dims());

        // Keep only the positions with something to report; the pushes, the
        // turns and the queries issued before register A is loaded reach
        // neither the loss nor the accuracy (see `dataset::IGNORE`).
        let n = batch_size * sequence_size;
        let logits = output.reshape([n, NUM_CLASSES]).select(0, scored.clone());
        let targets = targets.reshape([n]).select(0, scored);

        let loss = burn::nn::loss::CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits.clone(), targets.clone());

        ClassificationOutput::new(loss.clone(), logits, targets)
    }
}
