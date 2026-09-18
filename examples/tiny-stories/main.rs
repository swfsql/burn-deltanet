//! # TinyStories character-level language model
//!
//! An auto-regressive Gated DeltaNet LM over single **characters** of the
//! [TinyStories-GPT4-clean] corpus: two Gated DeltaNet blocks (each with the
//! reference SwiGLU MLP, joined by Multi-Gate residuals) between a **tied**
//! 48-character embedding and its transpose.
//!
//! [TinyStories-GPT4-clean]: https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean
//!
//! One item is one **story**, opened by four class latents rather than by a
//! separator character. Training scores every position of a 256-character window
//! against its next character — plus, in the story's first window, the latents'
//! readout against its first character — and walks the story's windows carrying
//! the (detached) state from each into the next for as long as the frontier gate
//! admits it; see the README's data section. Inference replays those latents with
//! one `prime` (which is already the first character's distribution), optionally
//! prefills a prompt with one chunkwise `forward`, and then samples one character
//! per `step`.
//!
//! Corpus knobs are forwarded after the trailing `--` (they are written into the
//! artifacts' `training_config.json`, so resuming a run keeps them):
//!
//! ```bash
//! # train and then sample (downloads the 673MB parquet once, if not cached yet)
//! cargo run --release --example tiny-stories --features backend-flex -- --training --inference
//! # a bigger corpus and a longer window
//! cargo run --release --example tiny-stories --features backend-flex -- --training \
//!     -- --train-stories 32768 --seq-len 512
//! ```

#![allow(clippy::let_and_return)]
#![allow(clippy::module_inception)]

pub use common::{
    cli::AppArgs,
    tiny_stories::dataset,
    tiny_stories::lm::{Overrides, TinyStoriesConfig},
    training::{CosineAnnealingLr, Lr, TrainingConfig},
};

/// Sampling from the trained LM.
pub mod inference;
/// The example's `model_config()`.
pub mod model;
/// Training entry point for the LM.
pub mod training;

/// Shared example infrastructure (included by path).
#[path = "../common/mod.rs"]
pub mod common;

/// Wire up the device, configs, and the train/infer flow for the LM.
pub fn launch(app_args: &AppArgs) {
    let overrides = Overrides::parse(&app_args.extra_args);
    app_args.create_artifact_dir();

    // `Device::default()` resolves to the enabled `backend-*` feature (honouring
    // the `BURN_DEVICE` env override); `configure_dtype` installs fp16/i32 when
    // `dev-f16` is on.
    let mut device = burn::prelude::Device::default();
    common::device::configure_dtype(&mut device);
    let autodiff_device = device.clone().autodiff();
    let dtype = burn::tensor::Tensor::<1>::zeros([1], &device).dtype();

    // setup training and model configs
    //
    // Batch 16 is the largest that is *free*: at this model size the GPU is
    // launch-bound, so batches 8 and 16 run at the same steps/second and 16
    // simply sees twice the corpus for the same wall clock. It does not extend —
    // 32 halves the rate — so this is the edge of the free regime rather than a
    // preference.
    let batch_size = 16;
    let num_epochs = 16;
    let loaded = app_args.load_training_config::<TinyStoriesConfig>();
    let is_fresh = loaded.is_none();
    let mut config = loaded.unwrap_or_else(|| {
        println!("Initializing new training config");
        // Muon on the block's hidden weight matrices, AdamW on everything else
        // (the per-head scalar channels, the norms, the embedding);
        // `--no-muon` returns to plain AdamW.
        let optimizer = common::training::OptimizerConfig::adamw_only(dtype)
            .with_muon_defaults(ADAMW_WEIGHT_DECAY);
        TinyStoriesConfig::new(
            TrainingConfig::new(optimizer)
                .with_num_epochs(num_epochs)
                .with_batch_size(batch_size)
                .with_num_workers(2),
        )
    });
    overrides.apply(&mut config);
    if is_fresh {
        // The cosine schedule spans the whole run, so it can only be sized once
        // the corpus knobs are settled. It is counted in *windows*, not in
        // dataloader items, and the training loop charges the schedule for the
        // windows the frontier gate skipped too — so a stalling gate shortens
        // the run rather than leaving the cosine unfinished.
        //
        // One item is one story, and a batch runs the windows of its *longest*
        // story — so the schedule is sized from that maximum, not from the mean:
        // ~1350 characters is the expected longest of `batch_size = 16` draws
        // (the corpus's mean is ~820, its 90th percentile 1103, its longest
        // story 4149).
        const CHARS_PER_LONGEST_STORY: usize = 1350;
        let batches_per_epoch = config.train_stories / config.training.batch_size;
        let iterations_per_epoch =
            batches_per_epoch * CHARS_PER_LONGEST_STORY.div_ceil(config.seq_len);
        config.training.lr = Lr::CosineAnnealing(
            CosineAnnealingLr::new(config.training.num_epochs * iterations_per_epoch)
                // A broad optimum — halving or doubling it both measure worse,
                // and anything much lower is far worse. It was tuned over the
                // opening few hundred steps, where the cosine has barely moved,
                // so read it as the peak rate rather than as a tuned anneal.
                .with_max_lr(24e-3)
                .with_min_lr(24e-4)
                .with_warmup_steps(iterations_per_epoch / 20), // 5% of an epoch
        );
    }
    // After the sizing, so `--epochs` rescales the schedule and `--max-lr`
    // replaces its peak on a fresh config too.
    app_args.override_training_config(&mut config.training);
    let model_config = app_args.load_model_config().unwrap_or_else(|| {
        println!("Initializing new model config");
        model::model_config()
    });
    // save configs
    app_args.save_training_config(&config);
    app_args.save_model_config(&model_config);

    if app_args.training {
        training::train(
            config.clone(),
            model_config.clone(),
            autodiff_device,
            app_args,
        );
    }

    if app_args.inference {
        inference::infer(model_config, device, app_args);
    }

    if !app_args.inference && !app_args.training {
        println!("neither training nor inference were enabled");
        println!("{}", common::cli::HELP);
    }
}

/// AdamW's default weight decay, mirrored into the Muon group so the two arms
/// decay the same weights by the same amount.
const ADAMW_WEIGHT_DECAY: f32 = 1e-4;

fn main() {
    let app_args = AppArgs::parse(common::ARTIFACT_PREFIX).unwrap();
    launch(&app_args);
}
