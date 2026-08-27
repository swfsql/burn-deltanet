//! # Register-majority — the smallest task a DeltaNet block is *needed* for
//!
//! One DeltaNet block whose recurrent state is a three-row register file, no
//! convolution, no residual: the model reads a stream of **writes** (`a+`, `a-`,
//! `b+`, …) and **queries** (`?`), and must report, at every query, the majority
//! of the three registers' current contents.
//!
//! The task is chosen so that nothing but the block's *keyed, erasing* write can
//! solve it:
//!
//! - the answer is not a function of the current symbol (`?` carries nothing),
//!   and the residual is switched off anyway,
//! - the lookback is unbounded (a register may have been written arbitrarily far
//!   back) and `use_short_conv = false` removes the only local window there was,
//! - **and the write has to erase**. Linear attention's `S ← αS + k vᵀ`
//!   accumulates: the only way it can remove one register's superseded value is
//!   a decay that fades every register. Two adversarial families pin that down
//!   from both sides — see [`dataset`](crate::dataset).
//!
//! ## Run
//!
//! ```bash
//! cargo run --release --example register-majority -- --training --inference
//!
//! # the claims above, measured: a hand-built exact solution, and a sweep
//! # showing no accumulating state reaches it
//! cargo test --release --example register-majority -- --nocapture
//! ```

#![allow(clippy::let_and_return)]
#![allow(clippy::module_inception)]

pub use common::{
    cli::AppArgs,
    training::{CosineAnnealingLr, Lr, TrainingConfig},
};

/// The register-majority dataset and its adversarial families.
pub mod dataset;
/// Inference: per-family accuracy on fresh eval sets.
pub mod inference;
/// The example's `model_config()`.
pub mod model;
/// Training entry point for the register-majority task.
pub mod training;

/// The hand-built solution and the accumulating-state sweep (see the module docs).
#[cfg(test)]
pub mod tests;

/// Shared example infrastructure (included by path).
#[path = "../common/mod.rs"]
pub mod common;

/// Wire up the device, configs, and the train/infer flow for the task.
pub fn launch(app_args: &AppArgs) {
    assert!(
        app_args.extra_args.is_empty(),
        "no extra arguments required"
    );
    app_args.create_artifact_dir();

    // `Device::default()` resolves to the enabled `backend-*` feature (honouring
    // the `BURN_DEVICE` env override); `configure_dtype` installs fp16/i32 when
    // `dev-f16` is on.
    let mut device = burn::prelude::Device::default();
    common::device::configure_dtype(&mut device);
    // training needs an autodiff-enabled device; inference uses the plain one.
    let autodiff_device = device.clone().autodiff();
    let dtype = burn::tensor::Tensor::<1>::zeros([1], &device).dtype();

    let (batch_size, num_epochs) = (64, 30);
    let training_config = app_args.load_training_config().unwrap_or_else(|| {
        println!("Initializing new training config");
        // The write-enable `β` has to saturate (a partial erase blends the old
        // value into the new one), which needs a large step to reach and a small
        // one to settle at — so this anneals rather than holding one rate.
        let total_steps = num_epochs * dataset::NUM_TRAIN.div_ceil(batch_size);
        TrainingConfig::new(common::training::OptimizerConfig::new(
            common::training::optimizer_config(dtype),
        ))
        .with_num_epochs(num_epochs)
        .with_batch_size(batch_size)
        .with_num_workers(2)
        .with_lr(Lr::CosineAnnealing(
            CosineAnnealingLr::new(total_steps)
                .with_max_lr(3e-2)
                .with_min_lr(1e-4)
                .with_warmup_steps(100),
        ))
    });
    let model_config = app_args.load_model_config().unwrap_or_else(|| {
        println!("Initializing new model config");
        model::model_config()
    });
    app_args.save_training_config(&training_config);
    app_args.save_model_config(&model_config);

    if app_args.training {
        training::train(
            training_config,
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

fn main() {
    let app_args = AppArgs::parse().unwrap();
    launch(&app_args);
}
