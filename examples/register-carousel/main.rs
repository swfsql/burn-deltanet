//! # Register-carousel — the smallest task that needs *two* Householders
//!
//! The rung above `register-majority`. Same three-row register file, same six
//! state scalars, one new instruction: `R` **turns the carousel**, permuting the
//! registers under the single input port. The model reads `+` / `-` (overwrite
//! register A), `R`, and `?` (read register A back), and must report the bit
//! currently at the port.
//!
//! A delta-rule transition is a generalised Householder `I − β k kᵀ`, whose
//! eigenvalues with `‖k‖ = 1` are `{1, 1, 1−β}` — all **real**:
//!
//! - `-- --turn swap` makes `R` a transposition. `I − 2wwᵀ` with
//!   `w = (e_A − e_B)/√2` *is* that transposition — so **one** factor suffices,
//!   provided `β` may reach 2 (`allow_neg_eigval`). With `β ≤ 1` the spectrum
//!   lies in `[0, 1]` and the state cannot oscillate at all.
//! - `-- --turn rotate` (the default) makes `R` a 3-cycle, whose eigenvalues are
//!   `1, e^{±2πi/3}`. No real Householder is that matrix, and no orbit of one
//!   has period 3 — but `swap(A,C) ∘ swap(A,B)` is, which is exactly what
//!   `n_householder = 2` gives per token.
//!
//! ## Run
//!
//! ```bash
//! cargo run --release --example register-carousel -- --training --inference
//!
//! # the rung below: one transposition per turn, one factor
//! cargo run --release --example register-carousel -- --training --inference -- --turn swap
//!
//! # the ablation: the 3-cycle with one factor, which cannot reach it
//! cargo run --release --example register-carousel -- --training --inference -- --factors 1
//!
//! # the claims above, measured: hand-built exact solutions and the sweep over
//! # every single Householder
//! cargo test --release --example register-carousel -- --nocapture
//! ```
//!
//! Two downstream flags, forwarded after the trailing `--`: `--turn
//! rotate|swap` (default `rotate`) selects the permutation the dataset applies,
//! and `--factors N` overrides `n_householder` in a **fresh** model config (a
//! persisted one wins on reload).

#![allow(clippy::let_and_return)]
#![allow(clippy::module_inception)]

pub use common::{
    cli::AppArgs,
    training::{CosineAnnealingLr, Lr, TrainingConfig},
};

/// The register-carousel dataset, its turns and its families.
pub mod dataset;
/// Inference: per-family accuracy on fresh eval sets.
pub mod inference;
/// The example's `model_config()`.
pub mod model;
/// Training entry point for the register-carousel task.
pub mod training;

/// The hand-built solutions and the single-Householder sweep.
#[cfg(test)]
pub mod tests;

/// Shared example infrastructure (included by path).
#[path = "../common/mod.rs"]
pub mod common;

use dataset::Turn;
use std::ffi::OsString;

/// Wire up the device, configs, and the train/infer flow for the task.
pub fn launch(app_args: &AppArgs) {
    let turn = Turn::parse(flag(&app_args.extra_args, "--turn").as_deref());
    let factors = flag(&app_args.extra_args, "--factors")
        .map(|v| v.parse().expect("--factors must be a positive integer"))
        .unwrap_or_else(|| model::default_factors(turn));
    app_args.create_artifact_dir();

    // `Device::default()` resolves to the enabled `backend-*` feature (honouring
    // the `BURN_DEVICE` env override); `configure_dtype` installs fp16/i32 when
    // `dev-f16` is on.
    let mut device = burn::prelude::Device::default();
    common::device::configure_dtype(&mut device);
    // training needs an autodiff-enabled device; inference uses the plain one.
    let autodiff_device = device.clone().autodiff();
    let dtype = burn::tensor::Tensor::<1>::zeros([1], &device).dtype();

    let (batch_size, num_epochs) = (64, 150);
    let training_config = app_args.load_training_config().unwrap_or_else(|| {
        println!("Initializing new training config");
        // Every factor has to settle on an exact reflection (`β = 2`, `k` on a
        // swap axis); a partial one leaks a little of each register into the
        // next turn and compounds. Worse, the one-factor solution is a broad
        // plateau on the way (see the README), and leaving it takes a long
        // stretch at a high rate — which is why the schedule is this long and
        // shortening it strands the run at ≈70%.
        let total_steps = num_epochs * dataset::NUM_TRAIN.div_ceil(batch_size);
        TrainingConfig::new(common::training::OptimizerConfig::new(
            common::training::optimizer_config(dtype),
        ))
        .with_num_epochs(num_epochs)
        .with_batch_size(batch_size)
        .with_num_workers(2)
        // Seed 1 rather than 0: finding the second reflection is a basin, and
        // not every init falls into it — see the README's note.
        .with_seed(1)
        .with_lr(Lr::CosineAnnealing(
            CosineAnnealingLr::new(total_steps)
                .with_max_lr(3e-2)
                .with_min_lr(1e-4)
                .with_warmup_steps(100),
        ))
    });
    let model_config = app_args.load_model_config().unwrap_or_else(|| {
        println!("Initializing new model config (turn {turn:?}, {factors} factor(s))");
        model::model_config(turn, factors)
    });
    app_args.save_training_config(&training_config);
    app_args.save_model_config(&model_config);

    if app_args.training {
        training::train(
            training_config,
            model_config.clone(),
            turn,
            autodiff_device,
            app_args,
        );
    }

    if app_args.inference {
        inference::infer(model_config, turn, device, app_args);
    }

    if !app_args.inference && !app_args.training {
        println!("neither training nor inference were enabled");
        println!("{}", common::cli::HELP);
    }
}

/// The value following `name` in the trailing `--` arguments, if present.
fn flag(extra_args: &[OsString], name: &str) -> Option<String> {
    extra_args
        .iter()
        .position(|a| a == name)
        .and_then(|i| extra_args.get(i + 1))
        .map(|v| v.to_string_lossy().into_owned())
}

fn main() {
    let app_args = AppArgs::parse().unwrap();
    launch(&app_args);
}
