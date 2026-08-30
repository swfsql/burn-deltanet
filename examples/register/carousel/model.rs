//! The model configuration for the register-carousel example — one DeltaProduct
//! block whose state is a three-row register file, with `n_householder` factors
//! per token (see [`model_config`]).

use crate::dataset::{NUM_CLASSES, NUM_REGISTERS, NUM_SYMBOLS, Turn};
use burn_deltanet::prelude::{
    DeltaBlockConfig, DeltaLatentNetConfig, DeltaLatentShape, DeltaNetworkShape, DeltaProductConfig,
};

/// Model width: four symbols placed at the vertices of a regular tetrahedron, so
/// every per-symbol table the block needs is a *unique* affine functional of the
/// embedding (and every embedding has the same norm, which makes the layer's
/// pre-`RmsNorm` a pass-through).
pub const D_MODEL: usize = 3;
/// Value width per head: the bit channel plus the reference axis.
pub const HEAD_V_DIM: usize = 2;

/// A DeltaProduct block applies `u = n_householder` transitions per token:
///
/// ```ignore
/// Sₜ = (I − β⁽ᵘ⁾k⁽ᵘ⁾k⁽ᵘ⁾ᵀ) ⋯ (I − β⁽¹⁾k⁽¹⁾k⁽¹⁾ᵀ) Sₜ₋₁ + writes
/// ```
///
/// At `head_k_dim = 3` and `head_v_dim = 2` the state is a three-row register
/// file — the same six scalars as `register-majority` — and the task adds one
/// instruction to that file: **turn the carousel**, i.e. permute its rows.
///
/// - a **push** is one factor with `β = 1` and `k = e_A`: register A is replaced.
/// - a **turn** is one factor per transposition, with `β = 2`, `k` the swap axis
///   `(eᵢ − eⱼ)/√2` and nothing written: `I − 2kkᵀ` *is* that transposition.
/// - a **query** spends every factor at `β = 0`, and `q = e_A` reads the port.
///
/// So [`Turn::Swap`] needs one factor and [`Turn::Rotate`] two, and that is the
/// whole point of the example: a 3-cycle has complex eigenvalues and no single
/// real Householder can be it. `factors` is deliberately a knob so the ablation
/// (`-- --turn rotate --factors 1`) is a run rather than an argument.
///
/// Three config choices are load-bearing:
///
/// - `allow_neg_eigval = true` lets `β` reach 2, which is what makes a factor an
///   exact *reflection* rather than a contraction. Without it the transition's
///   spectrum sits in `[0, 1]` and the state cannot oscillate at all.
/// - `use_forget_gate = false`: a turn must be norm-preserving, and there is
///   nothing here to forget. `α ≡ 1` also keeps the negative claim clean — the
///   transition is a pure Householder product, with no decay to hide behind.
/// - `use_short_conv = false` removes the local window, leaving the recurrent
///   state as the model's only memory.
pub fn model_config(turn: Turn, factors: usize) -> DeltaLatentNetConfig {
    assert!(factors >= 1, "at least one Householder factor per token");
    let block = DeltaProductConfig::new(D_MODEL)
        .with_n_householder(factors)
        .with_nheads(1)
        .with_head_k_dim(NUM_REGISTERS)
        .with_expand_v(HEAD_V_DIM as f64 / NUM_REGISTERS as f64)
        .with_use_forget_gate(false) // a turn is a permutation, not a decay
        .with_use_gate(false)
        .with_allow_neg_eigval(true) // β ∈ (0, 2): a factor may reflect
        .with_use_short_conv(false) // no conv: the state is the only memory
        .with_has_proj_bias(true);
    let _ = turn; // the turn shapes the dataset, not the block

    DeltaLatentNetConfig::new(
        DeltaLatentShape::new(
            NUM_SYMBOLS,
            NUM_CLASSES,
            DeltaNetworkShape::new(1).with_ignore_last_residual(true),
        )
        .with_final_norm(false),
        DeltaBlockConfig::DeltaProduct(block),
    )
}

/// The default number of Householder factors for a turn: exactly what the
/// permutation's order needs, and no more.
pub fn default_factors(turn: Turn) -> usize {
    match turn {
        Turn::Swap => 1,
        Turn::Rotate => 2,
    }
}
