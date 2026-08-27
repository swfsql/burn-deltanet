//! The model configuration for the register-majority example — one DeltaNet
//! block whose state is a three-row register file, sized so that nothing *but*
//! the block can solve the task (see [`model_config`]).

use crate::dataset::{NUM_CLASSES, NUM_REGISTERS, NUM_SYMBOLS};
use burn_deltanet::prelude::{
    DeltaLatentNetConfig, DeltaLatentShape, DeltaNetConfig, DeltaNetworkShape, QkActivation, QkNorm,
};

/// A DeltaNet block's state is a `[head_k_dim, head_v_dim]` matrix updated by
///
/// ```ignore
/// Sₜ = (I − βₜ kₜ kₜᵀ) Sₜ₋₁ + βₜ kₜ vₜᵀ         yₜ = Sₜᵀ qₜ / √head_k_dim
/// ```
///
/// With `‖k‖ = 1` (the L2 QK-norm) and `β = 1` that is exactly a **keyed
/// register write**: the row along `k` is replaced by `v` and every other row is
/// left alone. The task is built around precisely that, at `head_k_dim = 3`
/// (three registers) and `head_v_dim = 2`:
///
/// - **`k` selects the register.** The `a`/`b`/`c` write symbols project to the
///   three unit axes, so a write lands on one row and only that row.
/// - **`β` says whether this token writes at all.** `β ≈ 1` on a write
///   (`I − k kᵀ` is then a projection — the old contents are gone, not blended),
///   `β ≈ 0` on a query, which leaves the file untouched.
/// - **`v` carries the bit plus a reference axis**: `v = (±V, R)` with `R > 0`
///   constant. The bit alone would be one-dimensional, and the block's per-head
///   RMSNorm would flatten it to a hard sign; the constant second channel keeps
///   the normalised output a *direction*, so the margin stays proportional to
///   the vote and the readout stays differentiable.
/// - **`q` reads the whole file at once**: `q = (1, 1, 1)/√3`, so
///   `y = Sᵀq ∝ (V·Σbits, 3R)` — the majority the task asks for, and never a tie
///   because three `±1`s never sum to zero.
///
/// Three config choices are load-bearing:
///
/// - `use_short_conv = false` removes the causal convolution, so the recurrent
///   state is the model's only memory — there is no local window to shortcut
///   through, and no way to answer from the last few symbols.
/// - `ignore_last_residual` zeroes the single layer's residual, so `out_proj`
///   reads the block's output *alone*. Without it the head also sees the
///   embedding of the current token, which cannot give the answer but does
///   muddy the claim.
/// - `qk_activation = Identity` leaves `q`/`k` linear in the token embedding.
///   The default SiLU is folded into the convolution, which is switched off
///   here; keeping the axes exactly reachable is what makes the construction in
///   `tests.rs` closed-form.
///
/// `d_model = 4` is the smallest width that carries what the projections have to
/// read: a two-channel register code, the bit, and a write/query flag. Every
/// channel the block needs is an affine functional of those four, which is why
/// the hand-built solution exists at all (`tests.rs` asserts the fit is exact).
pub fn model_config() -> DeltaLatentNetConfig {
    // key_dim = 3 (one row per register), value_dim = 2 (the bit + the
    // reference axis), nheads = 1 ⇒ the state is a single 3×2 matrix: six
    // scalars, and the model's entire memory.
    let block = DeltaNetConfig::new(D_MODEL)
        .with_nheads(1)
        .with_expand_k(NUM_REGISTERS as f64 / D_MODEL as f64)
        .with_expand_v(HEAD_V_DIM as f64 / D_MODEL as f64)
        .with_use_beta(true) // β must be projected: it is the write-enable
        .with_use_gate(false)
        .with_allow_neg_eigval(false) // a write is a replacement, never a reflection
        .with_use_short_conv(false) // no conv: the state is the only memory
        .with_qk_activation(QkActivation::Identity)
        .with_qk_norm(QkNorm::L2) // ‖k‖ = 1 puts the write decision on β alone
        .with_has_proj_bias(true);

    // input  [batch, seq, NUM_SYMBOLS]  (one-hot symbol)
    // output [batch, seq, NUM_CLASSES]  (Neg / Pos logits, every scored position)
    DeltaLatentNetConfig::DeltaNet {
        shape: DeltaLatentShape::new(
            NUM_SYMBOLS,
            NUM_CLASSES,
            DeltaNetworkShape::new(1)
                // the single layer's residual: dropped, so the head sees only
                // what came out of the delta rule
                .with_ignore_last_residual(true),
        )
        // no final norm: the block's output is already O(1) (its RMSNorm bounds
        // it) and the head reads it directly.
        .with_final_norm(false),
        block,
    }
}

/// Model width — see [`model_config`].
pub const D_MODEL: usize = 4;
/// Value width per head: the bit channel plus the reference axis.
pub const HEAD_V_DIM: usize = 2;
