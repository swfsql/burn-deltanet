//! # The delta rule — the recurrence every family in this crate shares
//!
//! One associative memory `S ∈ ℝ^{head_k_dim × head_v_dim}` per head, written
//! with an error-correcting update rather than a pure accumulation:
//!
//! ```text
//!   Sₜ = αₜ (I − βₜ kₜ kₜᵀ) Sₜ₋₁ + βₜ kₜ vₜᵀ         (state)
//!   yₜ = Sₜᵀ qₜ · scale                               (readout)
//! ```
//!
//! equivalently, in the form the kernels actually evaluate:
//!
//! ```text
//!   S⁻ = αₜ Sₜ₋₁                 apply the forget gate
//!   uₜ = βₜ (vₜ − S⁻ᵀ kₜ)        the *delta*: how far the current association
//!                                is from the target, scaled by the write strength
//!   Sₜ = S⁻ + kₜ uₜᵀ             a rank-1 correction
//! ```
//!
//! Reading it that way is the whole point: linear attention writes `kₜ vₜᵀ`
//! unconditionally and lets collisions pile up, while the delta rule first
//! *retrieves* what it currently associates with `kₜ` and only writes the
//! difference. With `‖kₜ‖ = 1` the transition `I − βₜ kₜ kₜᵀ` is a generalised
//! Householder matrix with spectrum `{1, 1 − βₜ}` — the identity at `β = 0`, a
//! projection at `β = 1`, an exact reflection at `β = 2`. Allowing `β > 1`
//! (negative eigenvalues) is what lets the recurrence *track* group structure
//! instead of only forgetting it; see [`crate::common::norm`].
//!
//! `αₜ ∈ (0, 1]` is the optional forget gate: `αₜ ≡ 1` is
//! [DeltaNet](crate::deltanet), `αₜ = exp(Δₜ A)` is
//! [Gated DeltaNet](crate::gated_deltanet). Both live in the same code here,
//! with the gate as an `Option`, because the gated recurrence *is* the ungated
//! one at `g = 0` — but a `None` gate skips the decay tensors outright rather
//! than multiplying by ones.
//!
//! ## One rule, three gate widths
//!
//! `βₜ` above is *one number per head*: it sets both how much of the old
//! association is erased and how much of `vₜ` is written. Widening it onto the
//! channel axes decouples the two, and widening `αₜ` likewise gives each key
//! channel its own timescale — which is [GDN-2](crate::gdn2):
//!
//! ```text
//!   Sₜ = (I − kₜ (bₜ ⊙ kₜ)ᵀ) diag(αₜ) Sₜ₋₁ + kₜ (wₜ ⊙ vₜ)ᵀ
//!        bₜ ∈ ℝ^head_k_dim   erase        wₜ ∈ ℝ^head_v_dim   write
//! ```
//!
//! Setting `bₜ = wₜ = βₜ` and `αₜ` scalar recovers the line above *exactly*, so
//! the core carries the gates at their broadcast width (`1` or the full
//! channel count) and branches on nothing — see
//! [`DeltaInput`](path::DeltaInput).
//!
//! ## The chunkwise form (WY representation)
//!
//! Unrolling a chunk of `L` tokens gives a product of Householder matrices
//! applied to the incoming state. The classical **WY representation** collapses
//! that product into one rank-`L` correction (see the DeltaNet paper, Appendix
//! B, reproduced in the `flash-linear-attention` `delta_rule/README.md`):
//!
//! ```text
//!   P = Π (I − βᵢ kᵢ kᵢᵀ) = I − Kᵀ W,        H = Σ Pᵢ₊₁ βᵢ kᵢ vᵢᵀ = Kᵀ U
//!   T = (I + tril(diag(β) K Kᵀ, −1))⁻¹ diag(β)
//!   W = T K,      U = T V
//! ```
//!
//! so a whole chunk becomes four GEMMs against the incoming state `S₀`:
//!
//! ```text
//!   V' = U − W S₀                                   (the corrected values)
//!   O  = Q S₀ + tril(Q Kᵀ) V'                       (output)
//!   S  = S₀ + Kᵀ V'                                 (next state)
//! ```
//!
//! With the forget gate the same identities hold once every `k`, `q` and the
//! masks carry their cumulative decay `e^{Gᵢ−Gⱼ}` — that is the only change,
//! and it is why one implementation serves both families.
//!
//! Chunks are still processed **serially**: `V'` depends on `S₀`, so the
//! inter-chunk recurrence `S ← (α I − Kᵀ W) S + Kᵀ U` is matrix-valued with a
//! rank-`L` update, not a scalar decay, and admits no cheap parallel scan the
//! way Mamba-2's does. The win is entirely intra-chunk: `L` serial rank-1
//! updates become a handful of `[L, L]` and `[L, head_dim]` matmuls.
//!
//! ## Layout
//!
//! - [`tri`] — the `(I − N)⁻¹` the WY transform needs, in `log₂ L` matmuls.
//! - [`decay`] — the intra-chunk score matrices under a *per-channel* gate,
//!   where the decay no longer factors out of the key contraction.
//! - [`recurrent`] — the token-by-token recurrence: the definition above, and
//!   the primitive each family's `step()` decodes with.
//! - [`chunk`] — the chunkwise WY algorithm.
//! - [`path`] — [`DeltaInput`](path::DeltaInput) (what a block hands the core)
//!   and [`DeltaPath`](path::DeltaPath)
//!   (which algorithm runs, at what chunk length).
//!
//! ## Notation / dimension keys
//!
//! Tensor names carry a shape suffix, as in `burn-stack`. The "FLA" column
//! names the corresponding argument in `flash-linear-attention`.
//!
//! | Letter | Dimension | FLA | Typical |
//! |--------|-----------|-----|---------|
//! | `b` | `batch` | `B` | varies |
//! | `s` | `sequence` length | `T` | varies |
//! | `d` | `d_model` | `hidden_size` | 512 … 2048 |
//! | `h` | `nheads` | `H` | 4 … 16 |
//! | `k` | `head_k_dim` — query/key width, and the state's row rank | `K` | 64, 128 |
//! | `v` | `head_v_dim` — value width, and the state's column rank | `V` | 64 … 256 |
//! | `n` | `nchunks` = `sequence`/`chunk_len` | `NT` | varies |
//! | `l` | `chunk_len` | `BT` | 32 … 128 |
//! | `c` | `conv_kernel` | `conv_size` | 4 |
//! | `w` | `conv_dim` = 2·`nheads`·`head_k_dim` + `nheads`·`head_v_dim` | — | — |
//! | `u` | `n_householder` ([DeltaProduct](crate::delta_product)) | `num_householder` | 1 … 3 |
//!
//! Two upper-case *gate* axes appear on the delta rule's inputs: `K` is `1`
//! (one erase/forget value per head) or `head_k_dim` (one per key channel), and
//! `V` is `1` or `head_v_dim` on the write gate. See
//! [`DeltaInput`](path::DeltaInput).
//!
//! Upper-case = a *relation* of the base dimensions (offset/multiple/concat):
//! `S` is a padded `s`, `U` a multiple of `u`, and so on. Paper style (`Q, K, V, S, β`)
//! may appear in comments but never in code identifiers.

pub mod chunk;
pub mod decay;
pub mod path;
pub mod recurrent;
pub mod tri;

/// Public re-exports for the delta-rule core.
pub mod prelude {
    pub use super::path::{DeltaInput, DeltaPath, beta_gates};
    pub use super::recurrent::delta_step;
    pub use super::tri::{TriSolve, unit_lower_inverse};
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
