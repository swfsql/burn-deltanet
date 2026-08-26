//! # burn-deltanet — DeltaNet / Gated DeltaNet / DeltaProduct on Burn
//!
//! A minimal, readable reference implementation of the **delta rule** family of
//! linear-attention architectures on top of the
//! [Burn](https://github.com/tracel-ai/burn/) deep learning framework:
//!
//! - [DeltaNet](https://arxiv.org/abs/2406.06484) — *Parallelizing Linear
//!   Transformers with the Delta Rule over Sequence Length* (which itself
//!   parallelises [*Linear Transformers Are Secretly Fast Weight
//!   Programmers*](https://arxiv.org/abs/2102.11174)).
//! - [Gated DeltaNet](https://arxiv.org/abs/2412.06464) — *Gated Delta Networks:
//!   Improving Mamba2 with Delta Rule*.
//! - [DeltaProduct](https://arxiv.org/abs/2502.10297) — *DeltaProduct:
//!   Improving State-Tracking in Linear RNNs via Householder Products*.
//!
//! The goal is clarity: the official Triton kernels
//! ([`flash-linear-attention`](https://github.com/fla-org/flash-linear-attention))
//! are ported down to standard, portable Burn tensor operations, so the same
//! code runs on every backend (CPU, WGPU, CUDA, Metal, LibTorch, …).  There are
//! **no custom kernels**.
//!
//! ## What the delta rule is
//!
//! Every family here writes to an associative memory `S ∈ ℝ^{k×v}` — the same
//! fixed-size recurrent state a linear-attention model carries — but with an
//! *error-correcting* write instead of a pure accumulation:
//!
//! ```text
//!   Sₜ = αₜ (I − βₜ kₜ kₜᵀ) Sₜ₋₁ + βₜ kₜ vₜᵀ        (delta rule)
//!   yₜ = Sₜᵀ qₜ                                      (readout)
//! ```
//!
//! Reading the update out token-wise: the model first *retrieves* what it
//! currently associates with `kₜ`, then moves that association a fraction `βₜ`
//! of the way towards `vₜ`. The transition is therefore a **generalised
//! Householder** matrix rather than a scalar decay — that is exactly the
//! difference from a Mamba-2/SSD-style state update, and what buys the
//! associative-recall and state-tracking behaviour the papers report.
//!
//! - `βₜ ∈ (0, 1)` (or `(0, 2)` when negative eigenvalues are allowed) is the
//!   per-head write strength.
//! - `αₜ ∈ (0, 1]` is the optional scalar forget gate — `αₜ ≡ 1` is DeltaNet,
//!   `αₜ = exp(Δₜ A)` is Gated DeltaNet.
//! - `qₜ`, `kₜ` are (by default) L2-normalised, which is what keeps the
//!   Householder factor non-expansive.
//!
//! [`delta`] holds that recurrence and its chunkwise (WY-representation)
//! reformulation once; each family is a different way of *producing*
//! `(q, k, v, β, α)` from a token stream.
//!
//! ## Module families
//!
//! Each family lives in its own module and follows the same composition
//! (`Network` → `Layers` → `Layer` → `Block`):
//!
//! - [`deltanet`] — DeltaNet: no forget gate (`α ≡ 1`).
//! - [`gated_deltanet`] — Gated DeltaNet: adds the Mamba-2-style scalar decay
//!   `αₜ = exp(Δₜ A)`.
//! - [`delta_product`] — DeltaProduct: `n_householder` delta-rule micro-steps
//!   per token, i.e. a *product* of Householder transitions per transition.
//!
//! Everything *around* the block — the Pre-LN [`Layer`](burn_stack::modules::Layer),
//! the (virtual-)layer [`Layers`](burn_stack::modules::Layers) stack,
//! bidirectional pairs, latent/vocab networks, multi-gate residuals, class
//! tokens, LR/virtual-layer scheduling and the Muon parameter groups — lives in
//! the block-agnostic [`burn_stack`] crate. This crate supplies the
//! [`Block`](burn_stack::modules::Block) implementations and, in [`unified`],
//! the runtime-selectable enums that pick a family at run time.
//!
//! ## Two execution modes
//!
//! Every block, layer, and network exposes both a parallel `forward()` (used
//! for training and prompt prefill) and a recurrent `step()` (used for
//! token-by-token decoding).  The two are mathematically equivalent: a
//! `forward()` over a sequence equals unrolling `step()` token by token from the
//! same initial cache — a parity property the test suites assert on outputs,
//! final cache, and gradients.

#![warn(missing_docs)]
#![allow(clippy::let_and_return)]
#![allow(clippy::module_inception)]
#![allow(clippy::too_many_arguments)]

/// Shared block-level primitives: the fused causal short convolution and the
/// query/key normalisations every family applies before the delta rule.
pub mod common;
/// The delta-rule core: the recurrence, its chunkwise (WY) reformulation, and
/// the algorithm selector shared by all three families.
pub mod delta;

/// DeltaProduct: `n_householder` delta-rule micro-steps per token.
#[cfg(feature = "delta-product")]
pub mod delta_product;
/// DeltaNet: the delta rule with no forget gate.
#[cfg(feature = "deltanet")]
pub mod deltanet;
/// Gated DeltaNet: the delta rule plus a Mamba-2-style scalar decay.
#[cfg(feature = "gated-deltanet")]
pub mod gated_deltanet;

/// The runtime-selectable API: enums that pick a family at run time, plus the
/// `Block`/`BlockConfig`/`CacheStack` impls that plug each one into
/// [`burn_stack`].
pub mod unified;

/// Convenience re-exports: `use burn_deltanet::prelude::*;` brings the enabled
/// model families and their public types into scope.
pub mod prelude {
    pub use crate::common::prelude::*;
    pub use crate::delta::prelude::*;

    #[cfg(feature = "deltanet")]
    pub use crate::deltanet::{self, prelude::*};

    #[cfg(feature = "gated-deltanet")]
    pub use crate::gated_deltanet::{self, prelude::*};

    #[cfg(feature = "delta-product")]
    pub use crate::delta_product::{self, prelude::*};

    // The runtime-selectable unified API (this crate).
    pub use crate::unified::DeltaCaches;

    // The block-generic composition layer (`burn-stack`).
    pub use burn_stack::prelude::*;
}

/// Re-export of the block-generic composition crate this one builds on, so a
/// dependent can reach `Layer`/`Layers`/networks without naming it separately.
pub use burn_stack;
