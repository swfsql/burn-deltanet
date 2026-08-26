//! # Shared block-level primitives
//!
//! The two pieces every family in this crate puts *in front of* the delta rule:
//!
//! - [`conv`] — the fused causal depthwise short convolution over the
//!   concatenated `[q | k | v]` channels, with the rolling window a decode step
//!   needs. The reference implementation runs three separate `ShortConvolution`
//!   modules; a depthwise convolution is per-channel independent, so one
//!   convolution over the concatenation is the same function with one kernel
//!   launch, one weight and one cache tensor.
//! - [`norm`] — the query/key activation and normalisation. The L2 default is
//!   load-bearing rather than cosmetic: it bounds `‖k‖ = 1`, which is what keeps
//!   the delta rule's Householder factor `I − β k kᵀ` non-expansive for
//!   `β ∈ (0, 2)` — see [`crate::delta`].

/// The fused causal depthwise short convolution (+ its rolling window).
pub mod conv;
/// Query/key activation and normalisation applied before the delta rule.
pub mod norm;

/// Public re-exports for the shared primitives.
pub mod prelude {
    pub use super::conv::{ConvActivation, ShortConv, ShortConvConfig};
    pub use super::norm::{QkActivation, QkNorm, l2_normalize, sum_normalize};
}
