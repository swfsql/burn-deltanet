//! Query/key activation and normalisation applied before the delta rule.
//!
//! The delta rule's transition is the generalised Householder
//! `I − β k kᵀ`. Its eigenvalues are `1` (on `k`'s orthogonal complement) and
//! `1 − β‖k‖²` (along `k`), so the *scale* of `k` — not just `β` — decides
//! whether the state update is a contraction. Forcing `‖k‖ = 1` moves that
//! decision entirely onto `β`: the transition then has spectrum
//! `{1, 1 − β}`, non-expansive for `β ∈ [0, 2]`, an exact reflection at
//! `β = 2`, and the identity at `β = 0`. That is why [`QkNorm::L2`] is the
//! default and why `allow_neg_eigval` (`β ∈ (0, 2)`) is a meaningful knob at
//! all rather than a divergence.

use burn::prelude::*;
use burn_stack::modules::Silu;

/// The element-wise activation applied to `q`/`k` before normalisation.
///
/// With the short convolution enabled the SiLU is folded into the convolution
/// (that is what the reference does), so [`QkActivation::Identity`] is what a
/// block passes here in the common case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum QkActivation {
    /// No activation — the default when the short convolution already applied
    /// its own SiLU.
    #[default]
    Identity,
    /// `x · σ(x)`.
    Silu,
    /// `max(x, 0)`.
    Relu,
    /// `elu(x) + 1`, i.e. the strictly-positive feature map of the original
    /// linear-transformer line of work.
    EluPlusOne,
}

impl QkActivation {
    /// Apply the activation element-wise.
    pub fn apply<const D: usize>(self, x: Tensor<D>) -> Tensor<D> {
        match self {
            Self::Identity => x,
            Self::Silu => Silu::new().forward(x),
            Self::Relu => burn::tensor::activation::relu(x),
            // elu(x) + 1 = x + 1 for x ≥ 0, exp(x) for x < 0 — kept branch-free.
            Self::EluPlusOne => {
                let positive = x.clone().clamp_min(0.0) + 1.0;
                let negative = x.clone().clamp_max(0.0).exp();
                let is_negative = x.lower_elem(0.0);
                positive.mask_where(is_negative, negative)
            }
        }
    }
}

/// The normalisation applied to `q`/`k` over the head dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum QkNorm {
    /// `x / ‖x‖₂` — the default; see the module header for why.
    #[default]
    L2,
    /// `x / Σx` — only meaningful with a strictly-positive activation
    /// ([`QkActivation::Relu`] / [`QkActivation::EluPlusOne`]).
    Sum,
    /// Leave `q`/`k` as projected.
    None,
}

impl QkNorm {
    /// Apply the normalisation over the last (head) dimension.
    pub fn apply<const D: usize>(self, x: Tensor<D>) -> Tensor<D> {
        match self {
            Self::L2 => l2_normalize(x),
            Self::Sum => sum_normalize(x),
            Self::None => x,
        }
    }
}

/// The epsilon guarding the L2 denominator, matching the reference kernel.
///
/// It is an *additive* guard inside the square root (`1/√(Σx² + ε)`), so a
/// genuinely zero row — a padded position, say — maps to zero rather than
/// `NaN`, while a unit-norm row is unaffected to well under f16 resolution.
pub const L2_EPS: f64 = 1e-6;

/// `x / √(Σ x² + ε)` over the last dimension.
pub fn l2_normalize<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    let sq_sum = x.clone().powi_scalar(2).sum_dim(D - 1);
    let inv_norm = (sq_sum + L2_EPS).sqrt().recip();
    x * inv_norm
}

/// `x / Σ x` over the last dimension (assumes a strictly-positive `x`).
pub fn sum_normalize<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    let dtype = x.dtype();
    let denom = x.clone().sum_dim(D - 1) + burn_stack::utils::div_eps(dtype) as f64;
    x / denom
}
