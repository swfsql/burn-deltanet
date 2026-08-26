//! Inverting the unit lower-triangular matrix at the heart of the WY transform.
//!
//! The chunkwise delta rule needs
//!
//! ```text
//!   T = (I − N)⁻¹,     N strictly lower triangular, N[i,j] = −βᵢ (kᵢ·kⱼ) e^{Gᵢ−Gⱼ}
//! ```
//!
//! for every `[chunk_len, chunk_len]` block. `N` is **nilpotent** (`N^L = 0`),
//! so the Neumann series is not an approximation but an identity with finitely
//! many terms:
//!
//! ```text
//!   (I − N)⁻¹ = I + N + N² + … + N^{L−1}
//! ```
//!
//! The reference Triton kernel evaluates this by forward substitution — `L`
//! strictly serial row updates, cheap only because a warp does them in
//! registers. That shape is the worst possible one for portable tensor ops, so
//! this module uses the factorisation instead
//!
//! ```text
//!   Σ_{j<2^m} Nʲ = (I + N^{2^{m−1}}) ⋯ (I + N²)(I + N)
//! ```
//!
//! which reaches the same finite sum in `⌈log₂ L⌉` steps of two batched
//! matmuls each ([`TriSolve::Doubling`], the default: 12 matmuls at `L = 64`
//! instead of 64 serial row updates). [`TriSolve::Neumann`] accumulates the
//! series term by term and exists as the literal, obviously-correct reference
//! the doubling path is tested against.
//!
//! Both are exact in exact arithmetic, so they agree to floating-point
//! rounding on values *and* gradients.

use burn::prelude::*;
use burn_stack::modules::sanity as san;

/// How to invert `I − N` for a strictly-lower-triangular nilpotent `N`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum TriSolve {
    /// Repeated squaring: `⌈log₂ L⌉` steps, two matmuls each. The default.
    #[default]
    Doubling,
    /// Term-by-term Neumann accumulation: `L − 1` matmuls. The reference.
    Neumann,
}

/// `(I − N)⁻¹` for a batched strictly-lower-triangular `N` of shape
/// `[…, size, size]`.
///
/// The caller owes the strict-lower property (the delta rule's `N` is built
/// with a `tril(-1)`); nothing here re-masks it, and a non-nilpotent input
/// would silently return a truncated series.
pub fn unit_lower_inverse<const D: usize>(n_strict: Tensor<D>, solve: TriSolve) -> Tensor<D> {
    let dims = n_strict.dims();
    let size = dims[D - 1];
    assert_eq!(
        dims[D - 2],
        size,
        "unit_lower_inverse expects a square trailing block"
    );
    san(&n_strict);

    let identity: Tensor<D> = Tensor::eye(size, &n_strict.device()).unsqueeze();

    let t = match solve {
        TriSolve::Doubling => {
            // p = Σ_{j<reach} Nʲ,  m = N^reach
            let mut p = identity + n_strict.clone();
            let mut m = n_strict;
            let mut reach = 2usize;
            while reach < size {
                m = m.clone().matmul(m);
                p = p.clone() + m.clone().matmul(p);
                reach *= 2;
            }
            p
        }
        TriSolve::Neumann => {
            let mut p = identity.clone();
            let mut term = identity;
            for _ in 1..size {
                term = term.matmul(n_strict.clone());
                p = p + term.clone();
            }
            p
        }
    };
    san(&t);
    t
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
