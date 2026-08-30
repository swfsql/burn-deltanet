//! Inverting the unit lower-triangular matrix at the heart of the WY transform.
//!
//! The chunkwise delta rule needs
//!
//! ```text
//!   T = (I − N)⁻¹,     N strictly lower triangular, N[i,j] = −βᵢ (kᵢ·kⱼ) e^{Gᵢ−Gⱼ}
//! ```
//!
//! for every `[chunk_len, chunk_len]` block. `N` is **nilpotent** (`N^L = 0`),
//! so the Neumann series `I + N + N² + … + N^{L−1}` is not an approximation but
//! an identity — and it is nonetheless the wrong way to evaluate it. When the
//! keys inside a chunk point the same way (a constant input: an image
//! background, a run of padding) `N` is `−β` on the whole strict lower
//! triangle, `‖Nʲ‖` peaks around `C(L−2, L/2)` — `10¹⁷` at `L = 64` — and the
//! terms cancel back down to a `T` whose entries never exceed `β`. In float32
//! nothing survives that cancellation, and the block's state diverges to `NaN`
//! within a couple of optimiser steps.
//!
//! [`TriSolve::Blocked`] (the default) never forms a power of `N`. It inverts
//! `I − N` the way a blocked forward substitution does, doubling the block size
//! each step:
//!
//! ```text
//!   ⎡A  0⎤⁻¹   ⎡  A⁻¹      0  ⎤
//!   ⎣C  B⎦   = ⎣−B⁻¹CA⁻¹  B⁻¹ ⎦
//! ```
//!
//! With `P` the block-diagonal matrix of the level's inverses and `X` the
//! `C` blocks the level is about to absorb (`X = mask ⊙ N`, one mask per
//! level), that whole level is `P ← P + P X P` — two batched matmuls, and the
//! same `⌈log₂ L⌉` steps the series factorisation took. Every intermediate is
//! an exact inverse of a principal submatrix of `I − N`, so nothing grows.
//!
//! The reference Triton kernel does the same substitution, one row at a time —
//! `L` strictly serial register updates, the worst possible shape for portable
//! tensor ops. [`TriSolve::Neumann`] accumulates the series term by term and
//! exists as the literal, obviously-correct reference the blocked path is
//! tested against at the small sizes where the series is still conditioned.

use burn::prelude::*;
use burn_stack::modules::sanity as san;

/// How to invert `I − N` for a strictly-lower-triangular nilpotent `N`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum TriSolve {
    /// Blocked forward substitution: `⌈log₂ L⌉ `steps, two matmuls each. Exact
    /// for any `N`, and the only one of the two that is usable past `L ≈ 16`.
    /// The default.
    #[default]
    Blocked,
    /// Term-by-term Neumann accumulation: `L − 1` matmuls. Exact in exact
    /// arithmetic, but its partial sums are catastrophically larger than their
    /// own limit as soon as the chunk's keys correlate — a reference, not a
    /// production path (see the [module header](self)).
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

    let device = n_strict.device();
    let identity: Tensor<D> = Tensor::eye(size, &device).unsqueeze();

    let t = match solve {
        TriSolve::Blocked => {
            // `p` holds the exact inverse of every `[block, block]` diagonal
            // block of `I − N`; `lower` is that level's block-lower mask, so
            // `lower − coarser` selects exactly the `C` blocks being absorbed.
            let mut p = identity;
            let mut lower = block_strict_lower(size, 1, &device);
            let mut block = 1usize;
            while block < size {
                let coarser = block_strict_lower(size, block * 2, &device);
                let x: Tensor<D> = n_strict.clone() * (lower - coarser.clone()).unsqueeze();
                p = p.clone() + p.clone().matmul(x).matmul(p);
                lower = coarser;
                block *= 2;
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

/// `[size, size]` 0/1 mask of the entries strictly below the diagonal **in
/// units of `block`**: `1` where `⌊i/block⌋ > ⌊j/block⌋`.
///
/// Built by expanding a `tril(-1)` of the block grid, so the difference of two
/// consecutive levels (`block` and `2·block`) is the set of lower-left corner
/// blocks that the level's merge fills in.
fn block_strict_lower(size: usize, block: usize, device: &Device) -> Tensor<2> {
    if block >= size {
        return Tensor::zeros([size, size], device);
    }
    let nblocks = size.div_ceil(block);
    let padded = nblocks * block;
    Tensor::<2>::ones([nblocks, nblocks], device)
        .tril(-1)
        // Each grid entry becomes a `[block, block]` constant tile.
        .reshape([nblocks, 1, nblocks, 1])
        .expand([nblocks, block, nblocks, block])
        .reshape([padded, padded])
        .narrow(0, 0, size)
        .narrow(1, 0, size)
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
