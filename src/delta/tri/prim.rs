//! [`TriSolve::Blocked`](super::TriSolve) on backend primitives.
//!
//! The identical ladder as [`super::unit_lower_inverse`], written against
//! [`F`] instead of the high-level `Tensor` so it can run inside a custom
//! autodiff node — where the backend is a generic `B` and the `Dispatch`-pinned
//! `Tensor` is unavailable. [`chunk_recalculated`](crate::delta::chunk_recalculated)
//! is the only caller; it runs this in its forward *and* again in its backward's
//! recompute, and never records either.
//!
//! Only the blocked form is ported. [`TriSolve::Neumann`](super::TriSolve::Neumann)
//! is a reference for the *forward* algorithm and has no role in a path whose
//! whole point is the memory of the backward.

use burn::backend::Backend;
use burn::backend::tensor::Device;
use burn::backend::FloatDType;
use burn_stack::utils::fprim::F;

/// `T = (I − N)⁻¹` for a strictly-lower-triangular `n_strict_fss`.
///
/// `p` holds the exact inverse of every `[block, block]` diagonal block of
/// `I − N`; each level absorbs the `C` blocks one size up via `P ← P + P X P`,
/// so every intermediate is an exact inverse of a principal submatrix and
/// nothing grows. See [`super`] for why the Neumann series cannot be used here.
///
/// # Shape
/// - `n_strict_fss`, returned: `[flat, size, size]`
pub fn unit_lower_inverse<B: Backend>(n_strict_fss: F<B, 3>) -> F<B, 3> {
    let [flat, size, size2] = n_strict_fss.dims();
    assert_eq!(size, size2, "expects a square trailing block");
    let device = n_strict_fss.device();
    let dtype = n_strict_fss.dtype();

    let ones = |rows: usize, cols: usize| F::<B, 2>::full([rows, cols], 1.0, &device, dtype);
    // `eye = triu(0) − triu(1)`.
    let eye = ones(size, size).triu(0) - ones(size, size).triu(1);
    let mut p = eye.reshape([1, size, size]).expand([flat, size, size]);

    let mut lower = block_strict_lower::<B>(size, 1, &device, dtype);
    let mut block = 1usize;
    while block < size {
        let coarser = block_strict_lower::<B>(size, block * 2, &device, dtype);
        // `lower − coarser` selects exactly the `C` blocks this level absorbs.
        let mask = (lower - coarser.clone()).reshape([1, size, size]);
        let x = n_strict_fss.clone() * mask.expand([flat, size, size]);
        p = p.clone() + p.clone().matmul(x).matmul(p);
        lower = coarser;
        block *= 2;
    }
    p
}

/// `[size, size]` 0/1 mask of the entries strictly below the diagonal **in
/// units of `block`** — the primitive port of `super::block_strict_lower`.
fn block_strict_lower<B: Backend>(
    size: usize,
    block: usize,
    device: &Device<B>,
    dtype: FloatDType,
) -> F<B, 2> {
    if block >= size {
        return F::<B, 2>::zeros([size, size], device, dtype);
    }
    let nblocks = size.div_ceil(block);
    let padded = nblocks * block;
    let ones = F::<B, 2>::full([nblocks, nblocks], 1.0, device, dtype);
    // `tril(-1) = ones − triu(0)`.
    let grid = ones.clone() - ones.triu(0);
    grid.reshape([nblocks, 1, nblocks, 1])
        .expand([nblocks, block, nblocks, block])
        .reshape([padded, padded])
        .narrow(0, 0, size)
        .narrow(1, 0, size)
}
