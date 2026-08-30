//! The blocked triangular solve behind a custom autodiff node.
//!
//! [`TriSolve::Blocked`](super::TriSolve::Blocked) is `⌈log₂ L⌉` levels of two
//! `[L, L]` matmuls each; under plain autodiff every level leaves ~4 live
//! `[batch, nchunks, nheads, L, L]` tensors on the tape, and at the shapes this
//! crate trains at that ladder *is* the training memory.
//!
//! The inverse has an exact, one-line differential:
//!
//! ```text
//!   T = (I − N)⁻¹      ⇒   dT = T dN T   ⇒   Ḡ_N = tril(Tᵀ Ḡ Tᵀ, −1)
//! ```
//!
//! so the backward needs **only `T`** — the node's own output, already alive as
//! the input of the chunk's `U`/`W` matmuls — and costs two matmuls against the
//! `3⌈log₂ L⌉` a traversed ladder costs. The ladder therefore runs *inside* the
//! node, on the inner backend, and is never recorded. This is the reference
//! kernel's own backward: `prepare_wy_repr_bwd_kernel` in
//! `fla/ops/delta_rule/wy_fast.py` is `dA ← −tril(A · tril(dA) · A, −1)`, the
//! same two matmuls under the opposite sign convention for `N`.
//!
//! Structurally this mirrors `burn-mamba`'s
//! `mamba2::ssd::serial_recalculated`: a `#[backend_extension]` trait whose
//! default body is the ordinary forward on `B`'s primitives (via [`F`]), plus
//! one `impl … for Autodiff<B>` that registers a
//! [`Backward`](burn::backend::autodiff::ops::Backward) node. Nothing here
//! reaches below Burn's portable tensor ops.
//!
//! This is the **first** node of that analogue, not the whole of it:
//! [`DeltaPath::ChunkRecalculated`](crate::delta::path::DeltaPath::ChunkRecalculated)
//! currently differs from
//! [`DeltaPath::Chunk`](crate::delta::path::DeltaPath::Chunk) only by
//! substituting this node. The remaining `[·, L, L]` tensors and the serial
//! scan's per-chunk stream still ride the tape.

use burn::backend::tensor::FloatTensor;
use burn::backend::*;
use burn::backend::{Backend, Dispatch, backend_extension};
use burn::prelude::*;
use burn_stack::utils::fprim::F;

/// [`super::unit_lower_inverse`] at [`TriSolve::Blocked`](super::TriSolve::Blocked),
/// evaluated through the backend extension so autodiff backends substitute the
/// analytic backward. Same values, same shapes.
pub fn unit_lower_inverse<const D: usize>(n_strict: Tensor<D>) -> Tensor<D> {
    let dims = n_strict.dims();
    let size = dims[D - 1];
    assert_eq!(
        dims[D - 2],
        size,
        "unit_lower_inverse expects a square trailing block"
    );
    // Rank-erase to `[flat, size, size]`: the node works on runtime shapes, and
    // `reshape` is a transparent autodiff pass-through.
    let flat: usize = dims[..D - 2].iter().product();
    let n_strict_fss = n_strict.reshape([flat, size, size]);
    let t_fss: Tensor<3> =
        Tensor::from_dispatch(<Dispatch as DeltaTriBackendExt>::unit_lower_inverse(
            n_strict_fss.into_dispatch(),
        ));
    t_fss.reshape(dims)
}

/// Extends the backend and wraps it for `burn`.
#[backend_extension(
    Cpu:  cfg(feature = "backend-cpu"),
    Cuda: cfg(feature = "backend-cuda"),
    Rocm:  cfg(feature = "backend-rocm"),
    Metal:  cfg(feature = "backend-metal"),
    Vulkan:  cfg(feature = "backend-vulkan"),
    Wgpu:  cfg(feature = "backend-wgpu"),
    WebGpu:  cfg(feature = "backend-webgpu"),
    Flex:  cfg(feature = "backend-flex"),
    NdArray:  cfg(feature = "backend-ndarray"),
    LibTorch:  cfg(any(feature = "backend-tch-cpu", feature = "backend-tch-gpu")),
    Autodiff:  cfg(feature = "autodiff"),
)]
pub trait DeltaTriBackendExt: Backend {
    /// `T = (I − N)⁻¹` for `n_strict_fss` of shape `[flat, size, size]`,
    /// strictly lower triangular.
    fn unit_lower_inverse(n_strict_fss: FloatTensor<Self>) -> FloatTensor<Self> {
        blocked_inverse::<Self>(F::<Self, 3>::new(n_strict_fss)).inner()
    }
}

/// Primitive port of the [`TriSolve::Blocked`](super::TriSolve::Blocked) ladder.
///
/// `p` holds the exact inverse of every `[block, block]` diagonal block of
/// `I − N`; each level absorbs the `C` blocks one size up via `P ← P + P X P`.
pub(crate) fn blocked_inverse<B: Backend>(n_strict_fss: F<B, 3>) -> F<B, 3> {
    let [flat, size, size2] = n_strict_fss.dims();
    assert_eq!(size, size2, "expects a square trailing block");
    let device = n_strict_fss.device();
    let dtype = n_strict_fss.dtype();

    let ones = |rows: usize, cols: usize| F::<B, 2>::full([rows, cols], 1.0, &device, dtype);
    // `eye = triu(0) − triu(1)`: `F` carries `triu` but no `tril`.
    let eye = ones(size, size).triu(0) - ones(size, size).triu(1);
    let mut p = eye.reshape([1, size, size]).expand([flat, size, size]);

    let mut lower = block_strict_lower::<B>(size, 1, &device, dtype);
    let mut block = 1usize;
    while block < size {
        let coarser = block_strict_lower::<B>(size, block * 2, &device, dtype);
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
    device: &burn::backend::tensor::Device<B>,
    dtype: burn::backend::FloatDType,
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

/// Strictly-lower-triangular part: `g − triu(g, 0)`.
pub(crate) fn strict_tril<B: Backend>(g_fss: F<B, 3>) -> F<B, 3> {
    g_fss.clone() - g_fss.triu(0)
}

burn_stack::impl_backend_ext_for_burn_backends!(DeltaTriBackendExt);

burn_stack::decl_autodiff_backend_ext!(DeltaTriAutodiffBackendExt, DeltaTriBackendExt);

/// The registered custom `Backward` node.
#[cfg(feature = "autodiff")]
mod backward {
    use super::{DeltaTriBackendExt, strict_tril};
    use burn::backend::autodiff::{
        Autodiff,
        checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
        grads::Gradients,
        ops::{Backward, Ops, OpsKind},
    };
    use burn::backend::tensor::FloatTensor;
    use burn::backend::{Backend, BackendTypes};
    use burn_stack::utils::fprim::F;

    impl<B: Backend + DeltaTriBackendExt, C: CheckpointStrategy> DeltaTriBackendExt
        for Autodiff<B, C>
    {
        fn unit_lower_inverse(n_strict_fss: FloatTensor<Self>) -> FloatTensor<Self> {
            #[derive(Debug)]
            struct TriInverseBackward;

            impl<B: Backend + DeltaTriBackendExt> Backward<B, 1> for TriInverseBackward {
                /// Only the output `T` is kept — the level ladder is never taped.
                type State = <B as BackendTypes>::FloatTensorPrimitive;

                fn backward(
                    self,
                    ops: Ops<Self::State, 1>,
                    grads: &mut Gradients,
                    _checkpointer: &mut Checkpointer,
                ) {
                    let [node_n] = ops.parents;
                    let Some(node_n) = node_n else { return };

                    let g_fss = F::<B, 3>::new(grads.consume::<B>(&ops.node));
                    let t_fss = F::<B, 3>::new(ops.state);

                    // Ḡ_N = tril(Tᵀ Ḡ Tᵀ, −1)
                    let tt = t_fss.transpose();
                    let d_n = strict_tril(tt.clone().matmul(g_fss).matmul(tt));

                    grads.register::<B>(node_n.id, d_n.inner());
                }
            }

            match TriInverseBackward
                .prepare::<C>([n_strict_fss.node.clone()])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(prep) => {
                    let t = B::unit_lower_inverse(n_strict_fss.primitive);
                    prep.finish(t.clone(), t)
                }
                OpsKind::UnTracked(prep) => {
                    prep.finish(B::unit_lower_inverse(n_strict_fss.primitive))
                }
            }
        }
    }
}
