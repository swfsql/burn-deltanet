//! # Chunkwise WY with a custom, memory-efficient backward
//!
//! This is the `ChunkRecalculated` path. The forward is the same chunkwise WY
//! algorithm as [`chunk`](crate::delta::chunk), but it is routed through the
//! [`DeltaChunkBackendExt`] trait so that `Autodiff` backends can substitute a
//! **custom backward** that recomputes the chunk intermediates instead of
//! storing them (see [`backward`](super::backward) /
//! [`combined_backward`](super::combined_backward)).
//!
//! Every plain (non-autodiff) backend uses the trait's default body, which
//! replays the three `forward` stages. The
//! [`burn_stack::impl_backend_ext_for_burn_backends!`] /
//! [`burn_stack::decl_autodiff_backend_ext!`] macros wire up the per-backend
//! impls and the autodiff marker trait.

#![allow(non_snake_case)]

use burn::backend::tensor::FloatTensor;
use burn::backend::*;
use burn::backend::{Backend, Dispatch, backend_extension};
use burn::prelude::*;
use burn_stack::modules::sanity as san;
use burn_stack::utils::fprim::F;

use super::forward::{Chunked, ScanMode, unchunked};
use crate::delta::path::DeltaInput;

impl DeltaInput {
    /// Chunkwise WY evaluation of the delta rule, with the custom backward.
    ///
    /// Values are identical to [`DeltaInput::delta_chunk`] at
    /// [`TriSolve::Blocked`](crate::delta::tri::TriSolve::Blocked); what differs
    /// is what reaches the autodiff tape — here, only the seven leaf inputs.
    /// See [`DeltaInput::run`] for the returned shapes.
    #[allow(non_snake_case)]
    pub fn delta_chunk_recalculated(self, chunk_len: usize) -> (Tensor<4>, Tensor<4>) {
        let (batch, sequence, nheads, head_k_dim, head_v_dim) = self.dims();
        let scale = self.resolved_scale();
        let DeltaInput {
            q_bshk,
            k_bshk,
            v_bshv,
            erase_bshK,
            write_bshV,
            g_bshK,
            state_bhkv,
            scale: _,
        } = self;

        // The gate is optional, but the node's parent list is not. An absent
        // gate is passed as a `[1, 1, 1, 1]` constant that the op is told
        // (`has_gate`) never to read: it is untracked, so it is not a parent and
        // takes no gradient, and nothing of the ungated path's cost changes.
        let has_gate = g_bshK.is_some();
        let g_bshK = g_bshK
            .unwrap_or_else(|| Tensor::zeros(Shape::new([1, 1, 1, 1]), &q_bshk.device()));

        let (y_bshv, final_state_bhkv) =
            <Dispatch as DeltaChunkBackendExt>::delta_chunk_recalculated(
                q_bshk.into_dispatch(),
                k_bshk.into_dispatch(),
                v_bshv.into_dispatch(),
                erase_bshK.into_dispatch(),
                write_bshV.into_dispatch(),
                g_bshK.into_dispatch(),
                state_bhkv.into_dispatch(),
                has_gate,
                chunk_len,
                scale,
            );
        let y_bshv = Tensor::<4>::from_dispatch(y_bshv);
        let final_state_bhkv = Tensor::<4>::from_dispatch(final_state_bhkv);

        assert_eq!([batch, sequence, nheads, head_v_dim], y_bshv.dims());
        assert_eq!(
            [batch, nheads, head_k_dim, head_v_dim],
            final_state_bhkv.dims()
        );
        san(&y_bshv);
        san(&final_state_bhkv);
        (y_bshv, final_state_bhkv)
    }
}

/// Extends the backend and wraps it for `burn`.
#[backend_extension(
    // Every cubecl runtime — CUDA, ROCm, Metal, Vulkan, WebGPU, wgpu, CPU — is
    // this one backend; which of them a tensor runs on is what its device says.
    // The cfg mirrors burn's own `cube_backend`.
    Cube: cfg(any(
        feature = "backend-cpu",
        feature = "backend-cuda",
        feature = "backend-rocm",
        feature = "backend-metal",
        feature = "backend-vulkan",
        feature = "backend-wgpu",
        feature = "backend-webgpu"
    )),
    Flex:  cfg(feature = "backend-flex"),
    NdArray:  cfg(feature = "backend-ndarray"),
    LibTorch:  cfg(any(feature = "backend-tch-cpu", feature = "backend-tch-gpu")),
    Remote:  cfg(feature = "backend-remote"),
    Autodiff:  cfg(feature = "autodiff"),
)]
pub trait DeltaChunkBackendExt: Backend {
    /// The chunkwise WY delta rule.
    ///
    /// `g_bshK` is read only when `has_gate`; otherwise it is a placeholder
    /// standing in for `α ≡ 1`.
    ///
    /// # Returns
    /// - `y_bshv`: `[batch, sequence, nheads, head_v_dim]`
    /// - `final_state_bhkv`: `[batch, nheads, head_k_dim, head_v_dim]`
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn delta_chunk_recalculated(
        q_bshk: FloatTensor<Self>,
        k_bshk: FloatTensor<Self>,
        v_bshv: FloatTensor<Self>,
        erase_bshK: FloatTensor<Self>,
        write_bshV: FloatTensor<Self>,
        g_bshK: FloatTensor<Self>,
        state_bhkv: FloatTensor<Self>,
        has_gate: bool,
        chunk_len: usize,
        scale: f64,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        // Default impl: the three `forward` stages on `B`'s primitives. This
        // body runs under a generic `B`, where the `Dispatch`-pinned `Tensor`
        // is unavailable, so the math goes through the rank-tagged `F` wrapper.
        let (y_bnhlv, final_state_bhkv, dims) = forward_prim::<Self>(
            F::new(q_bshk),
            F::new(k_bshk),
            F::new(v_bshv),
            F::new(erase_bshK),
            F::new(write_bshV),
            has_gate.then(|| F::new(g_bshK)),
            F::new(state_bhkv),
            chunk_len,
            scale,
        );

        // Back to `[batch, sequence, nheads, head_v_dim]`, pad removed.
        let y_bShv = unchunked(y_bnhlv);
        let y_bshv = if dims.sequence_padded == dims.sequence {
            y_bShv
        } else {
            y_bShv.narrow(1, 0, dims.sequence)
        };
        (y_bshv.inner(), final_state_bhkv.inner())
    }
}

/// The forward, up to the chunk layout: shared verbatim by the trait's default
/// body and by the backward's recompute.
#[allow(non_snake_case, clippy::too_many_arguments)]
fn forward_prim<B: Backend>(
    q_bshk: F<B, 4>,
    k_bshk: F<B, 4>,
    v_bshv: F<B, 4>,
    erase_bshK: F<B, 4>,
    write_bshV: F<B, 4>,
    g_bshK: Option<F<B, 4>>,
    state_bhkv: F<B, 4>,
    chunk_len: usize,
    scale: f64,
) -> (F<B, 5>, F<B, 4>, super::forward::Dims) {
    let chunked = Chunked::<B>::prepare(
        q_bshk, k_bshk, v_bshv, erase_bshK, write_bshV, g_bshK, chunk_len, scale,
    );
    let wy = chunked.wy();
    let scan = chunked.scan(&wy, state_bhkv, ScanMode::Outputs);
    (
        scan.y_bnhlv.expect("ScanMode::Outputs builds y"),
        scan.final_state_bhkv,
        chunked.dims,
    )
}

burn_stack::impl_backend_ext_for_burn_backends!(DeltaChunkBackendExt);

burn_stack::decl_autodiff_backend_ext!(DeltaChunkAutodiffBackendExt, DeltaChunkBackendExt);
