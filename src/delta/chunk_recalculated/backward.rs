//! # The custom autodiff node
//!
//! Implements [`DeltaChunkBackendExt`] for `Autodiff<B>` by registering a single
//! Burn [`Backward`] node. The forward
//! stores only its seven leaf inputs; during backprop those are replayed through
//! the `forward` module and differentiated by
//! [`combined_backward`](super::combined_backward), so none of the chunk
//! intermediates — the two score matrices, the `⌈log₂ L⌉`-level ladder that
//! inverts `I − N`, `T`, `U`, `W`, `attn`, and the per-chunk state stream — has
//! to stay alive between the forward and the backward.
//!
//! The two forward outputs (`y` and `final_state`) are flattened into one
//! tracked 1-D tensor (via [`burn_stack::utils::combined_grad`]) so that a
//! single `Backward<B, 7>` node — one per differentiable input — covers both.

#![allow(non_snake_case)]

use burn::backend::autodiff::{
    Autodiff,
    checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients,
    ops::{Backward, Ops, OpsKind},
};
use burn::backend::tensor::FloatTensor;
use burn::backend::{Backend, BackendTypes, TensorMetadata};
use burn_stack::utils::fprim::F;

use super::chunk_recalculated::DeltaChunkBackendExt;
use super::combined_backward::{DeltaChunkGrads, combined_backward};

impl<B: Backend + DeltaChunkBackendExt, C: CheckpointStrategy> DeltaChunkBackendExt
    for Autodiff<B, C>
{
    /// Memory-efficient combined forward+backward.
    ///
    /// The two output tensors are concatenated into a single 1-dimensional
    /// tracked tensor so that one `Backward<B, 7>` node covers both outputs.
    /// The caller receives split+reshaped slices of that combined tensor;
    /// burn's autodiff accumulates their upstream gradients back into a single
    /// gradient vector before firing this backward.
    #[allow(clippy::too_many_arguments)]
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
        #[derive(Debug)]
        struct ChunkRecalculatedBackward;

        /// State carried across the forward→backward boundary: the seven
        /// original inputs and nothing else. Every intermediate is recomputed.
        #[derive(Clone, Debug)]
        struct State<B: Backend> {
            q_bshk: <B as BackendTypes>::FloatTensorPrimitive,
            k_bshk: <B as BackendTypes>::FloatTensorPrimitive,
            v_bshv: <B as BackendTypes>::FloatTensorPrimitive,
            erase_bshK: <B as BackendTypes>::FloatTensorPrimitive,
            write_bshV: <B as BackendTypes>::FloatTensorPrimitive,
            g_bshK: <B as BackendTypes>::FloatTensorPrimitive,
            state_bhkv: <B as BackendTypes>::FloatTensorPrimitive,
            has_gate: bool,
            chunk_len: usize,
            scale: f64,
            // For splitting the combined gradient vector back into the two
            // outputs.
            flat_len_y_BSHV: usize,
            flat_len_final_state_BHKV: usize,
            shape_y_bshv: [usize; 4],
            shape_final_state_bhkv: [usize; 4],
        }

        impl<B: Backend + DeltaChunkBackendExt> Backward<B, 7> for ChunkRecalculatedBackward {
            type State = State<B>;

            fn backward(
                self,
                ops: Ops<Self::State, 7>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let [
                    node_q,
                    node_k,
                    node_v,
                    node_erase,
                    node_write,
                    node_g,
                    node_state,
                ] = ops.parents;

                let d_combined = grads.consume::<B>(&ops.node);

                let State {
                    q_bshk,
                    k_bshk,
                    v_bshv,
                    erase_bshK,
                    write_bshV,
                    g_bshK,
                    state_bhkv,
                    has_gate,
                    chunk_len,
                    scale,
                    flat_len_y_BSHV,
                    flat_len_final_state_BHKV,
                    shape_y_bshv,
                    shape_final_state_bhkv,
                } = ops.state;

                let (d_y_bshv, d_final_state_bhkv) =
                    burn_stack::utils::combined_grad::unflatten_pair::<B, 4, 4>(
                        d_combined,
                        flat_len_y_BSHV,
                        flat_len_final_state_BHKV,
                        shape_y_bshv,
                        shape_final_state_bhkv,
                    );

                let DeltaChunkGrads {
                    d_q_bshk,
                    d_k_bshk,
                    d_v_bshv,
                    d_erase_bshK,
                    d_write_bshV,
                    d_g_bshK,
                    d_state_bhkv,
                } = combined_backward::<B>(
                    F::<B, 4>::new(d_y_bshv),
                    F::<B, 4>::new(d_final_state_bhkv),
                    F::<B, 4>::new(q_bshk),
                    F::<B, 4>::new(k_bshk),
                    F::<B, 4>::new(v_bshv),
                    F::<B, 4>::new(erase_bshK),
                    F::<B, 4>::new(write_bshV),
                    has_gate.then(|| F::<B, 4>::new(g_bshK)),
                    F::<B, 4>::new(state_bhkv),
                    chunk_len,
                    scale,
                );

                if let Some(n) = node_q {
                    grads.register::<B>(n.id, d_q_bshk.inner());
                }
                if let Some(n) = node_k {
                    grads.register::<B>(n.id, d_k_bshk.inner());
                }
                if let Some(n) = node_v {
                    grads.register::<B>(n.id, d_v_bshv.inner());
                }
                if let Some(n) = node_erase {
                    grads.register::<B>(n.id, d_erase_bshK.inner());
                }
                if let Some(n) = node_write {
                    grads.register::<B>(n.id, d_write_bshV.inner());
                }
                if let (Some(n), Some(d_g)) = (node_g, d_g_bshK) {
                    grads.register::<B>(n.id, d_g.inner());
                }
                if let Some(n) = node_state {
                    grads.register::<B>(n.id, d_state_bhkv.inner());
                }
            }
        }

        // ── Output shapes ──────────────────────────────────────────────────
        let [batch, sequence, nheads, head_k_dim] = q_bshk.primitive.shape().dims();
        let head_v_dim = v_bshv.primitive.shape().dims::<4>()[3];
        let shape_y_bshv = [batch, sequence, nheads, head_v_dim];
        let shape_final_state_bhkv = [batch, nheads, head_k_dim, head_v_dim];
        let flat_len_y_BSHV = batch * sequence * nheads * head_v_dim;
        let flat_len_final_state_BHKV = batch * nheads * head_k_dim * head_v_dim;

        let run_inner = |q, k, v, erase, write, g, state| {
            B::delta_chunk_recalculated(q, k, v, erase, write, g, state, has_gate, chunk_len, scale)
        };

        match ChunkRecalculatedBackward
            .prepare::<C>([
                q_bshk.node.clone(),
                k_bshk.node.clone(),
                v_bshv.node.clone(),
                erase_bshK.node.clone(),
                write_bshV.node.clone(),
                g_bshK.node.clone(),
                state_bhkv.node.clone(),
            ])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let (prim_y, prim_final_state) = run_inner(
                    q_bshk.primitive.clone(),
                    k_bshk.primitive.clone(),
                    v_bshv.primitive.clone(),
                    erase_bshK.primitive.clone(),
                    write_bshV.primitive.clone(),
                    g_bshK.primitive.clone(),
                    state_bhkv.primitive.clone(),
                );
                let (prim_combined, _, _) =
                    burn_stack::utils::combined_grad::flatten_pair::<B>(prim_y, prim_final_state);

                let state = State::<B> {
                    q_bshk: q_bshk.primitive,
                    k_bshk: k_bshk.primitive,
                    v_bshv: v_bshv.primitive,
                    erase_bshK: erase_bshK.primitive,
                    write_bshV: write_bshV.primitive,
                    g_bshK: g_bshK.primitive,
                    state_bhkv: state_bhkv.primitive,
                    has_gate,
                    chunk_len,
                    scale,
                    flat_len_y_BSHV,
                    flat_len_final_state_BHKV,
                    shape_y_bshv,
                    shape_final_state_bhkv,
                };
                let tracked_combined: FloatTensor<Autodiff<B, C>> =
                    prep.finish(state, prim_combined);

                // The narrow/reshape ops below are thin autodiff pass-throughs
                // whose backwards accumulate into the combined gradient vector
                // that `backward` above consumes.
                burn_stack::utils::combined_grad::autodiff_unflatten_pair::<B, C, 4, 4>(
                    tracked_combined,
                    flat_len_y_BSHV,
                    flat_len_final_state_BHKV,
                    shape_y_bshv,
                    shape_final_state_bhkv,
                )
            }

            OpsKind::UnTracked(prep) => {
                let (prim_y, prim_final_state) = run_inner(
                    q_bshk.primitive,
                    k_bshk.primitive,
                    v_bshv.primitive,
                    erase_bshK.primitive,
                    write_bshV.primitive,
                    g_bshK.primitive,
                    state_bhkv.primitive,
                );
                let (prim_combined, _, _) =
                    burn_stack::utils::combined_grad::flatten_pair::<B>(prim_y, prim_final_state);
                let tracked_combined: FloatTensor<Autodiff<B, C>> = prep.finish(prim_combined);

                burn_stack::utils::combined_grad::autodiff_unflatten_pair::<B, C, 4, 4>(
                    tracked_combined,
                    flat_len_y_BSHV,
                    flat_len_final_state_BHKV,
                    shape_y_bshv,
                    shape_final_state_bhkv,
                )
            }
        }
    }
}
