//! The chunkwise WY form of the delta rule.
//!
//! The algorithm is derived in the [module header](crate::delta); this file is
//! the mechanical transcription of `chunk_delta_rule` /
//! `chunk_gated_delta_rule` from `flash-linear-attention` (whose `naive.py`
//! reference this follows line for line), with the gate as an `Option` so the
//! ungated family pays for no decay tensors.
//!
//! ## Padding
//!
//! The sequence is zero-padded up to a multiple of `chunk_len`. A zero pad is
//! an exact identity step here: `k = 0` writes nothing, `β = 0` corrects
//! nothing, `g = 0` decays nothing and `q = 0` reads nothing — so the final
//! state of the padded last chunk *is* the state after the last real token, and
//! the pad rows are simply sliced off the output. (This is why `q`/`k` must be
//! L2-normalised **before** they get here: normalising a zero pad row would
//! divide by its own zero norm.)

use burn::prelude::*;
use burn_stack::modules::sanity as san;

use super::path::DeltaInput;
use super::tri::{TriSolve, unit_lower_inverse};

impl DeltaInput {
    /// Chunkwise WY evaluation of the delta rule.
    ///
    /// Serial over chunks, batched-GEMM within each. See
    /// [`DeltaInput::run`] for the returned shapes.
    #[allow(non_snake_case)]
    pub fn delta_chunk(self, chunk_len: usize, solve: TriSolve) -> (Tensor<4>, Tensor<4>) {
        let (batch, sequence, nheads, head_k_dim, head_v_dim) = self.dims();
        let scale = self.resolved_scale();
        let DeltaInput {
            q_bshk,
            k_bshk,
            v_bshv,
            beta_bsh,
            g_bsh,
            state_bhkv,
            scale: _,
        } = self;
        let device = q_bshk.device();

        // ── Pad to a whole number of chunks ─────────────────────────────────
        let sequence_padded = sequence.next_multiple_of(chunk_len);
        let pad = sequence_padded - sequence;
        let (q_bShk, k_bShk, v_bShv, beta_bSh, g_bSh) = if pad == 0 {
            (q_bshk, k_bshk, v_bshv, beta_bsh, g_bsh)
        } else {
            let zeros4 = |last: usize| {
                Tensor::<4>::zeros(Shape::new([batch, pad, nheads, last]), &device)
            };
            let zeros3 = Tensor::<3>::zeros(Shape::new([batch, pad, nheads]), &device);
            (
                Tensor::cat(vec![q_bshk, zeros4(head_k_dim)], 1),
                Tensor::cat(vec![k_bshk, zeros4(head_k_dim)], 1),
                Tensor::cat(vec![v_bshv, zeros4(head_v_dim)], 1),
                Tensor::cat(vec![beta_bsh, zeros3.clone()], 1),
                g_bsh.map(|g| Tensor::cat(vec![g, zeros3], 1)),
            )
        };

        // ── Reshape to [batch, nchunks, nheads, chunk_len, ·] ───────────────
        // Heads ahead of the chunk axis so every matmul below batches over
        // (batch, nchunks, nheads) and acts on the [chunk_len, ·] planes.
        let nchunks = sequence_padded / chunk_len;
        let chunked4 = |t: Tensor<4>, last: usize| -> Tensor<5> {
            t.reshape([batch, nchunks, chunk_len, nheads, last])
                .swap_dims(2, 3)
        };
        let chunked3 = |t: Tensor<3>| -> Tensor<4> {
            t.reshape([batch, nchunks, chunk_len, nheads])
                .swap_dims(2, 3)
        };

        let q_bnhlk = chunked4(q_bShk, head_k_dim) * scale;
        let k_bnhlk = chunked4(k_bShk, head_k_dim);
        let v_bnhlv = chunked4(v_bShv, head_v_dim);
        let beta_bnhl = chunked3(beta_bSh);
        let g_bnhl = g_bSh.map(chunked3);

        // ── Cumulative gate and the intra-chunk decay mask ──────────────────
        // `gc[t] = Σ_{j ≤ t} g[j]` (within the chunk), so
        // `decay[i, j] = exp(gc[i] − gc[j])` is the decay carried from step `j`
        // to step `i`, and zero above the diagonal. Working from differences of
        // a cumulative sum — rather than chaining products — is what keeps it
        // stable over long chunks; the same trick as `burn_stack`'s `segsum`.
        let gc_bnhl = g_bnhl.map(|g| g.cumsum(3));
        let decay_mask_bnhll: Option<Tensor<5>> = gc_bnhl.as_ref().map(|gc_bnhl| {
            let diff = gc_bnhl.clone().unsqueeze_dim::<5>(4) - gc_bnhl.clone().unsqueeze_dim::<5>(3);
            let causal = Tensor::full_like(&diff, f32::NEG_INFINITY).triu(1);
            let mask = (diff + causal).exp();
            san(&mask);
            mask
        });

        let beta_bnhl1 = beta_bnhl.unsqueeze_dim::<5>(4);
        let k_beta_bnhlk = k_bnhlk.clone() * beta_bnhl1.clone();
        let v_beta_bnhlv = v_bnhlv * beta_bnhl1;

        // ── The WY transform ────────────────────────────────────────────────
        //   N[i, j] = −βᵢ (kᵢ·kⱼ) e^{Gᵢ−Gⱼ}   (strictly lower)
        //   T       = (I − N)⁻¹
        //   U       = T (diag(β) V)            the chunk-local target updates
        //   W       = T (diag(β) K ⊙ e^G)      how the chunk reads the old state
        let n_bnhll = {
            let kk = k_beta_bnhlk
                .clone()
                .matmul(k_bnhlk.clone().swap_dims(3, 4));
            let kk = match &decay_mask_bnhll {
                Some(mask) => kk * mask.clone(),
                None => kk,
            };
            (-kk).tril(-1)
        };
        let t_bnhll = unit_lower_inverse(n_bnhll, solve);

        let u_bnhlv = t_bnhll.clone().matmul(v_beta_bnhlv);
        let w_bnhlk = {
            let k_beta = match &gc_bnhl {
                Some(gc) => k_beta_bnhlk * gc.clone().exp().unsqueeze_dim::<5>(4),
                None => k_beta_bnhlk,
            };
            t_bnhll.matmul(k_beta)
        };

        // ── Intra-chunk attention: tril(Q Kᵀ ⊙ decay), diagonal included ────
        // The diagonal is kept because the readout `yₜ = Sₜᵀ qₜ` sees the state
        // *after* the current token's own write.
        let attn_bnhll = {
            let qk = q_bnhlk.clone().matmul(k_bnhlk.clone().swap_dims(3, 4));
            let qk = match &decay_mask_bnhll {
                Some(mask) => qk * mask.clone(),
                None => qk,
            };
            qk.tril(0)
        };

        // ── Serial scan over chunks ─────────────────────────────────────────
        // The inter-chunk recurrence is `S ← (α I − Kᵀ W) S + Kᵀ U`: matrix
        // valued with a rank-`chunk_len` update, so there is no scalar-decay
        // shortcut to parallelise it the way Mamba-2's chunk scan is.
        let pick5 = |t: &Tensor<5>, i: usize| -> Tensor<4> { t.clone().narrow(1, i, 1).squeeze_dim(1) };
        let pick4 = |t: &Tensor<4>, i: usize| -> Tensor<3> { t.clone().narrow(1, i, 1).squeeze_dim(1) };

        let mut state_bhkv = state_bhkv;
        let mut ys = Vec::with_capacity(nchunks);
        for i in 0..nchunks {
            let q_bhlk = pick5(&q_bnhlk, i);
            let k_bhlk = pick5(&k_bnhlk, i);
            let w_bhlk = pick5(&w_bnhlk, i);
            let u_bhlv = pick5(&u_bnhlv, i);
            let attn_bhll = pick5(&attn_bnhll, i);
            let gc_bhl = gc_bnhl.as_ref().map(|gc| pick4(gc, i));

            // The chunk's own updates, corrected for what the incoming state
            // already holds: V' = U − W S₀.
            let v_new_bhlv = u_bhlv - w_bhlk.matmul(state_bhkv.clone());

            // Output = (decayed) read of the incoming state + the intra-chunk part.
            let q_read_bhlk = match &gc_bhl {
                Some(gc) => q_bhlk * gc.clone().exp().unsqueeze_dim::<4>(3),
                None => q_bhlk,
            };
            let y_bhlv = q_read_bhlk.matmul(state_bhkv.clone())
                + attn_bhll.matmul(v_new_bhlv.clone());

            // Carry the state to the chunk boundary: decay everything to the
            // last step, then add each token's write decayed from its own step.
            let (decayed_state_bhkv, k_carry_bhlk) = match &gc_bhl {
                Some(gc_bhl) => {
                    let gc_last_bh1 = gc_bhl.clone().narrow(2, chunk_len - 1, 1);
                    let decayed = state_bhkv * gc_last_bh1.clone().exp().unsqueeze_dim::<4>(3);
                    let carry =
                        k_bhlk * (gc_last_bh1 - gc_bhl.clone()).exp().unsqueeze_dim::<4>(3);
                    (decayed, carry)
                }
                None => (state_bhkv, k_bhlk),
            };
            state_bhkv = decayed_state_bhkv + k_carry_bhlk.swap_dims(2, 3).matmul(v_new_bhlv);

            ys.push(y_bhlv.unsqueeze_dim::<5>(1));
        }

        // ── Back to [batch, sequence, nheads, head_v_dim], pad removed ──────
        let y_bShv = Tensor::cat(ys, 1)
            .swap_dims(2, 3)
            .reshape([batch, sequence_padded, nheads, head_v_dim]);
        let y_bshv = if pad == 0 {
            y_bShv
        } else {
            y_bShv.narrow(1, 0, sequence)
        };
        assert_eq!([batch, sequence, nheads, head_v_dim], y_bshv.dims());
        assert_eq!(
            [batch, nheads, head_k_dim, head_v_dim],
            state_bhkv.dims()
        );
        san(&y_bshv);
        san(&state_bhkv);
        (y_bshv, state_bhkv)
    }
}
