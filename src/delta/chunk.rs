//! The chunkwise WY form of the delta rule.
//!
//! The algorithm is derived in the [module header](crate::delta); this file is
//! the mechanical transcription of `chunk_delta_rule` /
//! `chunk_gated_delta_rule` / `chunk_gdn2` from `flash-linear-attention` (whose
//! `naive.py` references this follows line for line), with the gate as an
//! `Option` so the ungated family pays for no decay tensors.
//!
//! The erase/write gates ride their broadcast axes (see
//! [`DeltaInput`]), so a per-head `β` and a
//! per-channel `(b, w)` pair run the *same* expressions here. The one place the
//! width matters is the pair of intra-chunk score matrices: a per-head decay
//! factors out of the key contraction into a plain `[chunk_len, chunk_len]`
//! mask, a per-channel one does not — see [`decay`](super::decay).
//!
//! ## Padding
//!
//! The sequence is zero-padded up to a multiple of `chunk_len`. A zero pad is
//! an exact identity step here: `k = 0` writes nothing, a zero erase gate
//! corrects nothing, a zero write gate commits nothing, `g = 0` decays nothing
//! and `q = 0` reads nothing — so the final state of the padded last chunk *is*
//! the state after the last real token, and the pad rows are simply sliced off
//! the output. (This is why `q`/`k` must be L2-normalised **before** they get
//! here: normalising a zero pad row would divide by its own zero norm.)

use burn::prelude::*;
use burn_stack::modules::sanity as san;

use super::decay::BlockDecay;
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
        let channel_decay = self.has_channel_decay();
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
        let device = q_bshk.device();
        let erase_width = erase_bshK.dims()[3];
        let write_width = write_bshV.dims()[3];
        let decay_width = g_bshK.as_ref().map_or(1, |g| g.dims()[3]);

        // ── Pad to a whole number of chunks ─────────────────────────────────
        let sequence_padded = sequence.next_multiple_of(chunk_len);
        let pad = sequence_padded - sequence;
        let (q_bShk, k_bShk, v_bShv, erase_bShK, write_bShV, g_bShK) = if pad == 0 {
            (q_bshk, k_bshk, v_bshv, erase_bshK, write_bshV, g_bshK)
        } else {
            let pad_to = |t: Tensor<4>, last: usize| {
                let zeros = Tensor::<4>::zeros(Shape::new([batch, pad, nheads, last]), &device);
                Tensor::cat(vec![t, zeros], 1)
            };
            (
                pad_to(q_bshk, head_k_dim),
                pad_to(k_bshk, head_k_dim),
                pad_to(v_bshv, head_v_dim),
                pad_to(erase_bshK, erase_width),
                pad_to(write_bshV, write_width),
                g_bshK.map(|g| pad_to(g, decay_width)),
            )
        };

        // ── Reshape to [batch, nchunks, nheads, chunk_len, ·] ───────────────
        // Heads ahead of the chunk axis so every matmul below batches over
        // (batch, nchunks, nheads) and acts on the [chunk_len, ·] planes.
        let nchunks = sequence_padded / chunk_len;
        let chunked = |t: Tensor<4>, last: usize| -> Tensor<5> {
            t.reshape([batch, nchunks, chunk_len, nheads, last])
                .swap_dims(2, 3)
        };

        let q_bnhlk = chunked(q_bShk, head_k_dim) * scale;
        let k_bnhlk = chunked(k_bShk, head_k_dim);
        let v_bnhlv = chunked(v_bShv, head_v_dim);
        let erase_bnhlK = chunked(erase_bShK, erase_width);
        let write_bnhlV = chunked(write_bShV, write_width);
        let g_bnhlK = g_bShK.map(|g| chunked(g, decay_width));

        // ── Cumulative gate ─────────────────────────────────────────────────
        // `gc[t] = Σ_{j ≤ t} g[j]` (within the chunk), so `gc[i] − gc[j]` is the
        // decay carried from step `j` to step `i`. Working from differences of a
        // cumulative sum — rather than chaining products — is what keeps it
        // stable over long chunks; the same trick as `burn_stack`'s `segsum`.
        let gc_bnhlK = g_bnhlK.map(|g| g.cumsum(3));

        let k_erase_bnhlk = k_bnhlk.clone() * erase_bnhlK;
        let v_write_bnhlv = v_bnhlv * write_bnhlV;

        // ── The two intra-chunk score matrices ──────────────────────────────
        //   Akk[i, j] = Σ_d (erase ⊙ k)ᵢ,d kⱼ,d e^{Gᵢ,d−Gⱼ,d}   (strictly lower)
        //   Aqk[i, j] = Σ_d qᵢ,d kⱼ,d e^{Gᵢ,d−Gⱼ,d}             (causal)
        let (akk_bnhll, aqk_bnhll) = if channel_decay {
            let gc = gc_bnhlK.clone().expect("channel decay implies a gate");
            let decay = BlockDecay::new(k_bnhlk.clone(), gc);
            (
                decay.scores(k_erase_bnhlk.clone()),
                decay.scores(q_bnhlk.clone()),
            )
        } else {
            // A per-head decay factors out of the contraction: one shared
            // `[chunk_len, chunk_len]` mask, and one matmul per matrix.
            let mask_bnhll: Option<Tensor<5>> = gc_bnhlK.as_ref().map(|gc_bnhl1| {
                let gc_bnhl = gc_bnhl1.clone().squeeze_dim::<4>(4);
                let diff =
                    gc_bnhl.clone().unsqueeze_dim::<5>(4) - gc_bnhl.unsqueeze_dim::<5>(3);
                let causal = Tensor::full_like(&diff, f32::NEG_INFINITY).triu(1);
                let mask = (diff + causal).exp();
                san(&mask);
                mask
            });
            let kT_bnhkl = k_bnhlk.clone().swap_dims(3, 4);
            let gated = |scores: Tensor<5>| match &mask_bnhll {
                Some(mask) => scores * mask.clone(),
                None => scores,
            };
            (
                gated(k_erase_bnhlk.clone().matmul(kT_bnhkl.clone())),
                gated(q_bnhlk.clone().matmul(kT_bnhkl)),
            )
        };

        // ── The WY transform ────────────────────────────────────────────────
        //   N = −tril(Akk, −1)
        //   T = (I − N)⁻¹
        //   U = T (write ⊙ V)                 the chunk-local target updates
        //   W = T ((erase ⊙ K) ⊙ e^G)         how the chunk reads the old state
        let t_bnhll = unit_lower_inverse((-akk_bnhll).tril(-1), solve);

        let u_bnhlv = t_bnhll.clone().matmul(v_write_bnhlv);
        let w_bnhlk = {
            let k_erase = match &gc_bnhlK {
                Some(gc) => k_erase_bnhlk * gc.clone().exp(),
                None => k_erase_bnhlk,
            };
            t_bnhll.matmul(k_erase)
        };

        // The diagonal is kept in the attention because the readout
        // `yₜ = Sₜᵀ qₜ` sees the state *after* the current token's own write.
        let attn_bnhll = aqk_bnhll.tril(0);

        // ── Serial scan over chunks ─────────────────────────────────────────
        // The inter-chunk recurrence is `S ← (α ⊙ S) − Kᵀ W S + Kᵀ U`: matrix
        // valued with a rank-`chunk_len` update, so there is no scalar-decay
        // shortcut to parallelise it the way Mamba-2's chunk scan is.
        let pick = |t: &Tensor<5>, i: usize| -> Tensor<4> {
            t.clone().narrow(1, i, 1).squeeze_dim(1)
        };

        let mut state_bhkv = state_bhkv;
        let mut ys = Vec::with_capacity(nchunks);
        for i in 0..nchunks {
            let q_bhlk = pick(&q_bnhlk, i);
            let k_bhlk = pick(&k_bnhlk, i);
            let w_bhlk = pick(&w_bnhlk, i);
            let u_bhlv = pick(&u_bnhlv, i);
            let attn_bhll = pick(&attn_bnhll, i);
            let gc_bhlK = gc_bnhlK.as_ref().map(|gc| pick(gc, i));

            // The chunk's own updates, corrected for what the incoming state
            // already holds: V' = U − W S₀.
            let v_new_bhlv = u_bhlv - w_bhlk.matmul(state_bhkv.clone());

            // Output = (decayed) read of the incoming state + the intra-chunk part.
            let q_read_bhlk = match &gc_bhlK {
                Some(gc) => q_bhlk * gc.clone().exp(),
                None => q_bhlk,
            };
            let y_bhlv =
                q_read_bhlk.matmul(state_bhkv.clone()) + attn_bhll.matmul(v_new_bhlv.clone());

            // Carry the state to the chunk boundary: decay everything to the
            // last step, then add each token's write decayed from its own step.
            let (decayed_state_bhkv, k_carry_bhlk) = match &gc_bhlK {
                Some(gc_bhlK) => {
                    let gc_last_bh1K = gc_bhlK.clone().narrow(2, chunk_len - 1, 1);
                    // `[batch, nheads, 1 | head_k_dim, 1]`: the decay is on the
                    // state's key axis.
                    let decayed = state_bhkv * gc_last_bh1K.clone().swap_dims(2, 3).exp();
                    let carry = k_bhlk * (gc_last_bh1K - gc_bhlK.clone()).exp();
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
        assert_eq!([batch, nheads, head_k_dim, head_v_dim], state_bhkv.dims());
        san(&y_bshv);
        san(&state_bhkv);
        (y_bshv, state_bhkv)
    }
}
