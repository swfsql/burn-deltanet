//! The chunkwise WY backward, written out by hand.
//!
//! [`combined_backward`] replays the `forward` module's three stages
//! from the seven leaf inputs and then walks them back. Nothing the forward
//! built is stored across the node boundary; the recompute is two matmuls per
//! chunk plus the `⌈log₂ L⌉`-level ladder, against the `~30`
//! `[batch, nchunks, nheads, chunk_len, chunk_len]` tensors autodiff would
//! otherwise keep alive per tracked layer.
//!
//! ## The pieces
//!
//! Reading the forward as
//!
//! ```text
//!   Ke = K ⊙ erase          Vw = V ⊙ write          Eg = e^G
//!   (Akk, Aqk) = scores(Ke), scores(Q)              N  = tril(−Akk, −1)
//!   T = (I − N)⁻¹           U = T Vw                W  = T (Ke ⊙ Eg)
//!   attn = tril(Aqk, 0)     then the serial scan
//! ```
//!
//! every arrow below is the ordinary transpose-and-multiply, with three that
//! are worth naming:
//!
//! - **The inverse.** `dT = T dN T`, so `Ḡ_N = tril(Tᵀ Ḡ_T Tᵀ, −1)` — two
//!   matmuls reading only `T`, never the ladder that produced it. This is the
//!   reference kernel's own backward (`prepare_wy_repr_bwd_kernel` in
//!   `fla/ops/delta_rule/wy_fast.py` is `dA ← −tril(A · tril(dA) · A, −1)`, the
//!   opposite sign convention for `N`).
//! - **The scan.** Walked in reverse, carrying `dS`; the state entering each
//!   chunk comes from the recomputed stream, because running the recurrence
//!   backwards would divide by `e^G` and that is unbounded.
//! - **The gate.** `G` is a cumulative sum, so its gradient is a *reverse*
//!   cumulative sum; every use of `G` (the score mask or the block-decay
//!   factors, `W`'s `e^G`, the readout's `e^G`, the chunk-boundary carry)
//!   accumulates into `Ḡ` first, and the flip happens once at the end.

#![allow(non_snake_case)]

use burn::backend::Backend;
use burn_stack::utils::fprim::F;

use super::forward::{
    BlockDecayPrim, Chunked, Dims, ScanMode, Wy, chunked, pad_sequence, pick, reduce_to_width,
    unchunked,
};
use crate::delta::decay::{MAX_EXPONENT, block_len};

/// The gradient of every input [`combined_backward`] is given.
#[allow(non_snake_case)]
pub struct DeltaChunkGrads<B: Backend> {
    /// `[batch, sequence, nheads, head_k_dim]`
    pub d_q_bshk: F<B, 4>,
    /// `[batch, sequence, nheads, head_k_dim]`
    pub d_k_bshk: F<B, 4>,
    /// `[batch, sequence, nheads, head_v_dim]`
    pub d_v_bshv: F<B, 4>,
    /// `[batch, sequence, nheads, erase_width]`
    pub d_erase_bshK: F<B, 4>,
    /// `[batch, sequence, nheads, write_width]`
    pub d_write_bshV: F<B, 4>,
    /// `[batch, sequence, nheads, decay_width]`; `None` when ungated.
    pub d_g_bshK: Option<F<B, 4>>,
    /// `[batch, nheads, head_k_dim, head_v_dim]`
    pub d_state_bhkv: F<B, 4>,
}

/// Recompute the chunkwise WY forward from its leaves, then differentiate it.
///
/// `d_y_bshv` / `d_final_state_bhkv` are the incoming cotangents of the two
/// outputs; the remaining arguments are exactly what the forward was given.
#[allow(non_snake_case, clippy::too_many_arguments)]
pub fn combined_backward<B: Backend>(
    d_y_bshv: F<B, 4>,
    d_final_state_bhkv: F<B, 4>,
    q_bshk: F<B, 4>,
    k_bshk: F<B, 4>,
    v_bshv: F<B, 4>,
    erase_bshK: F<B, 4>,
    write_bshV: F<B, 4>,
    g_bshK: Option<F<B, 4>>,
    state_bhkv: F<B, 4>,
    chunk_len: usize,
    scale: f64,
) -> DeltaChunkGrads<B> {
    // ── Recompute ───────────────────────────────────────────────────────────
    let input = Chunked::<B>::prepare(
        q_bshk, k_bshk, v_bshv, erase_bshK, write_bshV, g_bshK, chunk_len, scale,
    );
    let dims = input.dims;
    let wy = input.wy();
    let states_bnhkv = input
        .scan(&wy, state_bhkv, ScanMode::States)
        .states_bnhkv
        .expect("ScanMode::States keeps the stream");

    let pad = dims.sequence_padded - dims.sequence;
    let dy_bnhlv = chunked(pad_sequence(d_y_bshv, pad), dims.nchunks, dims.chunk_len);

    // ── Back through the serial scan ────────────────────────────────────────
    let scan = reverse_scan(&input, &wy, &states_bnhkv, dy_bnhlv, d_final_state_bhkv);
    drop(states_bnhkv);

    // ── Back through the WY transform ───────────────────────────────────────
    // `attn = tril(Aqk, 0)`; `U = T Vw`; `W = T (Ke ⊙ Eg)`.
    let d_aqk_bnhll = scan.d_attn_bnhll.tril(0);
    let d_t_bnhll = scan
        .d_u_bnhlv
        .clone()
        .matmul(wy.vw_bnhlv.clone().transpose())
        + scan
            .d_w_bnhlk
            .clone()
            .matmul(wy.kg_bnhlk.clone().transpose());
    let tT_bnhll = wy.t_bnhll.clone().transpose();
    let d_vw_bnhlv = tT_bnhll.clone().matmul(scan.d_u_bnhlv);
    let d_kg_bnhlk = tT_bnhll.clone().matmul(scan.d_w_bnhlk);

    // `Ḡ_N = tril(Tᵀ Ḡ_T Tᵀ, −1)`, and `N = tril(−Akk, −1)`.
    let d_akk_bnhll = -(tT_bnhll.clone().matmul(d_t_bnhll).matmul(tT_bnhll)).tril(-1);

    // `Ke ⊙ Eg` splits back onto the key term and the gate.
    let (mut d_ke_bnhlk, mut d_eg_bnhlK) = match &input.eg_bnhlK {
        Some(eg) => (
            d_kg_bnhlk.clone() * eg.clone(),
            Some(reduce_to_width(
                d_kg_bnhlk * wy.ke_bnhlk.clone(),
                dims.decay_width,
            )),
        ),
        None => (d_kg_bnhlk, None),
    };
    if let (Some(total), Some(from_scan)) = (d_eg_bnhlK.as_mut(), scan.d_eg_bnhlK) {
        *total = total.clone() + from_scan;
    }

    // ── Back through the two score matrices ─────────────────────────────────
    let mut d_q_bnhlk = scan.d_q_bnhlk;
    let mut d_k_bnhlk = scan.d_k_bnhlk;
    let mut d_gc_bnhlK = scan.d_gc_bnhlK;

    let scores = if dims.channel_decay() {
        channel_scores_backward(&input, &wy, d_akk_bnhll, d_aqk_bnhll)
    } else {
        head_scores_backward(&input, &wy, d_akk_bnhll, d_aqk_bnhll)
    };
    d_ke_bnhlk = d_ke_bnhlk + scores.d_ke_bnhlk;
    d_q_bnhlk = d_q_bnhlk + scores.d_q_bnhlk;
    d_k_bnhlk = d_k_bnhlk + scores.d_k_bnhlk;
    if let (Some(total), Some(from_scores)) = (d_gc_bnhlK.as_mut(), scores.d_gc_bnhlK) {
        *total = total.clone() + from_scores;
    }

    // ── The element-wise front-end ──────────────────────────────────────────
    // `Ke = K ⊙ erase`, `Vw = V ⊙ write`.
    d_k_bnhlk = d_k_bnhlk + d_ke_bnhlk.clone() * input.erase_bnhlK.clone();
    let d_erase_bnhlK = reduce_to_width(d_ke_bnhlk * input.k_bnhlk.clone(), dims.erase_width);
    let d_v_bnhlv = d_vw_bnhlv.clone() * input.write_bnhlV.clone();
    let d_write_bnhlV = reduce_to_width(d_vw_bnhlv * input.v_bnhlv.clone(), dims.write_width);

    // `G` is a cumulative sum: every use of `e^G` folds in here, and the
    // gradient of the sum is the reverse cumulative sum of the gradient.
    let d_g_bnhlK = d_gc_bnhlK.map(|d_gc| {
        let d_gc = match (d_eg_bnhlK, &input.eg_bnhlK) {
            (Some(d_eg), Some(eg)) => d_gc + d_eg * eg.clone(),
            _ => d_gc,
        };
        d_gc.flip(&[3]).cumsum(3).flip(&[3])
    });

    // ── Back to `[batch, sequence, nheads, ·]`, pad removed ─────────────────
    let unpad = |t: F<B, 5>| {
        let t = unchunked(t);
        if pad == 0 {
            t
        } else {
            t.narrow(1, 0, dims.sequence)
        }
    };

    DeltaChunkGrads {
        d_q_bshk: unpad(d_q_bnhlk.mul_scalar(scale)),
        d_k_bshk: unpad(d_k_bnhlk),
        d_v_bshv: unpad(d_v_bnhlv),
        d_erase_bshK: unpad(d_erase_bnhlK),
        d_write_bshV: unpad(d_write_bnhlV),
        d_g_bshK: d_g_bnhlK.map(unpad),
        d_state_bhkv: scan.d_state_bhkv,
    }
}

// ---------------------------------------------------------------------------
// The serial scan, in reverse
// ---------------------------------------------------------------------------

/// What the reverse scan hands to the rest of the backward.
#[allow(non_snake_case)]
struct ScanGrads<B: Backend> {
    d_u_bnhlv: F<B, 5>,
    d_w_bnhlk: F<B, 5>,
    d_attn_bnhll: F<B, 5>,
    d_q_bnhlk: F<B, 5>,
    d_k_bnhlk: F<B, 5>,
    /// From the chunk-boundary carry only; the `e^G` uses are in `d_eg_bnhlK`.
    d_gc_bnhlK: Option<F<B, 5>>,
    d_eg_bnhlK: Option<F<B, 5>>,
    d_state_bhkv: F<B, 4>,
}

/// Walk `S ← (α ⊙ S) − Kᵀ W S + Kᵀ U` backwards, carrying `dS`.
#[allow(non_snake_case)]
fn reverse_scan<B: Backend>(
    input: &Chunked<B>,
    wy: &Wy<B>,
    states_bnhkv: &F<B, 5>,
    dy_bnhlv: F<B, 5>,
    d_final_state_bhkv: F<B, 4>,
) -> ScanGrads<B> {
    let Dims {
        batch,
        nchunks,
        nheads,
        chunk_len,
        decay_width,
        ..
    } = input.dims;
    let device = dy_bnhlv.device();
    let dtype = dy_bnhlv.dtype();
    let gated = input.dims.gated();

    let mut d_us = Vec::with_capacity(nchunks);
    let mut d_ws = Vec::with_capacity(nchunks);
    let mut d_attns = Vec::with_capacity(nchunks);
    let mut d_qs = Vec::with_capacity(nchunks);
    let mut d_ks = Vec::with_capacity(nchunks);
    let mut d_gcs = Vec::with_capacity(nchunks);
    let mut d_egs = Vec::with_capacity(nchunks);

    let mut d_state_bhkv = d_final_state_bhkv;

    for i in (0..nchunks).rev() {
        let state_bhkv = pick(states_bnhkv, i);
        let u_bhlv = pick(&wy.u_bnhlv, i);
        let w_bhlk = pick(&wy.w_bnhlk, i);
        let attn_bhll = pick(&wy.attn_bnhll, i);
        let q_bhlk = pick(&input.q_bnhlk, i);
        let k_bhlk = pick(&input.k_bnhlk, i);
        let dy_bhlv = pick(&dy_bnhlv, i);
        let stateT_bhvk = state_bhkv.clone().transpose();

        // The forward's chunk-local quantities, rebuilt.
        let v_new_bhlv = u_bhlv - w_bhlk.clone().matmul(state_bhkv.clone());
        let gate = gated.then(|| {
            let gc_bhlK = pick(
                input.gc_bnhlK.as_ref().expect("gated implies a cumulative gate"),
                i,
            );
            let eg_bhlK = pick(input.eg_bnhlK.as_ref().expect("gated implies e^G"), i);
            let gl_bh1K = gc_bhlK.clone().narrow(2, chunk_len - 1, 1);
            // `e^{G_last − Gᵢ}`: what the chunk-boundary carry scales `k` by.
            let ec_bhlK = (gl_bh1K.clone() - gc_bhlK).exp();
            // The state's decay rides its *key* axis.
            let egl_bhK1 = gl_bh1K.swap_dims(2, 3).exp();
            (ec_bhlK, eg_bhlK, egl_bhK1)
        });

        let (kc_bhlk, qr_bhlk) = match &gate {
            Some((ec, eg, _)) => (k_bhlk.clone() * ec.clone(), q_bhlk.clone() * eg.clone()),
            None => (k_bhlk.clone(), q_bhlk.clone()),
        };

        // ── y = qr S + attn V' ; S' = S ⊙ e^{G_last} + kcᵀ V' ───────────────
        let d_attn_bhll = dy_bhlv.clone().matmul(v_new_bhlv.clone().transpose());
        let d_v_new_bhlv = attn_bhll.transpose().matmul(dy_bhlv.clone())
            + kc_bhlk.matmul(d_state_bhkv.clone());
        let d_kc_bhlk = v_new_bhlv.matmul(d_state_bhkv.clone().transpose());
        let d_qr_bhlk = dy_bhlv.clone().matmul(stateT_bhvk.clone());

        let d_u_bhlv = d_v_new_bhlv.clone();
        let d_w_bhlk = -(d_v_new_bhlv.clone().matmul(stateT_bhvk));

        let mut d_state_next_bhkv = qr_bhlk.transpose().matmul(dy_bhlv)
            - w_bhlk.transpose().matmul(d_v_new_bhlv);

        match &gate {
            Some((ec_bhlK, eg_bhlK, egl_bhK1)) => {
                d_state_next_bhkv =
                    d_state_next_bhkv + d_state_bhkv.clone() * egl_bhK1.clone();

                // `kc = k ⊙ e^{G_last − G}`
                d_ks.push(d_kc_bhlk.clone() * ec_bhlK.clone());
                let d_carry_bhlK = reduce_to_width(d_kc_bhlk * k_bhlk, decay_width)
                    * ec_bhlK.clone();
                // `G_last` is row `chunk_len − 1` of `G` itself.
                let mut d_gl_bh1K = d_carry_bhlK.clone().sum_dim(2);
                // `S ⊙ e^{G_last}`: sum the state's own axes back onto the gate's.
                let mut d_egl_bhK1 = (d_state_bhkv * state_bhkv).sum_dim(3);
                if decay_width == 1 {
                    d_egl_bhK1 = d_egl_bhK1.sum_dim(2);
                }
                d_gl_bh1K = d_gl_bh1K + (d_egl_bhK1 * egl_bhK1.clone()).swap_dims(2, 3);

                let d_gc_bhlK = -d_carry_bhlK
                    + scatter_last_row(d_gl_bh1K, chunk_len, &device, dtype);
                d_gcs.push(d_gc_bhlK);
                d_egs.push(reduce_to_width(d_qr_bhlk.clone() * q_bhlk, decay_width));
                d_qs.push(d_qr_bhlk * eg_bhlK.clone());
            }
            None => {
                d_state_next_bhkv = d_state_next_bhkv + d_state_bhkv;
                d_ks.push(d_kc_bhlk);
                d_qs.push(d_qr_bhlk);
            }
        }

        d_us.push(d_u_bhlv);
        d_ws.push(d_w_bhlk);
        d_attns.push(d_attn_bhll);
        d_state_bhkv = d_state_next_bhkv;
    }

    // The chunks were walked backwards.
    let stack = |mut v: Vec<F<B, 4>>| {
        v.reverse();
        F::stack::<5>(v, 1)
    };
    debug_assert_eq!(
        [batch, nheads, input.dims.head_k_dim, input.dims.head_v_dim],
        d_state_bhkv.dims()
    );

    ScanGrads {
        d_u_bnhlv: stack(d_us),
        d_w_bnhlk: stack(d_ws),
        d_attn_bnhll: stack(d_attns),
        d_q_bnhlk: stack(d_qs),
        d_k_bnhlk: stack(d_ks),
        d_gc_bnhlK: gated.then(|| stack(d_gcs)),
        d_eg_bnhlK: gated.then(|| stack(d_egs)),
        d_state_bhkv,
    }
}

/// Place a `[batch, nheads, 1, width]` gradient at row `chunk_len − 1` of an
/// otherwise-zero `[batch, nheads, chunk_len, width]` — the transpose of the
/// `narrow` that took `G`'s last row.
fn scatter_last_row<B: Backend>(
    row_bh1K: F<B, 4>,
    chunk_len: usize,
    device: &burn::backend::tensor::Device<B>,
    dtype: burn::backend::FloatDType,
) -> F<B, 4> {
    if chunk_len == 1 {
        return row_bh1K;
    }
    let [batch, nheads, _one, width] = row_bh1K.dims();
    let zeros = F::<B, 4>::zeros([batch, nheads, chunk_len - 1, width], device, dtype);
    F::cat(vec![zeros, row_bh1K], 2)
}

// ---------------------------------------------------------------------------
// The two score matrices, in reverse
// ---------------------------------------------------------------------------

/// What differentiating `(Akk, Aqk)` contributes.
#[allow(non_snake_case)]
struct ScoreGrads<B: Backend> {
    d_ke_bnhlk: F<B, 5>,
    d_q_bnhlk: F<B, 5>,
    d_k_bnhlk: F<B, 5>,
    d_gc_bnhlK: Option<F<B, 5>>,
}

/// The per-head branch: `A = (rows Kᵀ) ⊙ e^{Gᵢ−Gⱼ}`, where the decay is a plain
/// `[chunk_len, chunk_len]` mask that factors out of the contraction.
#[allow(non_snake_case)]
fn head_scores_backward<B: Backend>(
    input: &Chunked<B>,
    wy: &Wy<B>,
    d_akk_bnhll: F<B, 5>,
    d_aqk_bnhll: F<B, 5>,
) -> ScoreGrads<B> {
    let mask_bnhll = input.head_decay_mask();
    let kT_bnhkl = input.k_bnhlk.clone().transpose();

    let (d_raw_kk, d_raw_qk) = match &mask_bnhll {
        Some(mask) => (
            d_akk_bnhll.clone() * mask.clone(),
            d_aqk_bnhll.clone() * mask.clone(),
        ),
        None => (d_akk_bnhll.clone(), d_aqk_bnhll.clone()),
    };

    let d_ke_bnhlk = d_raw_kk.clone().matmul(input.k_bnhlk.clone());
    let d_q_bnhlk = d_raw_qk.clone().matmul(input.k_bnhlk.clone());
    let d_k_bnhlk = d_raw_kk.transpose().matmul(wy.ke_bnhlk.clone())
        + d_raw_qk.transpose().matmul(input.q_bnhlk.clone());

    // The mask's own gradient, and from there `G`'s.
    let d_gc_bnhlK = mask_bnhll.map(|mask| {
        let raw_kk = wy.ke_bnhlk.clone().matmul(kT_bnhkl.clone());
        let raw_qk = input.q_bnhlk.clone().matmul(kT_bnhkl);
        let d_mask = d_akk_bnhll * raw_kk + d_aqk_bnhll * raw_qk;
        // `mask = e^{diff}` masked to the causal triangle, so `d_diff` is
        // already zero everywhere the mask is.
        let d_diff = d_mask * mask;
        // `diff[i, j] = Gᵢ − Gⱼ`.
        d_diff.clone().sum_dim(4) - d_diff.sum_dim(3).swap_dims(3, 4)
    });

    ScoreGrads {
        d_ke_bnhlk,
        d_q_bnhlk,
        d_k_bnhlk,
        d_gc_bnhlK,
    }
}

/// The per-channel branch: the decay shares the contraction's channel index, so
/// it rides [`BlockDecayPrim`]'s two bounded factors instead of a mask.
#[allow(non_snake_case)]
fn channel_scores_backward<B: Backend>(
    input: &Chunked<B>,
    wy: &Wy<B>,
    d_akk_bnhll: F<B, 5>,
    d_aqk_bnhll: F<B, 5>,
) -> ScoreGrads<B> {
    let Dims {
        batch,
        nchunks,
        nheads,
        chunk_len,
        head_k_dim,
        ..
    } = input.dims;
    let group = input.dims.group();
    let m = block_len(chunk_len);
    let nblocks = chunk_len / m;
    let flat = [group, chunk_len, head_k_dim];
    let blocked = [group, nblocks, m, head_k_dim];

    let decay = BlockDecayPrim::new(input);
    let gc_glk = input
        .gc_bnhlK
        .clone()
        .expect("channel decay implies a gate")
        .reshape(flat);
    let k_glk = input.k_bnhlk.clone().reshape(flat);
    let ke_glk = wy.ke_bnhlk.clone().reshape(flat);
    let q_glk = input.q_bnhlk.clone().reshape(flat);
    let device = gc_glk.device();
    let dtype = gc_glk.dtype();

    let rows_ke_gpmk = decay.rows_scaled(wy.ke_bnhlk.clone());
    let rows_q_gpmk = decay.rows_scaled(input.q_bnhlk.clone());
    let d_akk_gpml = d_akk_bnhll.reshape([group, nblocks, m, chunk_len]);
    let d_aqk_gpml = d_aqk_bnhll.reshape([group, nblocks, m, chunk_len]);

    // ── `out = rows_scaled @ colᵀ` ──────────────────────────────────────────
    let d_rows_ke_glk = d_akk_gpml
        .clone()
        .matmul(decay.col_gplk.clone())
        .reshape(flat);
    let d_rows_q_glk = d_aqk_gpml
        .clone()
        .matmul(decay.col_gplk.clone())
        .reshape(flat);
    let d_col_gplk = d_akk_gpml.swap_dims(2, 3).matmul(rows_ke_gpmk)
        + d_aqk_gpml.swap_dims(2, 3).matmul(rows_q_gpmk);

    // ── The row factor `e^{Gᵢ − G_ref}` ─────────────────────────────────────
    let d_ke_glk = d_rows_ke_glk.clone() * decay.row_glk.clone();
    let d_q_glk = d_rows_q_glk.clone() * decay.row_glk.clone();
    let d_row_glk = d_rows_ke_glk * ke_glk + d_rows_q_glk * q_glk;

    // ── The column factor `kⱼ e^{G_ref − Gⱼ}` ───────────────────────────────
    let d_k_from_col_glk = (d_col_gplk.clone() * decay.col_exp_gplk.clone())
        .sum_dim(1)
        .reshape(flat);
    // `−∞` outside the causal band makes `col_exp` exactly `0` there, so this
    // stays finite without any masking of its own.
    let d_col_arg_gplk =
        d_col_gplk * k_glk.clone().unsqueeze_dim::<4>(1) * decay.col_exp_gplk.clone();

    let gc_gpmk = gc_glk.clone().reshape(blocked);
    let ref_gp1k = gc_gpmk.clone().narrow(2, m / 2, 1);
    let e_gplk = ref_gp1k.clone() - gc_glk.unsqueeze_dim::<4>(1);
    let d_e_gplk = d_col_arg_gplk.mask_fill(e_gplk.ge_elem(MAX_EXPONENT), 0.0);

    let d_ref_from_col_gp1k = d_e_gplk.clone().sum_dim(2);
    let d_gc_from_col_glk = -d_e_gplk.sum_dim(1).reshape(flat);

    // ── The row factor's own exponent ───────────────────────────────────────
    let row_arg_gpmk = gc_gpmk - ref_gp1k;
    let d_row_arg_gpmk = (d_row_glk * decay.row_glk.clone())
        .reshape(blocked)
        .mask_fill(row_arg_gpmk.ge_elem(MAX_EXPONENT), 0.0);
    let d_ref_gp1k = d_ref_from_col_gp1k - d_row_arg_gpmk.clone().sum_dim(2);

    // `G_ref` is row `m / 2` of each block.
    let d_gc_glk = (d_row_arg_gpmk + scatter_block_row(d_ref_gp1k, m, m / 2, &device, dtype))
        .reshape(flat)
        + d_gc_from_col_glk;

    let unflatten =
        |t: F<B, 3>| t.reshape([batch, nchunks, nheads, chunk_len, head_k_dim]);

    ScoreGrads {
        d_ke_bnhlk: unflatten(d_ke_glk),
        d_q_bnhlk: unflatten(d_q_glk),
        d_k_bnhlk: unflatten(d_k_from_col_glk),
        d_gc_bnhlK: Some(unflatten(d_gc_glk)),
    }
}

/// Place a `[group, nblocks, 1, width]` gradient at row `at` of an
/// otherwise-zero `[group, nblocks, m, width]`.
fn scatter_block_row<B: Backend>(
    row_gp1k: F<B, 4>,
    m: usize,
    at: usize,
    device: &burn::backend::tensor::Device<B>,
    dtype: burn::backend::FloatDType,
) -> F<B, 4> {
    if m == 1 {
        return row_gp1k;
    }
    let [group, nblocks, _one, width] = row_gp1k.dims();
    let zeros = |rows: usize| F::<B, 4>::zeros([group, nblocks, rows, width], device, dtype);
    let mut parts = Vec::with_capacity(3);
    if at > 0 {
        parts.push(zeros(at));
    }
    parts.push(row_gp1k);
    if at + 1 < m {
        parts.push(zeros(m - at - 1));
    }
    F::cat(parts, 2)
}
