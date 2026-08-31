//! The chunkwise WY forward on backend primitives.
//!
//! A line-for-line port of [`chunk`](crate::delta::chunk) against [`F`] instead
//! of the `Dispatch`-pinned `Tensor`, which is what lets it run inside a custom
//! autodiff node under a generic `B`. It is written in three stages so the
//! [backward](super::combined_backward) can replay exactly the same stages
//! rather than store their results:
//!
//! 1. [`Chunked::prepare`] — pad, reshape to `[batch, nchunks, nheads, chunk_len, ·]`,
//!    scale `q`, and take the chunk-cumulative gate.
//! 2. [`Chunked::wy`] — the two intra-chunk score matrices, `T = (I − N)⁻¹`,
//!    and the `U`/`W`/`attn` the scan consumes.
//! 3. [`Chunked::scan`] — the serial inter-chunk recurrence.
//!
//! The maths, the padding argument and the two gate widths are all documented
//! in [`chunk`](crate::delta::chunk); nothing here decides anything the
//! high-level path does not.

#![allow(non_snake_case)]

use burn::backend::Backend;
use burn_stack::utils::fprim::{F, Mask, san};

use super::prim::FPrimExt;
use crate::delta::decay::{MAX_EXPONENT, block_len};
use crate::delta::tri::prim::unit_lower_inverse;

/// Every dimension the chunk body is written in terms of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Dims {
    /// `batch`
    pub batch: usize,
    /// The caller's (unpadded) sequence length.
    pub sequence: usize,
    /// `nchunks · chunk_len`
    pub sequence_padded: usize,
    /// `nchunks`
    pub nchunks: usize,
    /// `chunk_len`
    pub chunk_len: usize,
    /// `nheads`
    pub nheads: usize,
    /// `head_k_dim`
    pub head_k_dim: usize,
    /// `head_v_dim`
    pub head_v_dim: usize,
    /// The erase gate's last axis: `1` or `head_k_dim`.
    pub erase_width: usize,
    /// The write gate's last axis: `1` or `head_v_dim`.
    pub write_width: usize,
    /// The forget gate's last axis: `1` or `head_k_dim`; `0` when ungated.
    pub decay_width: usize,
}

impl Dims {
    /// Whether a forget gate is present at all.
    pub fn gated(&self) -> bool {
        self.decay_width > 0
    }

    /// Whether the forget gate varies per key channel — the branch that cannot
    /// factor the decay out of the key contraction (see
    /// [`decay`](crate::delta::decay)).
    pub fn channel_decay(&self) -> bool {
        self.decay_width > 1
    }

    /// `batch · nchunks · nheads`, the axis every chunk-local matmul batches over.
    pub fn group(&self) -> usize {
        self.batch * self.nchunks * self.nheads
    }
}

// ---------------------------------------------------------------------------
// Layout helpers
// ---------------------------------------------------------------------------

/// Zero-pad the sequence axis of a `[batch, sequence, nheads, ·]` tensor.
pub(crate) fn pad_sequence<B: Backend>(t: F<B, 4>, pad: usize) -> F<B, 4> {
    if pad == 0 {
        return t;
    }
    let [batch, _s, nheads, last] = t.dims();
    let zeros = F::<B, 4>::zeros([batch, pad, nheads, last], &t.device(), t.dtype());
    F::cat(vec![t, zeros], 1)
}

/// `[batch, sequence_padded, nheads, ·]` → `[batch, nchunks, nheads, chunk_len, ·]`.
///
/// Heads ahead of the chunk axis so every matmul batches over
/// `(batch, nchunks, nheads)` and acts on the `[chunk_len, ·]` planes.
pub(crate) fn chunked<B: Backend>(t: F<B, 4>, nchunks: usize, chunk_len: usize) -> F<B, 5> {
    let [batch, _s, nheads, last] = t.dims();
    t.reshape([batch, nchunks, chunk_len, nheads, last])
        .swap_dims(2, 3)
}

/// The inverse of [`chunked`].
pub(crate) fn unchunked<B: Backend>(t: F<B, 5>) -> F<B, 4> {
    let [batch, nchunks, nheads, chunk_len, last] = t.dims();
    t.swap_dims(2, 3)
        .reshape([batch, nchunks * chunk_len, nheads, last])
}

/// Sum a full-width `[.., full]` tensor back onto a gate's own axis: a no-op
/// when the gate is per channel, a keepdim sum when it is one value per head.
pub(crate) fn reduce_to_width<B: Backend, const D: usize>(t: F<B, D>, width: usize) -> F<B, D> {
    if width == 1 && t.dims()[D - 1] != 1 {
        t.sum_dim(D - 1)
    } else {
        t
    }
}

/// Narrow a chunked tensor to one chunk: `[b, n, h, l, ·]` → `[b, h, l, ·]`.
pub(crate) fn pick<B: Backend>(t: &F<B, 5>, i: usize) -> F<B, 4> {
    let [batch, _n, nheads, chunk_len, last] = t.dims();
    t.clone()
        .narrow(1, i, 1)
        .reshape([batch, nheads, chunk_len, last])
}

// ---------------------------------------------------------------------------
// Stage 1: the chunked inputs
// ---------------------------------------------------------------------------

/// The block's inputs, padded and reshaped into the chunk layout.
///
/// `q` already carries the readout scale, and `gc` is the chunk-cumulative
/// gate — both of which the backward needs in exactly this form.
#[allow(non_snake_case)]
pub(crate) struct Chunked<B: Backend> {
    /// `q · scale`. `[batch, nchunks, nheads, chunk_len, head_k_dim]`
    pub q_bnhlk: F<B, 5>,
    /// `[batch, nchunks, nheads, chunk_len, head_k_dim]`
    pub k_bnhlk: F<B, 5>,
    /// `[batch, nchunks, nheads, chunk_len, head_v_dim]`
    pub v_bnhlv: F<B, 5>,
    /// `[batch, nchunks, nheads, chunk_len, erase_width]`
    pub erase_bnhlK: F<B, 5>,
    /// `[batch, nchunks, nheads, chunk_len, write_width]`
    pub write_bnhlV: F<B, 5>,
    /// `Gᵢ = Σ_{j ≤ i} gⱼ` within the chunk.
    /// `[batch, nchunks, nheads, chunk_len, decay_width]`
    pub gc_bnhlK: Option<F<B, 5>>,
    /// `e^G`, built once because the scan, `W` and the readout all want it.
    pub eg_bnhlK: Option<F<B, 5>>,
    /// Shapes.
    pub dims: Dims,
}

impl<B: Backend> Chunked<B> {
    /// Pad, reshape and pre-multiply. See the [module header](self) for the stages.
    #[allow(non_snake_case, clippy::too_many_arguments)]
    pub(crate) fn prepare(
        q_bshk: F<B, 4>,
        k_bshk: F<B, 4>,
        v_bshv: F<B, 4>,
        erase_bshK: F<B, 4>,
        write_bshV: F<B, 4>,
        g_bshK: Option<F<B, 4>>,
        chunk_len: usize,
        scale: f64,
    ) -> Self {
        let [batch, sequence, nheads, head_k_dim] = q_bshk.dims();
        let head_v_dim = v_bshv.dims()[3];
        let sequence_padded = sequence.next_multiple_of(chunk_len);
        let pad = sequence_padded - sequence;
        let nchunks = sequence_padded / chunk_len;

        let dims = Dims {
            batch,
            sequence,
            sequence_padded,
            nchunks,
            chunk_len,
            nheads,
            head_k_dim,
            head_v_dim,
            erase_width: erase_bshK.dims()[3],
            write_width: write_bshV.dims()[3],
            decay_width: g_bshK.as_ref().map_or(0, |g| g.dims()[3]),
        };

        let to_chunks = |t: F<B, 4>| chunked(pad_sequence(t, pad), nchunks, chunk_len);

        let q_bnhlk = to_chunks(q_bshk).mul_scalar(scale);
        let k_bnhlk = to_chunks(k_bshk);
        let v_bnhlv = to_chunks(v_bshv);
        let erase_bnhlK = to_chunks(erase_bshK);
        let write_bnhlV = to_chunks(write_bshV);
        // `gc[i] − gc[j]` is the decay carried from step `j` to step `i`;
        // differences of a cumulative sum are what keeps that stable.
        let gc_bnhlK = g_bshK.map(|g| to_chunks(g).cumsum(3));
        let eg_bnhlK = gc_bnhlK.as_ref().map(|gc| gc.clone().exp());

        Self {
            q_bnhlk,
            k_bnhlk,
            v_bnhlv,
            erase_bnhlK,
            write_bnhlV,
            gc_bnhlK,
            eg_bnhlK,
            dims,
        }
    }
}

// ---------------------------------------------------------------------------
// Stage 2: the WY transform
// ---------------------------------------------------------------------------

/// Everything the scan consumes, plus the two factors the backward re-reads.
#[allow(non_snake_case)]
pub(crate) struct Wy<B: Backend> {
    /// `erase ⊙ K`. `[batch, nchunks, nheads, chunk_len, head_k_dim]`
    pub ke_bnhlk: F<B, 5>,
    /// `write ⊙ V`. `[batch, nchunks, nheads, chunk_len, head_v_dim]`
    pub vw_bnhlv: F<B, 5>,
    /// `(erase ⊙ K) ⊙ e^G` — `W`'s right-hand side, kept for `dT`.
    pub kg_bnhlk: F<B, 5>,
    /// `T = (I − N)⁻¹`. `[batch, nchunks, nheads, chunk_len, chunk_len]`
    pub t_bnhll: F<B, 5>,
    /// `U = T (write ⊙ V)`
    pub u_bnhlv: F<B, 5>,
    /// `W = T ((erase ⊙ K) ⊙ e^G)`
    pub w_bnhlk: F<B, 5>,
    /// `tril(Aqk, 0)` — the intra-chunk attention.
    pub attn_bnhll: F<B, 5>,
}

impl<B: Backend> Chunked<B> {
    /// The two score matrices, `T`, and `U`/`W`/`attn`.
    #[allow(non_snake_case)]
    pub(crate) fn wy(&self) -> Wy<B> {
        let ke_bnhlk = self.k_bnhlk.clone() * self.erase_bnhlK.clone();
        let vw_bnhlv = self.v_bnhlv.clone() * self.write_bnhlV.clone();

        let (akk_bnhll, aqk_bnhll) = self.scores(ke_bnhlk.clone());

        //   N = −tril(Akk, −1),  T = (I − N)⁻¹
        let n_bnhll = (-akk_bnhll).tril(-1);
        let t_bnhll = self.solve(n_bnhll);

        let kg_bnhlk = match &self.eg_bnhlK {
            Some(eg) => ke_bnhlk.clone() * eg.clone(),
            None => ke_bnhlk.clone(),
        };

        let u_bnhlv = t_bnhll.clone().matmul(vw_bnhlv.clone());
        let w_bnhlk = t_bnhll.clone().matmul(kg_bnhlk.clone());

        // The diagonal stays: the readout `yₜ = Sₜᵀ qₜ` sees the state *after*
        // the current token's own write.
        let attn_bnhll = aqk_bnhll.tril(0);

        Wy {
            ke_bnhlk,
            vw_bnhlv,
            kg_bnhlk,
            t_bnhll,
            u_bnhlv,
            w_bnhlk,
            attn_bnhll,
        }
    }

    /// `(Akk, Aqk)` — the key-key and query-key intra-chunk scores, both
    /// unmasked (the caller applies its own `tril`).
    #[allow(non_snake_case)]
    pub(crate) fn scores(&self, ke_bnhlk: F<B, 5>) -> (F<B, 5>, F<B, 5>) {
        if self.dims.channel_decay() {
            let decay = BlockDecayPrim::new(self);
            (
                decay.scores(ke_bnhlk),
                decay.scores(self.q_bnhlk.clone()),
            )
        } else {
            let mask = self.head_decay_mask();
            let kT_bnhkl = self.k_bnhlk.clone().transpose();
            let gated = |scores: F<B, 5>| match &mask {
                Some(mask) => scores * mask.clone(),
                None => scores,
            };
            (
                gated(ke_bnhlk.matmul(kT_bnhkl.clone())),
                gated(self.q_bnhlk.clone().matmul(kT_bnhkl)),
            )
        }
    }

    /// `e^{Gᵢ − Gⱼ}` as a plain causal `[chunk_len, chunk_len]` mask — the form
    /// a *per-head* decay takes, where it factors out of the key contraction.
    ///
    /// `None` when there is no gate at all.
    pub(crate) fn head_decay_mask(&self) -> Option<F<B, 5>> {
        let gc_bnhlK = self.gc_bnhlK.as_ref()?;
        let Dims {
            batch,
            nchunks,
            nheads,
            chunk_len,
            ..
        } = self.dims;
        debug_assert_eq!(1, self.dims.decay_width, "per-head branch, per-head gate");
        let gc_bnhl = gc_bnhlK.clone().squeeze_dim::<4>(4);
        let diff = gc_bnhl.clone().unsqueeze_dim::<5>(4) - gc_bnhl.unsqueeze_dim::<5>(3);
        // `−∞` strictly above the diagonal, so the mask is exactly `0` there
        // (the additive `−∞` of the high-level path, without materialising it).
        // `tril_mask(0)` is `true` exactly where a `tril(0)` fills: strictly
        // above the diagonal, which the readout's own token must stay out of.
        let causal = Mask::tril_mask(chunk_len, chunk_len, 0, &diff.device())
            .reshape([1, 1, 1, chunk_len, chunk_len])
            .expand([batch, nchunks, nheads, chunk_len, chunk_len]);
        let mask = diff.mask_fill(causal, f32::NEG_INFINITY).exp();
        san(&mask);
        Some(mask)
    }

    /// `T = (I − N)⁻¹`, rank-erased to the `[flat, l, l]` the ladder works on.
    pub(crate) fn solve(&self, n_bnhll: F<B, 5>) -> F<B, 5> {
        let Dims {
            batch,
            nchunks,
            nheads,
            chunk_len,
            ..
        } = self.dims;
        let group = self.dims.group();
        unit_lower_inverse(n_bnhll.reshape([group, chunk_len, chunk_len]))
            .reshape([batch, nchunks, nheads, chunk_len, chunk_len])
    }
}

// ---------------------------------------------------------------------------
// Stage 3: the serial scan
// ---------------------------------------------------------------------------

/// What the scan is being run for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScanMode {
    /// The forward: the per-token outputs and the final state.
    Outputs,
    /// The backward's recompute: only the state entering each chunk. The
    /// outputs are not built — the backward derives its own `V'` and never
    /// reads `y`.
    States,
}

/// What the inter-chunk scan produces.
pub(crate) struct Scan<B: Backend> {
    /// `[batch, nchunks, nheads, chunk_len, head_v_dim]`; `None` under
    /// [`ScanMode::States`].
    pub y_bnhlv: Option<F<B, 5>>,
    /// `[batch, nheads, head_k_dim, head_v_dim]`
    pub final_state_bhkv: F<B, 4>,
    /// The state *entering* each chunk, `[batch, nchunks, nheads, head_k_dim,
    /// head_v_dim]` — the one stream the backward has to walk in reverse, and
    /// the reason it recomputes the scan before it can start. `None` under
    /// [`ScanMode::Outputs`].
    pub states_bnhkv: Option<F<B, 5>>,
}

impl<B: Backend> Chunked<B> {
    /// `S ← (α ⊙ S) − Kᵀ W S + Kᵀ U`, chunk by chunk.
    ///
    /// The inter-chunk recurrence is matrix-valued with a rank-`chunk_len`
    /// update, so there is no scalar-decay shortcut to parallelise it.
    #[allow(non_snake_case)]
    pub(crate) fn scan(&self, wy: &Wy<B>, state_bhkv: F<B, 4>, mode: ScanMode) -> Scan<B> {
        let Dims {
            batch,
            nchunks,
            nheads,
            chunk_len,
            head_k_dim,
            head_v_dim,
            ..
        } = self.dims;

        let mut state_bhkv = state_bhkv;
        let mut ys = Vec::with_capacity(nchunks);
        let mut states = (mode == ScanMode::States).then(|| Vec::with_capacity(nchunks));

        for i in 0..nchunks {
            if let Some(states) = states.as_mut() {
                states.push(state_bhkv.clone());
            }
            let k_bhlk = pick(&self.k_bnhlk, i);
            let w_bhlk = pick(&wy.w_bnhlk, i);
            let u_bhlv = pick(&wy.u_bnhlv, i);

            // V' = U − W S₀: the chunk's own updates, corrected for what the
            // incoming state already holds.
            let v_new_bhlv = u_bhlv - w_bhlk.matmul(state_bhkv.clone());

            if mode == ScanMode::Outputs {
                let q_bhlk = pick(&self.q_bnhlk, i);
                let attn_bhll = pick(&wy.attn_bnhll, i);
                let q_read_bhlk = match &self.eg_bnhlK {
                    Some(eg) => q_bhlk * pick(eg, i),
                    None => q_bhlk,
                };
                ys.push(
                    q_read_bhlk.matmul(state_bhkv.clone())
                        + attn_bhll.matmul(v_new_bhlv.clone()),
                );
            }

            // Carry to the chunk boundary: decay everything to the last step,
            // then add each token's write decayed from its own step.
            let (decayed_bhkv, k_carry_bhlk) = match &self.gc_bnhlK {
                Some(gc_bnhlK) => {
                    let gc_bhlK = pick(gc_bnhlK, i);
                    let gc_last_bh1K = gc_bhlK.clone().narrow(2, chunk_len - 1, 1);
                    // The decay is on the state's key axis.
                    let decayed =
                        state_bhkv * gc_last_bh1K.clone().swap_dims(2, 3).exp();
                    let carry = k_bhlk * (gc_last_bh1K - gc_bhlK).exp();
                    (decayed, carry)
                }
                None => (state_bhkv, k_bhlk),
            };
            state_bhkv = decayed_bhkv + k_carry_bhlk.transpose().matmul(v_new_bhlv);
        }

        let y_bnhlv = (mode == ScanMode::Outputs).then(|| {
            let y = F::stack::<5>(ys, 1);
            debug_assert_eq!([batch, nchunks, nheads, chunk_len, head_v_dim], y.dims());
            y
        });
        debug_assert_eq!([batch, nheads, head_k_dim, head_v_dim], state_bhkv.dims());

        Scan {
            y_bnhlv,
            final_state_bhkv: state_bhkv,
            states_bnhkv: states.map(|s| F::stack::<5>(s, 1)),
        }
    }
}

// ---------------------------------------------------------------------------
// The per-channel decay factors
// ---------------------------------------------------------------------------

/// The primitive port of [`BlockDecay`](crate::delta::decay::BlockDecay): the
/// two bounded factors `e^{Gᵢ − Gⱼ}` is split into when the gate is per key
/// channel. The derivation, and why the split is necessary at all, is in that
/// module's header.
#[allow(non_snake_case)]
pub(crate) struct BlockDecayPrim<B: Backend> {
    /// `e^{Gᵢ − G_ref(p(i))}`. `[group, chunk_len, head_k_dim]`
    pub row_glk: F<B, 3>,
    /// `e^{G_ref(p) − Gⱼ}`, exactly `0` outside the causal block band.
    /// `[group, nblocks, chunk_len, head_k_dim]`
    pub col_exp_gplk: F<B, 4>,
    /// `kⱼ · col_exp`, the factor [`Self::scores`] contracts against.
    pub col_gplk: F<B, 4>,
    /// Rows per reference block.
    pub block_len: usize,
    /// Shapes.
    pub dims: Dims,
}

impl<B: Backend> BlockDecayPrim<B> {
    /// Build the factors from the keys and the chunk-cumulative log decay.
    #[allow(non_snake_case)]
    pub(crate) fn new(chunked: &Chunked<B>) -> Self {
        let dims = chunked.dims;
        let Dims {
            chunk_len,
            head_k_dim,
            ..
        } = dims;
        let group = dims.group();
        let m = block_len(chunk_len);
        let nblocks = chunk_len / m;
        let gc_bnhlK = chunked
            .gc_bnhlK
            .as_ref()
            .expect("channel decay implies a gate");
        let device = gc_bnhlK.device();
        let dtype = gc_bnhlK.dtype();

        let gc_glk = gc_bnhlK.clone().reshape([group, chunk_len, head_k_dim]);
        let k_glk = chunked
            .k_bnhlk
            .clone()
            .reshape([group, chunk_len, head_k_dim]);

        // The reference point of each block, at its middle row.
        let gc_gpmk = gc_glk.clone().reshape([group, nblocks, m, head_k_dim]);
        let ref_gp1k = gc_gpmk.clone().narrow(2, m / 2, 1);

        let row_glk = (gc_gpmk - ref_gp1k.clone())
            .clamp_max(MAX_EXPONENT)
            .exp()
            .reshape([group, chunk_len, head_k_dim]);

        // `−∞` wherever column `j` lies in a block *after* row block `p`: those
        // exponents are unbounded above, and `exp(−∞) = 0` keeps them out
        // without ever forming an `inf` to multiply by zero.
        let band_1pl1 = F::<B, 2>::full([nblocks, nblocks], f32::NEG_INFINITY, &device, dtype)
            .triu(1)
            .unsqueeze_dim::<3>(2)
            .expand([nblocks, nblocks, m])
            .reshape([1, nblocks, chunk_len, 1]);

        let e_gplk = ref_gp1k - gc_glk.unsqueeze_dim::<4>(1);
        let col_exp_gplk = (e_gplk.clamp_max(MAX_EXPONENT) + band_1pl1).exp();
        let col_gplk = k_glk.unsqueeze_dim::<4>(1) * col_exp_gplk.clone();
        san(&row_glk);
        san(&col_exp_gplk);

        Self {
            row_glk,
            col_exp_gplk,
            col_gplk,
            block_len: m,
            dims,
        }
    }

    /// `M[i, j] = Σ_d rowsᵢ,d kⱼ,d e^{Gᵢ,d − Gⱼ,d}`, valid for `i ≥ j` and left
    /// unmasked.
    pub(crate) fn scores(&self, rows_bnhlk: F<B, 5>) -> F<B, 5> {
        self.scores_from(self.rows_scaled(rows_bnhlk))
    }

    /// `rows ⊙ row_glk`, reshaped to the `[group, nblocks, m, head_k_dim]` the
    /// contraction wants — the one intermediate the backward needs by name.
    pub(crate) fn rows_scaled(&self, rows_bnhlk: F<B, 5>) -> F<B, 4> {
        let Dims {
            chunk_len,
            head_k_dim,
            ..
        } = self.dims;
        let group = self.dims.group();
        let m = self.block_len;
        (rows_bnhlk.reshape([group, chunk_len, head_k_dim]) * self.row_glk.clone())
            .reshape([group, chunk_len / m, m, head_k_dim])
    }

    /// [`Self::scores`] from an already-scaled row factor.
    pub(crate) fn scores_from(&self, rows_gpmk: F<B, 4>) -> F<B, 5> {
        let Dims {
            batch,
            nchunks,
            nheads,
            chunk_len,
            ..
        } = self.dims;
        rows_gpmk
            .matmul(self.col_gplk.clone().swap_dims(2, 3))
            .reshape([batch, nchunks, nheads, chunk_len, chunk_len])
    }
}
