//! The intra-chunk score matrices when the forget gate is **per key channel**.
//!
//! With a per-head gate the chunk's decay factors out of the key contraction,
//! so `exp(Gᵢ − Gⱼ)` is a plain `[chunk_len, chunk_len]` mask and
//! [`chunk`](super::chunk) builds both score matrices with one matmul each. A
//! per-channel gate ([GDN-2](crate::gdn2)) does not factor:
//!
//! ```text
//!   M[i, j] = Σ_d rowsᵢ,d · kⱼ,d · e^{Gᵢ,d − Gⱼ,d}
//! ```
//!
//! The contraction and the decay share the channel index `d`. Splitting it as
//! `e^{Gᵢ,d} · e^{−Gⱼ,d}` restores the matmul — but `G ≤ 0` accumulates over
//! the chunk, so `e^{−Gⱼ}` overflows f32 within a few dozen tokens.
//!
//! ## Reference points
//!
//! The fix is the reference kernel's own (`fla/ops/gdn2/chunk_intra.py`): split
//! the chunk into blocks of [`DECAY_BLOCK_LEN`] rows and give each block `p` its
//! own reference `G_ref(p)`, taken at the block's **middle** row, so that
//!
//! ```text
//!   M[i, j] = Σ_d (rowsᵢ,d e^{Gᵢ,d − G_ref(p),d}) · (kⱼ,d e^{G_ref(p),d − Gⱼ,d})
//! ```
//!
//! Neither factor spans more than half a block of decay for `i` in block `p`
//! and `j` at or before it, which is what keeps both in range. Columns beyond
//! the block's own diagonal are the unbounded ones; their exponent is set to
//! `−∞` (so the factor is exactly `0`, never `inf`), and they are masked away
//! downstream regardless.
//!
//! Cost is unchanged in FLOPs — the `p` row blocks partition the rows, so the
//! `p` smaller matmuls do exactly the work of the one big one — at the price of
//! materialising the column factor `p` times.

use burn::prelude::*;

/// Rows sharing one reference point. The reference kernel's `BC`.
pub const DECAY_BLOCK_LEN: usize = 16;

/// The largest block length that divides `chunk_len` and does not exceed
/// [`DECAY_BLOCK_LEN`]. Every chunk length in practice is a power of two, so
/// this is [`DECAY_BLOCK_LEN`]; the fallback keeps odd lengths exact (at worst
/// one row per block, where the row factor is `1` and nothing can overflow).
fn block_len(chunk_len: usize) -> usize {
    (1..=DECAY_BLOCK_LEN.min(chunk_len))
        .rev()
        .find(|m| chunk_len % m == 0)
        .expect("1 divides everything")
}

/// The two bounded factors `exp(Gᵢ − Gⱼ)` is split into, built once per chunk
/// batch and reused for both score matrices (they share their **columns**: the
/// keys).
pub(super) struct BlockDecay {
    /// `e^{Gᵢ − G_ref(p(i))}`. `[batch·nchunks·nheads, chunk_len, head_k_dim]`
    row_glk: Tensor<3>,
    /// `kⱼ e^{G_ref(p) − Gⱼ}`, zero outside the causal block band.
    /// `[batch·nchunks·nheads, nblocks, chunk_len, head_k_dim]`
    col_gplk: Tensor<4>,
    /// `[batch, nchunks, nheads, chunk_len, head_k_dim]`
    dims: [usize; 5],
    /// Rows per reference block.
    block_len: usize,
}

impl BlockDecay {
    /// Prepare the factors from the keys and the chunk-cumulative log decay.
    ///
    /// # Shapes
    /// - `k_bnhlk`, `gc_bnhlk`: `[batch, nchunks, nheads, chunk_len, head_k_dim]`
    pub(super) fn new(k_bnhlk: Tensor<5>, gc_bnhlk: Tensor<5>) -> Self {
        let dims @ [batch, nchunks, nheads, chunk_len, head_k_dim] = k_bnhlk.dims();
        assert_eq!(dims, gc_bnhlk.dims());
        let device = k_bnhlk.device();
        let group = batch * nchunks * nheads;
        let m = block_len(chunk_len);
        let nblocks = chunk_len / m;

        let gc_glk = gc_bnhlk.reshape([group, chunk_len, head_k_dim]);
        let k_glk = k_bnhlk.reshape([group, chunk_len, head_k_dim]);

        // The reference point of each block, at its middle row.
        let gc_gpmk = gc_glk.clone().reshape([group, nblocks, m, head_k_dim]);
        let ref_gp1k = gc_gpmk.clone().narrow(2, m / 2, 1);

        let row_glk = (gc_gpmk - ref_gp1k.clone())
            .exp()
            .reshape([group, chunk_len, head_k_dim]);

        // `−∞` wherever column `j` lies in a block *after* row block `p`: those
        // exponents are unbounded above, and `exp(−∞) = 0` keeps them out
        // without ever forming an `inf` to multiply by zero.
        let band_1pl1 = {
            let zeros_pp = Tensor::<2>::zeros(Shape::new([nblocks, nblocks]), &device);
            let tri_pp = Tensor::full_like(&zeros_pp, f32::NEG_INFINITY).triu(1);
            let band_ppm = tri_pp.unsqueeze_dim::<3>(2)
                + Tensor::<3>::zeros(Shape::new([nblocks, nblocks, m]), &device);
            band_ppm.reshape([1, nblocks, chunk_len, 1])
        };
        let e_gplk = ref_gp1k - gc_glk.unsqueeze_dim::<4>(1);
        let col_gplk = k_glk.unsqueeze_dim::<4>(1) * (e_gplk + band_1pl1).exp();

        Self {
            row_glk,
            col_gplk,
            dims,
            block_len: m,
        }
    }

    /// `M[i, j] = Σ_d rowsᵢ,d kⱼ,d e^{Gᵢ,d − Gⱼ,d}`, valid for `i ≥ j` and
    /// left unmasked (the caller applies its own `tril`).
    ///
    /// # Shapes
    /// - `rows_bnhlk`: `[batch, nchunks, nheads, chunk_len, head_k_dim]`
    /// - returns `[batch, nchunks, nheads, chunk_len, chunk_len]`
    pub(super) fn scores(&self, rows_bnhlk: Tensor<5>) -> Tensor<5> {
        let [batch, nchunks, nheads, chunk_len, head_k_dim] = self.dims;
        assert_eq!(self.dims, rows_bnhlk.dims());
        let group = batch * nchunks * nheads;
        let m = self.block_len;
        let nblocks = chunk_len / m;

        let rows_gpmk = (rows_bnhlk.reshape([group, chunk_len, head_k_dim])
            * self.row_glk.clone())
        .reshape([group, nblocks, m, head_k_dim]);
        rows_gpmk
            .matmul(self.col_gplk.clone().swap_dims(2, 3))
            .reshape([batch, nchunks, nheads, chunk_len, chunk_len])
    }
}
