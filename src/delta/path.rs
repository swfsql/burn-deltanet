//! The delta-rule core's input bundle and algorithm selector.
//!
//! A block's job is to produce `(q, k, v, β, g)` from a token stream; this
//! module is where it hands them over. [`DeltaInput::run`] dispatches to the
//! [`recurrent`](super::recurrent) or [`chunk`](super::chunk) evaluation of the
//! very same recurrence — the two agree on outputs, final state and gradients,
//! which is what the families' forward/step parity rests on.

use burn::prelude::*;
use burn_stack::modules::sanity as san;

use super::tri::TriSolve;

/// Which evaluation of the delta rule runs.
///
/// The chunk length trades the intra-chunk GEMM work (super-linear: the WY
/// inverse is `log₂ L` matmuls of `[L, L]`) against the number of serial
/// inter-chunk steps. Every reference kernel settles on 64, which is also
/// [`DeltaPath::DEFAULT_CHUNK_LEN`]; `None` means exactly that.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeltaPath {
    /// Token-by-token recurrence: `O(sequence)` serial steps, no chunk-level
    /// algebra. The definition — used for decoding, short prefills, and as the
    /// correctness reference the chunked path is tested against.
    Recurrent,
    /// Chunkwise WY, **backward via autodiff**: every intermediate the forward
    /// builds — including the `⌈log₂ L⌉` levels of the WY inverse — stays on
    /// the tape. The plain, minimal form; for training, prefer
    /// [`Self::ChunkRecalculated`].
    Chunk {
        /// Tokens per chunk; `None` ⇒ [`DeltaPath::DEFAULT_CHUNK_LEN`].
        chunk_len: Option<usize>,
        /// How the WY transform's `(I − N)⁻¹` is evaluated.
        solve: TriSolve,
    },
    /// The same chunkwise WY forward with a **custom, memory-efficient
    /// backward**, and the default.
    ///
    /// Values are identical to [`Self::Chunk`] at [`TriSolve::Blocked`]; what
    /// differs is what reaches the tape. The whole body runs inside one custom
    /// autodiff node that retains only its seven leaf inputs, and its backward
    /// replays the forward before differentiating it in closed form — so the
    /// score matrices, the `⌈log₂ L⌉`-level ladder that inverts `I − N`, `T`,
    /// `U`, `W`, `attn` and the per-chunk state stream never have to stay
    /// alive. See [`chunk_recalculated`](super::chunk_recalculated).
    ///
    /// The three arms are the same recurrence at three points on the
    /// memory/complexity curve: [`Self::Recurrent`] is the definition,
    /// [`Self::Chunk`] the production forward differentiated plainly, and this
    /// the same forward with the backward written out by hand.
    ChunkRecalculated {
        /// Tokens per chunk; `None` ⇒ [`DeltaPath::DEFAULT_CHUNK_LEN`].
        chunk_len: Option<usize>,
    },
}

impl Default for DeltaPath {
    fn default() -> Self {
        Self::ChunkRecalculated { chunk_len: None }
    }
}

impl DeltaPath {
    /// The chunk length used when a [`DeltaPath::Chunk`] carries `None`.
    pub const DEFAULT_CHUNK_LEN: usize = 64;

    /// Chunkwise at the default chunk length and backward — i.e.
    /// [`Self::default`].
    pub fn chunk() -> Self {
        Self::default()
    }

    /// Chunkwise with an explicit chunk length, at the default backward.
    pub fn chunk_len(chunk_len: usize) -> Self {
        Self::ChunkRecalculated {
            chunk_len: Some(chunk_len),
        }
    }

    /// Chunkwise with an explicit chunk length, differentiated by autodiff
    /// ([`Self::Chunk`] at the default [`TriSolve`]).
    pub fn chunk_len_on_tape(chunk_len: usize) -> Self {
        Self::Chunk {
            chunk_len: Some(chunk_len),
            solve: TriSolve::default(),
        }
    }

    /// The chunk length this variant runs at ([`Self::Recurrent`] answers 1 —
    /// it *is* the chunk length of a token-at-a-time pass).
    pub fn resolved_chunk_len(&self) -> usize {
        match self {
            Self::Recurrent => 1,
            Self::Chunk { chunk_len, .. } | Self::ChunkRecalculated { chunk_len } => {
                chunk_len.unwrap_or(Self::DEFAULT_CHUNK_LEN)
            }
        }
    }
}

/// Everything the delta-rule core consumes, per head.
///
/// `q`/`k` arrive **already activated and normalised** (see
/// [`crate::common::norm`]) but **not** scaled: [`Self::scale`] is applied to
/// `q` inside, so the same bundle reads identically on either path. The gates
/// arrive already squashed to their ranges — the core applies no `sigmoid`,
/// `softplus` or `exp` of its own beyond `exp(g)`.
///
/// ## The gate axes `K` and `V`
///
/// The three gates are carried at their **broadcast** width: their last axis is
/// either `1` (one value per head — a scalar `β`, a scalar `α`) or the full
/// `head_k_dim` / `head_v_dim` (one value per channel — [GDN-2](crate::gated_deltanet_2)).
/// The two are the same function: a per-head `β` *is* `erase = β·1_k`,
/// `write = β·1_v`, and every expression below broadcasts over the axis without
/// branching. Only the [chunked](super::chunk) path looks at the width, because
/// a per-head decay factors out of the key contraction and a per-channel one
/// does not.
#[allow(non_snake_case)]
pub struct DeltaInput {
    /// Queries. `[batch, sequence, nheads, head_k_dim]`
    pub q_bshk: Tensor<4>,
    /// Keys. `[batch, sequence, nheads, head_k_dim]`
    pub k_bshk: Tensor<4>,
    /// Values. `[batch, sequence, nheads, head_v_dim]`
    pub v_bshv: Tensor<4>,
    /// Erase gate on the key axis: how much of the association currently held
    /// at `k` is removed. `β ∈ (0, 1)`, or `(0, 2)` with negative eigenvalues
    /// allowed. `[batch, sequence, nheads, K]`
    pub erase_bshK: Tensor<4>,
    /// Write gate on the value axis: how much of `v` is committed.
    /// `[batch, sequence, nheads, V]`
    pub write_bshV: Tensor<4>,
    /// Log forget gate `g = log α ≤ 0`, or `None` for no gate (`α ≡ 1`).
    /// `[batch, sequence, nheads, K]`
    pub g_bshK: Option<Tensor<4>>,
    /// Incoming state `S₀`. `[batch, nheads, head_k_dim, head_v_dim]`
    pub state_bhkv: Tensor<4>,
    /// Readout scale on `q`; `None` ⇒ `1/√head_k_dim`.
    pub scale: Option<f64>,
}

/// The `(erase, write)` pair a per-head `β` stands for: the same number
/// broadcast over both channel axes.
///
/// Shapes: `[batch, .., nheads]` → two of `[batch, .., nheads, 1]`.
pub fn beta_gates<const D: usize, const DP1: usize>(beta: Tensor<D>) -> (Tensor<DP1>, Tensor<DP1>) {
    let gate: Tensor<DP1> = beta.unsqueeze_dim(D);
    (gate.clone(), gate)
}

impl DeltaInput {
    /// Dimensions `(batch, sequence, nheads, head_k_dim, head_v_dim)`.
    pub fn dims(&self) -> (usize, usize, usize, usize, usize) {
        let [batch, sequence, nheads, head_k_dim] = self.q_bshk.dims();
        let [_b, _s, _h, head_v_dim] = self.v_bshv.dims();
        (batch, sequence, nheads, head_k_dim, head_v_dim)
    }

    /// The readout scale actually applied: [`Self::scale`] or `1/√head_k_dim`.
    pub fn resolved_scale(&self) -> f64 {
        let (_b, _s, _h, head_k_dim, _v) = self.dims();
        self.scale
            .unwrap_or_else(|| 1.0 / (head_k_dim as f64).sqrt())
    }

    /// Whether the forget gate varies per key channel rather than being one
    /// number per head — the only place the two gate widths take different
    /// code paths.
    pub fn has_channel_decay(&self) -> bool {
        self.g_bshK.as_ref().is_some_and(|g| g.dims()[3] > 1)
    }

    /// Check every shape agrees, and run the
    /// [`NaN`/`Inf` guards](burn_stack::modules::misc::sanity).
    #[allow(non_snake_case)]
    pub fn sanity(&self) {
        let (batch, sequence, nheads, head_k_dim, head_v_dim) = self.dims();
        assert!(sequence > 0, "sequence length must be at least 1");
        assert_eq!([batch, sequence, nheads, head_k_dim], self.k_bshk.dims());
        assert_eq!([batch, sequence, nheads, head_v_dim], self.v_bshv.dims());
        assert_eq!(
            [batch, nheads, head_k_dim, head_v_dim],
            self.state_bhkv.dims()
        );
        // A gate axis is either shared across the head (width 1) or per channel.
        let gate_axis = |t: &Tensor<4>, full: usize, what: &str| {
            let dims = t.dims();
            assert_eq!([batch, sequence, nheads], [dims[0], dims[1], dims[2]], "{what}");
            assert!(
                dims[3] == 1 || dims[3] == full,
                "{what}: last axis must be 1 or {full}, got {}",
                dims[3],
            );
        };
        gate_axis(&self.erase_bshK, head_k_dim, "erase gate");
        gate_axis(&self.write_bshV, head_v_dim, "write gate");
        san(&self.q_bshk);
        san(&self.k_bshk);
        san(&self.v_bshv);
        san(&self.erase_bshK);
        san(&self.write_bshV);
        san(&self.state_bhkv);
        if let Some(g_bshK) = &self.g_bshK {
            gate_axis(g_bshK, head_k_dim, "forget gate");
            san(g_bshK);
        }
    }

    /// Run the selected algorithm.
    ///
    /// # Returns
    /// - `y_bshv`: `[batch, sequence, nheads, head_v_dim]`
    /// - `final_state_bhkv`: `[batch, nheads, head_k_dim, head_v_dim]`
    pub fn run(self, path: DeltaPath) -> (Tensor<4>, Tensor<4>) {
        self.sanity();
        let resolve = |chunk_len: Option<usize>| {
            let chunk_len = chunk_len.unwrap_or(DeltaPath::DEFAULT_CHUNK_LEN);
            assert!(chunk_len > 0, "chunk_len must be at least 1");
            chunk_len
        };
        match path {
            DeltaPath::Recurrent => self.delta_recurrent(),
            DeltaPath::Chunk { chunk_len, solve } => {
                self.delta_chunk(resolve(chunk_len), solve)
            }
            DeltaPath::ChunkRecalculated { chunk_len } => {
                self.delta_chunk_recalculated(resolve(chunk_len))
            }
        }
    }
}
