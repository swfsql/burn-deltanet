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
    /// Chunkwise WY, backward via autodiff.
    Chunk {
        /// Tokens per chunk; `None` ⇒ [`DeltaPath::DEFAULT_CHUNK_LEN`].
        chunk_len: Option<usize>,
        /// How the WY transform's `(I − N)⁻¹` is evaluated.
        solve: TriSolve,
    },
}

impl Default for DeltaPath {
    fn default() -> Self {
        Self::Chunk {
            chunk_len: None,
            solve: TriSolve::Doubling,
        }
    }
}

impl DeltaPath {
    /// The chunk length used when a [`DeltaPath::Chunk`] carries `None`.
    pub const DEFAULT_CHUNK_LEN: usize = 64;

    /// Chunkwise with the default chunk length and triangular solve.
    pub fn chunk() -> Self {
        Self::default()
    }

    /// Chunkwise with an explicit chunk length.
    pub fn chunk_len(chunk_len: usize) -> Self {
        Self::Chunk {
            chunk_len: Some(chunk_len),
            solve: TriSolve::Doubling,
        }
    }

    /// The chunk length this variant runs at ([`Self::Recurrent`] answers 1 —
    /// it *is* the chunk length of a token-at-a-time pass).
    pub fn resolved_chunk_len(&self) -> usize {
        match self {
            Self::Recurrent => 1,
            Self::Chunk { chunk_len, .. } => chunk_len.unwrap_or(Self::DEFAULT_CHUNK_LEN),
        }
    }
}

/// Everything the delta-rule core consumes, per head.
///
/// `q`/`k` arrive **already activated and normalised** (see
/// [`crate::common::norm`]) but **not** scaled: [`Self::scale`] is applied to
/// `q` inside, so the same bundle reads identically on either path. `β` and `g`
/// arrive already squashed to their ranges — the core applies no `sigmoid`,
/// `softplus` or `exp` of its own beyond `exp(g)`.
pub struct DeltaInput {
    /// Queries. `[batch, sequence, nheads, head_k_dim]`
    pub q_bshk: Tensor<4>,
    /// Keys. `[batch, sequence, nheads, head_k_dim]`
    pub k_bshk: Tensor<4>,
    /// Values. `[batch, sequence, nheads, head_v_dim]`
    pub v_bshv: Tensor<4>,
    /// Write strength `β ∈ (0, 1)`, or `(0, 2)` with negative eigenvalues
    /// allowed. `[batch, sequence, nheads]`
    pub beta_bsh: Tensor<3>,
    /// Log forget gate `g = log α ≤ 0`, or `None` for no gate (`α ≡ 1`).
    /// `[batch, sequence, nheads]`
    pub g_bsh: Option<Tensor<3>>,
    /// Incoming state `S₀`. `[batch, nheads, head_k_dim, head_v_dim]`
    pub state_bhkv: Tensor<4>,
    /// Readout scale on `q`; `None` ⇒ `1/√head_k_dim`.
    pub scale: Option<f64>,
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

    /// Check every shape agrees, and run the
    /// [`NaN`/`Inf` guards](burn_stack::modules::misc::sanity).
    pub fn sanity(&self) {
        let (batch, sequence, nheads, head_k_dim, head_v_dim) = self.dims();
        assert!(sequence > 0, "sequence length must be at least 1");
        assert_eq!([batch, sequence, nheads, head_k_dim], self.k_bshk.dims());
        assert_eq!([batch, sequence, nheads, head_v_dim], self.v_bshv.dims());
        assert_eq!([batch, sequence, nheads], self.beta_bsh.dims());
        assert_eq!(
            [batch, nheads, head_k_dim, head_v_dim],
            self.state_bhkv.dims()
        );
        san(&self.q_bshk);
        san(&self.k_bshk);
        san(&self.v_bshv);
        san(&self.beta_bsh);
        san(&self.state_bhkv);
        if let Some(g_bsh) = &self.g_bsh {
            assert_eq!([batch, sequence, nheads], g_bsh.dims());
            san(g_bsh);
        }
    }

    /// Run the selected algorithm.
    ///
    /// # Returns
    /// - `y_bshv`: `[batch, sequence, nheads, head_v_dim]`
    /// - `final_state_bhkv`: `[batch, nheads, head_k_dim, head_v_dim]`
    pub fn run(self, path: DeltaPath) -> (Tensor<4>, Tensor<4>) {
        self.sanity();
        match path {
            DeltaPath::Recurrent => self.delta_recurrent(),
            DeltaPath::Chunk { chunk_len, solve } => {
                let chunk_len = chunk_len.unwrap_or(DeltaPath::DEFAULT_CHUNK_LEN);
                assert!(chunk_len > 0, "chunk_len must be at least 1");
                self.delta_chunk(chunk_len, solve)
            }
        }
    }
}
