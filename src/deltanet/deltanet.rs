//! # DeltaNet — the delta rule with no forget gate
//!
//! *Parallelizing Linear Transformers with the Delta Rule over Sequence Length*
//! (Yang, Wang, Zhang, Kim, Shen, Gu; 2024), which made the fast-weight
//! delta rule of *Linear Transformers Are Secretly Fast Weight Programmers*
//! (Schlag, Irie, Schmidhuber; 2021) trainable at scale.
//!
//! ## The block
//!
//! ```text
//!   x → in_proj → [ q | k | v | β | gate? ]
//!                   └── short conv + SiLU ──┘
//!     → head split → QK-norm(q, k), σ(β)
//!     → delta rule:  Sₜ = (I − βₜ kₜ kₜᵀ) Sₜ₋₁ + βₜ kₜ vₜᵀ,  yₜ = Sₜᵀ qₜ/√k
//!     → per-head RMSNorm (gated by `gate`)
//!     → out_proj
//! ```
//!
//! Two things distinguish it from the linear attention it generalises, and both
//! matter more than they look:
//!
//! - **The transition is a matrix.** Linear attention's state update is
//!   `S ← S + k vᵀ` — writes accumulate and collide. DeltaNet's is
//!   `S ← (I − β k kᵀ) S + β k vᵀ`: it *removes* the value currently associated
//!   with `k` before writing the new one. With `‖k‖ = 1` this is a generalised
//!   Householder with spectrum `{1, 1 − β}`, so nothing outside `span(k)` is
//!   disturbed. That bounded, targeted overwrite is what the associative-recall
//!   results rest on.
//! - **There is no decay.** `α ≡ 1`, so the state is only ever changed
//!   *deliberately*, by a write. [Gated DeltaNet](crate::gated_deltanet) adds
//!   the scalar forget gate back; that is the entire difference between the two
//!   families.
//!
//! ## Where the compute goes
//!
//! `forward` runs the [chunkwise WY algorithm](crate::delta::chunk) — serial
//! over chunks, batched GEMMs within — and `step` the
//! [recurrence](crate::delta::recurrent) directly, at `O(nheads · head_k_dim ·
//! head_v_dim)` per token with no growing KV cache. The two are the same
//! function; the test suite asserts it on outputs, cache and gradients.
//!
//! ## Notation
//!
//! See the [`delta`](crate::delta) module header for the dimension keys.

use burn::module::Module;
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::prelude::*;
use burn_stack::modules::sanity as san;

use crate::common::norm::{OutNorm, QkActivation, QkNorm};
use crate::common::qkv::{QkvProjection, QkvProjectionConfig, WriteGate};
use crate::delta::path::{DeltaInput, DeltaPath};
use crate::delta::recurrent::delta_step;
use crate::deltanet::cache::{DeltaNetCache, DeltaNetCacheConfig, DeltaNetCaches, DeltaNetCachesConfig};

// ---------------------------------------------------------------------------
// DeltaNet  (the block)
// ---------------------------------------------------------------------------

/// The DeltaNet block.
///
/// - [`Self::forward`] — chunkwise, for training and prefill.
/// - [`Self::step`] — recurrent, for token-by-token decoding.
#[derive(Module, Debug)]
pub struct DeltaNet {
    /// Fused `[q | k | v | β | gate?]` projection, short convolution, QK-norm.
    pub qkv: QkvProjection,
    /// Per-head output RMSNorm, gated iff the projection produces a gate.
    pub norm: OutNorm,
    /// `value_dim → d_model`.
    pub out_proj: Linear,
}

impl DeltaNet {
    /// Number of heads the recurrent state carries. Equals the projected
    /// query/key head count — DeltaNet does not group values.
    pub fn nheads(&self) -> usize {
        self.qkv.state_heads()
    }
    /// Query/key width per head — the state's row rank.
    pub fn head_k_dim(&self) -> usize {
        self.qkv.head_k_dim
    }
    /// Value width per head — the state's column rank.
    pub fn head_v_dim(&self) -> usize {
        self.qkv.head_v_dim
    }
    /// `nheads · head_k_dim`.
    pub fn key_dim(&self) -> usize {
        self.qkv.key_dim()
    }
    /// `nheads · head_v_dim`.
    pub fn value_dim(&self) -> usize {
        self.qkv.value_dim()
    }
    /// Model width.
    pub fn d_model(&self) -> usize {
        let [d_model, _out] = self.qkv.in_proj.weight.dims();
        d_model
    }

    /// Zero caches for `n_virtual` layers at this batch size.
    pub fn zero_caches(&self, batch: usize, n_virtual: usize, device: &Device) -> DeltaNetCaches {
        DeltaNetCachesConfig::new(
            n_virtual,
            DeltaNetCacheConfig {
                batch,
                nheads: self.nheads(),
                head_k_dim: self.head_k_dim(),
                head_v_dim: self.head_v_dim(),
                conv_dim: self.qkv.conv_dim(),
                conv_kernel: self.qkv.conv_kernel(),
            },
        )
        .init(device)
    }

    fn zero_cache(&self, batch: usize, device: &Device) -> DeltaNetCache {
        DeltaNetCache {
            conv_bwc: self.qkv.zero_conv_window(batch, device),
            state_bhkv: Tensor::zeros(
                Shape::new([batch, self.nheads(), self.head_k_dim(), self.head_v_dim()]),
                device,
            ),
        }
    }

    /// Process a full sequence with the chunkwise WY algorithm.
    ///
    /// # Shapes
    /// - `input_bsd`: `[batch, sequence, d_model]`
    /// - output: `[batch, sequence, d_model]`
    pub fn forward(
        &self,
        input_bsd: Tensor<3>,
        cache: Option<DeltaNetCache>,
        path: DeltaPath,
    ) -> (Tensor<3>, DeltaNetCache) {
        let [batch, sequence, d_model] = input_bsd.dims();
        assert_eq!(d_model, self.d_model());
        san(&input_bsd);

        let cache = cache.unwrap_or_else(|| self.zero_cache(batch, &input_bsd.device()));
        cache.sanity();
        let DeltaNetCache {
            conv_bwc,
            state_bhkv,
        } = cache;

        let (qkv, next_conv_bwc) = self.qkv.forward(input_bsd, conv_bwc);

        // No forget gate: `g = None` is what makes this DeltaNet rather than
        // Gated DeltaNet — the core then builds no decay tensors at all.
        let (y_bshv, next_state_bhkv) = DeltaInput {
            q_bshk: qkv.q_bshk,
            k_bshk: qkv.k_bShk,
            v_bshv: qkv.v_bShv,
            erase_bshK: qkv.erase_bShK,
            write_bshV: qkv.write_bShV,
            g_bshK: None,
            state_bhkv,
            scale: None,
        }
        .run(path);
        assert_eq!(
            [batch, sequence, self.nheads(), self.head_v_dim()],
            y_bshv.dims()
        );

        let y_bshv = self.norm.forward(y_bshv, qkv.gate_bshv);
        let out_bsd = self
            .out_proj
            .forward(y_bshv.reshape([batch, sequence, self.value_dim()]));
        assert_eq!([batch, sequence, d_model], out_bsd.dims());
        san(&out_bsd);

        (
            out_bsd,
            DeltaNetCache {
                conv_bwc: next_conv_bwc,
                state_bhkv: next_state_bhkv,
            },
        )
    }

    /// Process a single token with the recurrent form.
    ///
    /// # Shapes
    /// - `input_bd`: `[batch, d_model]`
    /// - output: `[batch, d_model]`
    pub fn step(
        &self,
        input_bd: Tensor<2>,
        cache: Option<DeltaNetCache>,
    ) -> (Tensor<2>, DeltaNetCache) {
        let [batch, d_model] = input_bd.dims();
        assert_eq!(d_model, self.d_model());

        let cache = cache.unwrap_or_else(|| self.zero_cache(batch, &input_bd.device()));
        let DeltaNetCache {
            conv_bwc,
            state_bhkv,
        } = cache;

        let (qkv, next_conv_bwc) = self.qkv.step(input_bd, conv_bwc);
        // `n_householder == 1`, so the micro-step axis is a singleton here.
        let (y_bhv, next_state_bhkv) = delta_step(
            qkv.q_bhk,
            qkv.k_buhk.squeeze_dim(1),
            qkv.v_buhv.squeeze_dim(1),
            qkv.erase_buhK.squeeze_dim(1),
            qkv.write_buhV.squeeze_dim(1),
            None,
            state_bhkv,
            1.0 / (self.head_k_dim() as f64).sqrt(),
        );

        let y_bhv = self.norm.forward(y_bhv, qkv.gate_bhv);
        let out_bd = self
            .out_proj
            .forward(y_bhv.reshape([batch, self.value_dim()]));
        assert_eq!([batch, d_model], out_bd.dims());
        san(&out_bd);

        (
            out_bd,
            DeltaNetCache {
                conv_bwc: next_conv_bwc,
                state_bhkv: next_state_bhkv,
            },
        )
    }
}

// ---------------------------------------------------------------------------
// DeltaNetConfig
// ---------------------------------------------------------------------------

/// Hyperparameters for [`DeltaNet`].
///
/// The key/value widths are given as *expansion ratios of `d_model`*, which is
/// how the paper and the reference implementation parameterise the block.
#[derive(Config, Debug)]
pub struct DeltaNetConfig {
    /// Model width.
    pub d_model: usize,

    /// Number of heads. Must divide both `key_dim` and `value_dim`.
    #[config(default = 4)]
    pub nheads: usize,

    /// `key_dim = expand_k · d_model` — the total query/key width, and with it
    /// the state's row rank.
    #[config(default = 1.0)]
    pub expand_k: f64,

    /// `value_dim = expand_v · d_model` — the total value width, and with it
    /// the state's column rank.
    #[config(default = 1.0)]
    pub expand_v: f64,

    /// Project `β`. With `false`, `β ≡ 1` and every write is a full
    /// replacement (`I − k kᵀ` is then a projection, not an interpolation).
    #[config(default = true)]
    pub use_beta: bool,

    /// Project an output gate and use a gated output RMSNorm.
    #[config(default = false)]
    pub use_gate: bool,

    /// Let `β` reach `(0, 2)` instead of `(0, 1)`, so the Householder can
    /// *reflect* rather than only contract. This is what
    /// [*Unlocking State-Tracking in Linear RNNs Through Negative
    /// Eigenvalues*](https://arxiv.org/abs/2411.12537) turns on.
    #[config(default = false)]
    pub allow_neg_eigval: bool,

    /// Use the causal short convolution. The reference warns loudly against
    /// disabling it: the recurrence is deliberately weak at local mixing.
    #[config(default = true)]
    pub use_short_conv: bool,

    /// Convolution window length.
    #[config(default = 4)]
    pub conv_kernel: usize,

    /// Whether the convolution carries a bias.
    #[config(default = false)]
    pub conv_bias: bool,

    /// Activation for `q`/`k`. Folded into the convolution when it is SiLU.
    #[config(default = "QkActivation::Silu")]
    pub qk_activation: QkActivation,

    /// Normalisation for `q`/`k`; `L2` is what bounds the Householder.
    #[config(default = "QkNorm::L2")]
    pub qk_norm: QkNorm,

    /// Whether `in_proj`/`out_proj` carry biases.
    #[config(default = false)]
    pub has_proj_bias: bool,
}

impl DeltaNetConfig {
    /// `key_dim = expand_k · d_model`.
    pub fn key_dim(&self) -> usize {
        scaled_dim(self.d_model, self.expand_k, "expand_k")
    }

    /// `value_dim = expand_v · d_model`.
    pub fn value_dim(&self) -> usize {
        scaled_dim(self.d_model, self.expand_v, "expand_v")
    }

    /// `head_k_dim = key_dim / nheads`.
    pub fn head_k_dim(&self) -> usize {
        self.key_dim() / self.nheads
    }

    /// `head_v_dim = value_dim / nheads`.
    pub fn head_v_dim(&self) -> usize {
        self.value_dim() / self.nheads
    }

    /// Allocate and initialise the block on `device`.
    pub fn init(&self, device: &Device) -> DeltaNet {
        let (key_dim, value_dim) = (self.key_dim(), self.value_dim());
        assert_eq!(
            key_dim % self.nheads,
            0,
            "key_dim ({key_dim}) must be divisible by nheads ({})",
            self.nheads,
        );
        assert_eq!(
            value_dim % self.nheads,
            0,
            "value_dim ({value_dim}) must be divisible by nheads ({})",
            self.nheads,
        );

        let qkv = QkvProjectionConfig::new(
            self.d_model,
            self.nheads,
            self.head_k_dim(),
            self.head_v_dim(),
        )
        .with_write_gate(if self.use_beta { WriteGate::Scalar } else { WriteGate::Fixed })
        .with_has_gate(self.use_gate)
        .with_allow_neg_eigval(self.allow_neg_eigval)
        .with_use_short_conv(self.use_short_conv)
        .with_conv_kernel(self.conv_kernel)
        .with_conv_bias(self.conv_bias)
        .with_qk_activation(self.qk_activation)
        .with_qk_norm(self.qk_norm)
        .with_has_proj_bias(self.has_proj_bias)
        .init(device);

        let bound = 1.0 / (value_dim as f64).sqrt();
        let out_proj = LinearConfig::new(value_dim, self.d_model)
            .with_bias(self.has_proj_bias)
            .with_initializer(Initializer::Uniform {
                min: -bound,
                max: bound,
            })
            .init(device);

        DeltaNet {
            qkv,
            norm: OutNorm::init(self.head_v_dim(), self.use_gate, device),
            out_proj,
        }
    }

    /// The block's 2-D weights Muon may own, and where their fused columns
    /// split.
    ///
    /// `in_proj` is one allocation holding independent maps, so it is listed
    /// segment by segment: `q`/`k`/`v`/`gate` are genuine matrices Muon should
    /// orthogonalise on their own, while `β`'s per-head scalar channels are a
    /// vector-valued map and stay on AdamW. The convolution weight is 3-D and
    /// is never listed. See [`burn_stack::optim`].
    #[cfg(feature = "optim")]
    pub fn muon_projections(&self) -> Vec<burn_stack::optim::ProjSpec> {
        use burn_stack::optim::{ProjSegment as Seg, ProjSpec};
        let mut segments = vec![
            Seg::muon("q", self.key_dim()),
            Seg::muon("k", self.key_dim()),
            Seg::muon("v", self.value_dim()),
        ];
        if self.use_beta {
            segments.push(Seg::adamw("beta", self.nheads));
        }
        if self.use_gate {
            segments.push(Seg::muon("gate", self.value_dim()));
        }
        vec![
            ProjSpec::block("qkv.in_proj.weight", segments),
            ProjSpec::block_whole("out_proj.weight", self.d_model),
        ]
    }
}

/// `round(d_model · ratio)`, asserting the ratio lands on a whole number of
/// channels (`expand_k = 0.5` at `d_model = 100` is fine; at `d_model = 101` it
/// is a configuration error, not a rounding decision to make silently).
fn scaled_dim(d_model: usize, ratio: f64, name: &str) -> usize {
    let exact = d_model as f64 * ratio;
    let rounded = exact.round();
    assert!(
        (exact - rounded).abs() < 1e-9,
        "{name} = {ratio} does not give a whole number of channels at d_model = {d_model} \
         (got {exact})",
    );
    assert!(rounded >= 1.0, "{name} = {ratio} gives no channels at all");
    rounded as usize
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
