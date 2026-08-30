//! # Gated DeltaNet — the delta rule with a forget gate
//!
//! *Gated Delta Networks: Improving Mamba2 with Delta Rule* (Yang, Kautz,
//! Hatamizadeh; 2024). The block is [DeltaNet](crate::deltanet) plus a scalar
//! decay, and the paper's framing is exactly that pairing:
//!
//! ```text
//!   Mamba-2:          Sₜ = αₜ Sₜ₋₁ + kₜ vₜᵀ                  decay, no targeting
//!   DeltaNet:         Sₜ = (I − βₜ kₜ kₜᵀ) Sₜ₋₁ + βₜ kₜ vₜᵀ   targeting, no decay
//!   Gated DeltaNet:   Sₜ = αₜ (I − βₜ kₜ kₜᵀ) Sₜ₋₁ + βₜ kₜ vₜᵀ
//! ```
//!
//! The two mechanisms do different jobs and the paper's ablations separate
//! them: `α` erases *indiscriminately* (useful when the context has genuinely
//! moved on, e.g. across a document boundary), while `β kkᵀ` erases *one
//! association* (useful when a specific fact is being updated). Neither
//! subsumes the other, which is why the combination beats both.
//!
//! ## The gate
//!
//! `α` is produced exactly as Mamba-2 produces its decay — the same
//! parameterisation, deliberately:
//!
//! ```text
//!   Δₜ = softplus(a_projₜ + dt_bias)      per head, > 0
//!   A  = −exp(a_log)                       per head, < 0 by construction
//!   gₜ = Δₜ · A  ≤ 0                       the log decay
//!   αₜ = exp(gₜ) ∈ (0, 1]
//! ```
//!
//! Storing `log|A|` and negating makes `A < 0` unconditional, so no
//! sign-constraint is needed during descent and `α` can never exceed 1.
//! `dt_bias` is initialised by inverting the softplus over a log-uniform spread
//! of `Δ`, which spreads the heads' initial timescales over `[dt_min, dt_max]`
//! instead of starting them all together.
//!
//! ## Grouped values
//!
//! `n_value_heads > nheads` gives several value heads per query/key head — the
//! configuration Gated DeltaNet is deployed in. See [`crate::common::qkv`].
//!
//! ## Notation
//!
//! See the [`delta`](crate::delta) module header for the dimension keys.

use burn::module::Module;
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::prelude::*;
use burn_stack::modules::sanity as san;

use crate::common::gate::{ForgetGate, ForgetGateConfig};
use crate::common::norm::{OutNorm, QkActivation, QkNorm};
use crate::common::qkv::{QkvProjection, QkvProjectionConfig, WriteGate};
use crate::delta::path::{DeltaInput, DeltaPath};
use crate::delta::recurrent::delta_step;
use crate::common::cache::{DeltaCache, DeltaCacheConfig, DeltaCaches, DeltaCachesConfig};

// ---------------------------------------------------------------------------
// GatedDeltaNet1  (the block)
// ---------------------------------------------------------------------------

/// The Gated DeltaNet block.
///
/// - [`Self::forward`] — chunkwise, for training and prefill.
/// - [`Self::step`] — recurrent, for token-by-token decoding.
#[derive(Module, Debug)]
pub struct GatedDeltaNet1 {
    /// Fused `[q | k | v | β | gate? | Δ_raw]` projection, short convolution,
    /// QK-norm. The trailing `extra` segment is the forget gate's `Δ`.
    pub qkv: QkvProjection,

    /// The per-head scalar forget gate. Always present — it is the family.
    pub gate: ForgetGate,

    /// Per-head output RMSNorm, gated iff the projection produces a gate.
    pub norm: OutNorm,

    /// `value_dim → d_model`.
    pub out_proj: Linear,
}

impl GatedDeltaNet1 {
    /// Number of heads the recurrent state carries (the *value* head count).
    pub fn nheads(&self) -> usize {
        self.qkv.state_heads()
    }
    /// Number of projected query/key heads.
    pub fn n_qk_heads(&self) -> usize {
        self.qkv.nheads
    }
    /// Query/key width per head — the state's row rank.
    pub fn head_k_dim(&self) -> usize {
        self.qkv.head_k_dim
    }
    /// Value width per head — the state's column rank.
    pub fn head_v_dim(&self) -> usize {
        self.qkv.head_v_dim
    }
    /// `n_value_heads · head_v_dim`.
    pub fn value_dim(&self) -> usize {
        self.qkv.value_dim()
    }
    /// Model width.
    pub fn d_model(&self) -> usize {
        let [d_model, _out] = self.qkv.in_proj.weight.dims();
        d_model
    }

    /// Zero caches for `n_virtual` layers at this batch size.
    pub fn zero_caches(
        &self,
        batch: usize,
        n_virtual: usize,
        device: &Device,
    ) -> DeltaCaches {
        DeltaCachesConfig::new(
            n_virtual,
            DeltaCacheConfig {
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

    fn zero_cache(&self, batch: usize, device: &Device) -> DeltaCache {
        DeltaCache {
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
        cache: Option<DeltaCache>,
        path: DeltaPath,
    ) -> (Tensor<3>, DeltaCache) {
        let [batch, sequence, d_model] = input_bsd.dims();
        assert_eq!(d_model, self.d_model());
        san(&input_bsd);

        let cache = cache.unwrap_or_else(|| self.zero_cache(batch, &input_bsd.device()));
        cache.sanity();
        let DeltaCache {
            conv_bwc,
            state_bhkv,
        } = cache;

        let (qkv, next_conv_bwc) = self.qkv.forward(input_bsd, conv_bwc);
        let dt_raw_bsh = qkv.extra_bsx.expect("the forget-gate segment is always projected");
        let g_bsh = self.gate.log_decay(dt_raw_bsh);
        assert_eq!([batch, sequence, self.nheads()], g_bsh.dims());

        let (y_bshv, next_state_bhkv) = DeltaInput {
            q_bshk: qkv.q_bshk,
            k_bshk: qkv.k_bShk,
            v_bshv: qkv.v_bShv,
            erase_bshK: qkv.erase_bShK,
            write_bshV: qkv.write_bShV,
            g_bshK: Some(g_bsh.unsqueeze_dim::<4>(3)),
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
            DeltaCache {
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
        cache: Option<DeltaCache>,
    ) -> (Tensor<2>, DeltaCache) {
        let [batch, d_model] = input_bd.dims();
        assert_eq!(d_model, self.d_model());

        let cache = cache.unwrap_or_else(|| self.zero_cache(batch, &input_bd.device()));
        let DeltaCache {
            conv_bwc,
            state_bhkv,
        } = cache;

        let (qkv, next_conv_bwc) = self.qkv.step(input_bd, conv_bwc);
        let dt_raw_bh = qkv.extra_bx.expect("the forget-gate segment is always projected");
        let g_bh = self.gate.log_decay(dt_raw_bh);

        // `n_householder == 1`, so the micro-step axis is a singleton here.
        let (y_bhv, next_state_bhkv) = delta_step(
            qkv.q_bhk,
            qkv.k_buhk.squeeze_dim(1),
            qkv.v_buhv.squeeze_dim(1),
            qkv.erase_buhK.squeeze_dim(1),
            qkv.write_buhV.squeeze_dim(1),
            Some(g_bh.unsqueeze_dim::<3>(2)),
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
            DeltaCache {
                conv_bwc: next_conv_bwc,
                state_bhkv: next_state_bhkv,
            },
        )
    }
}

// ---------------------------------------------------------------------------
// GatedDeltaNet1Config
// ---------------------------------------------------------------------------

/// Hyperparameters for [`GatedDeltaNet1`].
///
/// Parameterised the way the paper and the reference are: an explicit
/// `head_k_dim` and head count, with the value width as an expansion of it.
/// At the reference's defaults the block lands at roughly `6·d_model²`
/// parameters, the same budget as a Transformer layer.
#[derive(Config, Debug)]
pub struct GatedDeltaNet1Config {
    /// Model width.
    pub d_model: usize,

    /// Number of query/key heads.
    #[config(default = 4)]
    pub nheads: usize,

    /// Number of value heads; `0` means "same as `nheads`". A larger multiple
    /// gives grouped values: several value heads share one query/key head.
    #[config(default = 0)]
    pub n_value_heads: usize,

    /// Query/key width per head — the state's row rank.
    #[config(default = 64)]
    pub head_k_dim: usize,

    /// `head_v_dim = expand_v · head_k_dim`.
    #[config(default = 2.0)]
    pub expand_v: f64,

    /// Project an output gate and use a gated output RMSNorm. On by default,
    /// unlike [DeltaNet](crate::deltanet).
    #[config(default = true)]
    pub use_gate: bool,

    /// Let `β` reach `(0, 2)` instead of `(0, 1)`, so the Householder can
    /// *reflect* rather than only contract.
    #[config(default = false)]
    pub allow_neg_eigval: bool,

    /// Use the causal short convolution.
    #[config(default = true)]
    pub use_short_conv: bool,

    /// Convolution window length.
    #[config(default = 4)]
    pub conv_kernel: usize,

    /// Whether the convolution carries a bias.
    #[config(default = false)]
    pub conv_bias: bool,

    /// Range `[lo, hi]` for the uniform initialisation of `|A|`, stored as
    /// `a_log = log(Uniform(lo, hi))`.
    #[config(default = "(0., 16.)")]
    pub a_init_range: (f64, f64),

    /// Minimum of the initial `Δ` spread; sets `dt_bias`.
    #[config(default = 1e-3)]
    pub dt_min: f64,

    /// Maximum of the initial `Δ` spread; sets `dt_bias`.
    #[config(default = 0.1)]
    pub dt_max: f64,

    /// Floor clamped onto the sampled initial `Δ` before inverting the
    /// softplus.
    #[config(default = 1e-4)]
    pub dt_init_floor: f64,

    /// Hard clamp on `Δ` at runtime. The default only clamps at zero (the
    /// upper bound is f16's maximum).
    #[config(default = "(0., 6.5504e+4)")]
    pub dt_limit: (f64, f64),

    /// Whether `in_proj`/`out_proj` carry biases.
    #[config(default = false)]
    pub has_proj_bias: bool,
}

impl GatedDeltaNet1Config {
    /// The value head count actually used (`nheads` when unset).
    pub fn n_value_heads_resolved(&self) -> usize {
        if self.n_value_heads == 0 {
            self.nheads
        } else {
            self.n_value_heads
        }
    }

    /// `head_v_dim = expand_v · head_k_dim`.
    pub fn head_v_dim(&self) -> usize {
        let exact = self.head_k_dim as f64 * self.expand_v;
        let rounded = exact.round();
        assert!(
            (exact - rounded).abs() < 1e-9,
            "expand_v = {} does not give a whole head_v_dim at head_k_dim = {} (got {exact})",
            self.expand_v,
            self.head_k_dim,
        );
        rounded as usize
    }

    /// `nheads · head_k_dim`.
    pub fn key_dim(&self) -> usize {
        self.nheads * self.head_k_dim
    }

    /// `n_value_heads · head_v_dim`.
    pub fn value_dim(&self) -> usize {
        self.n_value_heads_resolved() * self.head_v_dim()
    }

    /// Allocate and initialise the block on `device`.
    pub fn init(&self, device: &Device) -> GatedDeltaNet1 {
        let nheads_v = self.n_value_heads_resolved();
        let value_dim = self.value_dim();

        let qkv = QkvProjectionConfig::new(
            self.d_model,
            self.nheads,
            self.head_k_dim,
            self.head_v_dim(),
        )
        .with_n_value_heads(self.n_value_heads)
        .with_write_gate(WriteGate::Scalar)
        .with_has_gate(self.use_gate)
        .with_allow_neg_eigval(self.allow_neg_eigval)
        .with_use_short_conv(self.use_short_conv)
        .with_conv_kernel(self.conv_kernel)
        .with_conv_bias(self.conv_bias)
        // Gated DeltaNet always SiLU-activates and L2-normalises q/k.
        .with_qk_activation(QkActivation::Silu)
        .with_qk_norm(QkNorm::L2)
        // The forget gate's raw Δ rides along in the fused projection.
        .with_extra_channels(nheads_v)
        .with_has_proj_bias(self.has_proj_bias)
        .init(device);

        let gate = ForgetGateConfig::new(nheads_v)
            .with_a_init_range(self.a_init_range)
            .with_dt_min(self.dt_min)
            .with_dt_max(self.dt_max)
            .with_dt_init_floor(self.dt_init_floor)
            .with_dt_limit(self.dt_limit)
            .init(device);

        let bound = 1.0 / (value_dim as f64).sqrt();
        let out_proj = LinearConfig::new(value_dim, self.d_model)
            .with_bias(self.has_proj_bias)
            .with_initializer(Initializer::Uniform {
                min: -bound,
                max: bound,
            })
            .init(device);

        GatedDeltaNet1 {
            qkv,
            gate,
            norm: OutNorm::init(self.head_v_dim(), self.use_gate, device),
            out_proj,
        }
    }

    /// The block's 2-D weights Muon may own, and where their fused columns
    /// split.
    ///
    /// `q`/`k`/`v`/`gate` are independent matrices sharing one allocation and
    /// are orthogonalised separately; `β` and the forget gate's `Δ` are
    /// per-head *scalar* channels — vector-valued maps, not matrices — and stay
    /// on AdamW, as do the 1-D `a_log`/`dt_bias` and the 3-D convolution
    /// weight. See [`burn_stack::optim`].
    #[cfg(feature = "optim")]
    pub fn muon_projections(&self) -> Vec<burn_stack::optim::ProjSpec> {
        use burn_stack::optim::{ProjSegment as Seg, ProjSpec};
        let nheads_v = self.n_value_heads_resolved();
        let mut segments = vec![
            Seg::muon("q", self.key_dim()),
            Seg::muon("k", self.key_dim()),
            Seg::muon("v", self.value_dim()),
            Seg::adamw("beta", nheads_v),
        ];
        if self.use_gate {
            segments.push(Seg::muon("gate", self.value_dim()));
        }
        segments.push(Seg::adamw("dt", nheads_v));
        vec![
            ProjSpec::block("qkv.in_proj.weight", segments),
            ProjSpec::block_whole("out_proj.weight", self.d_model),
        ]
    }
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
