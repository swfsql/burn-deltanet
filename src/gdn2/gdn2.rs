//! # GDN-2 — the delta rule with erase and write decoupled
//!
//! *Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention*. The
//! block is [Gated DeltaNet](crate::gated_deltanet) with every one of its
//! per-head gates widened onto a channel axis:
//!
//! ```text
//!   Gated DeltaNet: Sₜ = αₜ (I − βₜ kₜ kₜᵀ) Sₜ₋₁ + βₜ kₜ vₜᵀ        α, β scalars
//!   KDA:            Sₜ = (I − βₜ kₜ kₜᵀ) diag(αₜ) Sₜ₋₁ + βₜ kₜ vₜᵀ   α ∈ ℝ^k
//!   GDN-2:          Sₜ = (I − kₜ (bₜ ⊙ kₜ)ᵀ) diag(αₜ) Sₜ₋₁ + kₜ (wₜ ⊙ vₜ)ᵀ
//! ```
//!
//! with `b ∈ ℝ^head_k_dim` the **erase** gate and `w ∈ ℝ^head_v_dim` the
//! **write** gate. Setting `b = w = β` recovers KDA, and collapsing `α` to a
//! scalar as well recovers Gated DeltaNet — exactly, which this crate asserts
//! by running both from one set of weights.
//!
//! ## What the decoupling buys
//!
//! A scalar `β` makes one decision serve two jobs: `u = β(v − Sᵀk)` erases a
//! `β`-fraction of the association at `k` *and* commits a `β`-fraction of `v`.
//! The two corners a single number cannot reach are the useful ones:
//!
//! ```text
//!   b = 1, w = 0    delete: drop what is stored at k, write nothing
//!   b = 0, w = 1    accumulate: add v at k without disturbing what is there
//!   b = 1, w = 1    replace  (this is β = 1)
//! ```
//!
//! So "forget this fact" and "add to this fact" become single-token operations
//! rather than things the model has to approximate over several steps. Widening
//! them to *channels* rather than one scalar each additionally lets one part of
//! a value be overwritten while another part is left alone.
//!
//! The per-channel `α` is the KDA half: each row of the state gets its own
//! timescale, so a head can hold some keys for a long time while letting others
//! lapse, instead of decaying everything it knows at one rate.
//!
//! ## Shape
//!
//! Two low-rank bottlenecks, as in the reference: the per-channel `Δ` and the
//! output gate would each otherwise need a dense `d_model → key_dim` /
//! `d_model → value_dim` map, which at the deployed shape is the size of the
//! rest of the block. Both bottlenecks' *first* factors are ordinary segments
//! of the fused `in_proj`; their second factors are
//! [`ChannelForgetGate::up`] and [`GatedDeltaNet2::out_gate`].
//!
//! ## Notation
//!
//! See the [`delta`](crate::delta) module header for the dimension keys; `K`
//! and `V` are the gate axes.

use burn::module::Module;
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::prelude::*;
use burn_stack::modules::sanity as san;

use crate::common::gate::{ChannelForgetGate, ChannelForgetGateConfig};
use crate::common::norm::{OutNorm, QkActivation, QkNorm};
use crate::common::qkv::{QkvProjection, QkvProjectionConfig, WriteGate};
use crate::delta::path::{DeltaInput, DeltaPath};
use crate::delta::recurrent::delta_step;
use crate::gdn2::cache::{
    GatedDeltaNet2Cache, GatedDeltaNet2CacheConfig, GatedDeltaNet2Caches,
    GatedDeltaNet2CachesConfig,
};

// ---------------------------------------------------------------------------
// GatedDeltaNet2  (the block)
// ---------------------------------------------------------------------------

/// The GDN-2 block.
///
/// - [`Self::forward`] — chunkwise, for training and prefill.
/// - [`Self::step`] — recurrent, for token-by-token decoding.
#[derive(Module, Debug)]
pub struct GatedDeltaNet2 {
    /// Fused `[q | k | v | b | w | Δ_in (| gate_in)]` projection, short
    /// convolution, QK-norm. The trailing `extra` segment holds the two
    /// bottleneck inputs.
    pub qkv: QkvProjection,

    /// The per-key-channel forget gate. Always present — it is the family.
    pub gate: ChannelForgetGate,

    /// `bottleneck → value_dim`, the output gate's second factor. `None` when
    /// the block runs without an output gate.
    pub out_gate: Option<Linear>,

    /// Per-head output RMSNorm, gated iff [`Self::out_gate`] is present.
    pub norm: OutNorm,

    /// `value_dim → d_model`.
    pub out_proj: Linear,
}

impl GatedDeltaNet2 {
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
    /// Width of each low-rank bottleneck segment.
    pub fn bottleneck(&self) -> usize {
        let [bottleneck, _out] = self.gate.up.weight.dims();
        bottleneck
    }

    /// Zero caches for `n_virtual` layers at this batch size.
    pub fn zero_caches(
        &self,
        batch: usize,
        n_virtual: usize,
        device: &Device,
    ) -> GatedDeltaNet2Caches {
        GatedDeltaNet2CachesConfig::new(
            n_virtual,
            GatedDeltaNet2CacheConfig {
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

    fn zero_cache(&self, batch: usize, device: &Device) -> GatedDeltaNet2Cache {
        GatedDeltaNet2Cache {
            conv_bwc: self.qkv.zero_conv_window(batch, device),
            state_bhkv: Tensor::zeros(
                Shape::new([batch, self.nheads(), self.head_k_dim(), self.head_v_dim()]),
                device,
            ),
        }
    }

    /// Split the fused `extra` segment into the two bottleneck inputs.
    fn bottlenecks<const D: usize>(&self, extra: Option<Tensor<D>>) -> (Tensor<D>, Option<Tensor<D>>) {
        let extra = extra.expect("the bottleneck segments are always projected");
        let bottleneck = self.bottleneck();
        let mut sizes = vec![bottleneck];
        if self.out_gate.is_some() {
            sizes.push(bottleneck);
        }
        let mut parts = extra.split_with_sizes(sizes, D - 1).into_iter();
        let decay_in = parts.next().expect("the Δ bottleneck");
        (decay_in, parts.next())
    }

    /// Process a full sequence with the chunkwise WY algorithm.
    ///
    /// # Shapes
    /// - `input_bsd`: `[batch, sequence, d_model]`
    /// - output: `[batch, sequence, d_model]`
    #[allow(non_snake_case)]
    pub fn forward(
        &self,
        input_bsd: Tensor<3>,
        cache: Option<GatedDeltaNet2Cache>,
        path: DeltaPath,
    ) -> (Tensor<3>, GatedDeltaNet2Cache) {
        let [batch, sequence, d_model] = input_bsd.dims();
        assert_eq!(d_model, self.d_model());
        san(&input_bsd);

        let cache = cache.unwrap_or_else(|| self.zero_cache(batch, &input_bsd.device()));
        cache.sanity();
        let GatedDeltaNet2Cache {
            conv_bwc,
            state_bhkv,
        } = cache;

        let (qkv, next_conv_bwc) = self.qkv.forward(input_bsd, conv_bwc);
        let (decay_in_bsr, gate_in_bsr) = self.bottlenecks(qkv.extra_bsx);

        // The decay rides the query/key heads, so it is replicated across a
        // grouped-value group exactly as `q`, `k` and the erase gate are.
        let g_bshk: Tensor<4> = self
            .qkv
            .expand_to_value_heads::<4, 5>(self.gate.log_decay::<3, 4>(decay_in_bsr));
        assert_eq!(
            [batch, sequence, self.nheads(), self.head_k_dim()],
            g_bshk.dims()
        );

        let (y_bshv, next_state_bhkv) = DeltaInput {
            q_bshk: qkv.q_bshk,
            k_bshk: qkv.k_bShk,
            v_bshv: qkv.v_bShv,
            erase_bshK: qkv.erase_bShK,
            write_bshV: qkv.write_bShV,
            g_bshK: Some(g_bshk),
            state_bhkv,
            scale: None,
        }
        .run(path);
        assert_eq!(
            [batch, sequence, self.nheads(), self.head_v_dim()],
            y_bshv.dims()
        );

        let out_gate_bshv = self.out_gate.as_ref().map(|out_gate| {
            out_gate
                .forward(gate_in_bsr.expect("an output gate implies its bottleneck"))
                .reshape([batch, sequence, self.nheads(), self.head_v_dim()])
        });
        let y_bshv = self.norm.forward(y_bshv, out_gate_bshv);
        let out_bsd = self
            .out_proj
            .forward(y_bshv.reshape([batch, sequence, self.value_dim()]));
        assert_eq!([batch, sequence, d_model], out_bsd.dims());
        san(&out_bsd);

        (
            out_bsd,
            GatedDeltaNet2Cache {
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
    #[allow(non_snake_case)]
    pub fn step(
        &self,
        input_bd: Tensor<2>,
        cache: Option<GatedDeltaNet2Cache>,
    ) -> (Tensor<2>, GatedDeltaNet2Cache) {
        let [batch, d_model] = input_bd.dims();
        assert_eq!(d_model, self.d_model());

        let cache = cache.unwrap_or_else(|| self.zero_cache(batch, &input_bd.device()));
        let GatedDeltaNet2Cache {
            conv_bwc,
            state_bhkv,
        } = cache;

        let (qkv, next_conv_bwc) = self.qkv.step(input_bd, conv_bwc);
        let (decay_in_br, gate_in_br) = self.bottlenecks(qkv.extra_bx);
        let g_bhk: Tensor<3> = self
            .qkv
            .expand_to_value_heads::<3, 4>(self.gate.log_decay::<2, 3>(decay_in_br));

        // `n_householder == 1`, so the micro-step axis is a singleton here.
        let (y_bhv, next_state_bhkv) = delta_step(
            qkv.q_bhk,
            qkv.k_buhk.squeeze_dim(1),
            qkv.v_buhv.squeeze_dim(1),
            qkv.erase_buhK.squeeze_dim(1),
            qkv.write_buhV.squeeze_dim(1),
            Some(g_bhk),
            state_bhkv,
            1.0 / (self.head_k_dim() as f64).sqrt(),
        );

        let out_gate_bhv = self.out_gate.as_ref().map(|out_gate| {
            out_gate
                .forward(gate_in_br.expect("an output gate implies its bottleneck"))
                .reshape([batch, self.nheads(), self.head_v_dim()])
        });
        let y_bhv = self.norm.forward(y_bhv, out_gate_bhv);
        let out_bd = self
            .out_proj
            .forward(y_bhv.reshape([batch, self.value_dim()]));
        assert_eq!([batch, d_model], out_bd.dims());
        san(&out_bd);

        (
            out_bd,
            GatedDeltaNet2Cache {
                conv_bwc: next_conv_bwc,
                state_bhkv: next_state_bhkv,
            },
        )
    }
}

// ---------------------------------------------------------------------------
// GatedDeltaNet2Config
// ---------------------------------------------------------------------------

/// Hyperparameters for [`GatedDeltaNet2`].
///
/// Parameterised as [Gated DeltaNet](crate::gated_deltanet::gated_deltanet::GatedDeltaNetConfig)
/// is — an explicit `head_k_dim` and head count with the value width as an
/// expansion of it — plus [`Self::bottleneck`], the rank shared by the `Δ` and
/// output-gate projections.
#[derive(Config, Debug)]
pub struct GatedDeltaNet2Config {
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

    /// `head_v_dim = expand_v · head_k_dim`. GDN-2's own default is 1.
    #[config(default = 1.0)]
    pub expand_v: f64,

    /// Rank of the `Δ` and output-gate bottlenecks; `0` means `head_v_dim`,
    /// which is what the reference uses.
    #[config(default = 0)]
    pub bottleneck: usize,

    /// Project an output gate and use a gated output RMSNorm.
    #[config(default = true)]
    pub use_gate: bool,

    /// Let the **erase** gate reach `(0, 2)` instead of `(0, 1)`, so the
    /// Householder can *reflect* rather than only contract. The write gate is
    /// unaffected — that asymmetry is the point of decoupling them.
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
    #[config(default = "(1., 16.)")]
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

impl GatedDeltaNet2Config {
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

    /// The bottleneck rank actually used (`head_v_dim` when unset).
    pub fn bottleneck_resolved(&self) -> usize {
        if self.bottleneck == 0 {
            self.head_v_dim()
        } else {
            self.bottleneck
        }
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
    pub fn init(&self, device: &Device) -> GatedDeltaNet2 {
        let value_dim = self.value_dim();
        let bottleneck = self.bottleneck_resolved();
        let extra_channels = bottleneck + if self.use_gate { bottleneck } else { 0 };

        let qkv = QkvProjectionConfig::new(
            self.d_model,
            self.nheads,
            self.head_k_dim,
            self.head_v_dim(),
        )
        .with_n_value_heads(self.n_value_heads)
        .with_write_gate(WriteGate::Channel)
        // The output gate is low-rank here, so it does not ride the fused
        // projection as a full-width segment the way the other families' does.
        .with_has_gate(false)
        .with_allow_neg_eigval(self.allow_neg_eigval)
        .with_use_short_conv(self.use_short_conv)
        .with_conv_kernel(self.conv_kernel)
        .with_conv_bias(self.conv_bias)
        // GDN-2 always SiLU-activates and L2-normalises q/k.
        .with_qk_activation(QkActivation::Silu)
        .with_qk_norm(QkNorm::L2)
        // Both bottlenecks' first factors ride along in the fused projection.
        .with_extra_channels(extra_channels)
        .with_has_proj_bias(self.has_proj_bias)
        .init(device);

        let gate = ChannelForgetGateConfig::new(bottleneck, self.nheads, self.head_k_dim)
            .with_a_init_range(self.a_init_range)
            .with_dt_min(self.dt_min)
            .with_dt_max(self.dt_max)
            .with_dt_init_floor(self.dt_init_floor)
            .with_dt_limit(self.dt_limit)
            .init(device);

        let uniform = |fan_in: usize| {
            let bound = 1.0 / (fan_in as f64).sqrt();
            Initializer::Uniform {
                min: -bound,
                max: bound,
            }
        };
        let out_gate = self.use_gate.then(|| {
            LinearConfig::new(bottleneck, value_dim)
                .with_bias(true)
                .with_initializer(uniform(bottleneck))
                .init(device)
        });
        let out_proj = LinearConfig::new(value_dim, self.d_model)
            .with_bias(self.has_proj_bias)
            .with_initializer(uniform(value_dim))
            .init(device);

        GatedDeltaNet2 {
            qkv,
            gate,
            out_gate,
            norm: OutNorm::init(self.head_v_dim(), self.use_gate, device),
            out_proj,
        }
    }

    /// The block's 2-D weights Muon may own, and where their fused columns
    /// split.
    ///
    /// Every segment here is a genuine matrix, which is the practical
    /// difference from the scalar families: their `β` and `Δ` project *one
    /// number per head* and so stay on AdamW, while GDN-2's `b`, `w` and `Δ`
    /// project feature vectors. Only the 1-D `a_log`/`dt_bias`/`γ`, the
    /// output-gate bias and the 3-D convolution weight are left out. See
    /// [`burn_stack::optim`].
    #[cfg(feature = "optim")]
    pub fn muon_projections(&self) -> Vec<burn_stack::optim::ProjSpec> {
        use burn_stack::optim::{ProjSegment as Seg, ProjSpec};
        let bottleneck = self.bottleneck_resolved();
        let mut segments = vec![
            Seg::muon("q", self.key_dim()),
            Seg::muon("k", self.key_dim()),
            Seg::muon("v", self.value_dim()),
            Seg::muon("erase", self.key_dim()),
            Seg::muon("write", self.value_dim()),
            Seg::muon("dt_in", bottleneck),
        ];
        let mut specs = vec![];
        if self.use_gate {
            segments.push(Seg::muon("gate_in", bottleneck));
            specs.push(ProjSpec::block_whole("out_gate.weight", self.value_dim()));
        }
        specs.insert(0, ProjSpec::block("qkv.in_proj.weight", segments));
        specs.push(ProjSpec::block_whole("gate.up.weight", self.key_dim()));
        specs.push(ProjSpec::block_whole("out_proj.weight", self.d_model));
        specs
    }
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
