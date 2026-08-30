//! # DeltaProduct — a *product* of Householders per transition
//!
//! *DeltaProduct: Improving State-Tracking in Linear RNNs via Householder
//! Products* (Siems, Carstensen, Zela, Hutter, Pontil, Grazzi; 2025).
//!
//! ## Why more than one Householder
//!
//! [DeltaNet](crate::deltanet)'s transition is a *single* generalised
//! Householder, `I − β k kᵀ`. That is a rank-1 perturbation of the identity, so
//! whatever the parameters do, one step can only ever act non-trivially along
//! one direction. With `β ∈ (0, 2)` it reaches reflections, which is already
//! enough for parity — but not for a group that needs two independent
//! directions moved at once.
//!
//! DeltaProduct takes `u = n_householder` delta-rule micro-steps per token, so
//! the transition becomes
//!
//! ```text
//!   Sₜ = (I − βₜ,ᵤ₋₁ kₜ,ᵤ₋₁ kₜ,ᵤ₋₁ᵀ) ⋯ (I − βₜ,₀ kₜ,₀ kₜ,₀ᵀ) · αₜ Sₜ₋₁ + writes
//! ```
//!
//! A product of `u` Householder reflections spans the orthogonal
//! transformations of a `u`-dimensional subspace — every element of `O(u)` is
//! such a product, by the Cartan–Dieudonné theorem. So `u` is a direct dial on
//! how much group structure one transition can track, at `u`× the recurrence
//! work and no extra state. `u = 1` *is* [Gated DeltaNet](crate::gated_deltanet_1)
//! — exactly, not approximately, which the test suite asserts by running the
//! two from one set of weights.
//!
//! This is the same trade the [Mamba-3](https://arxiv.org/abs/2603.15569)
//! rotation kinds make from the other side: there the transition is given a
//! rotation group directly, here it is *factored* into reflections the data
//! chooses.
//!
//! ## How it is evaluated
//!
//! Not with a new kernel. The micro-steps are folded into the sequence — `k`,
//! `v` and `β` are projected `u`-wide and unrolled to length `sequence · u` by
//! [`crate::common::qkv`] — and then the ordinary delta rule runs over the
//! longer sequence. Two placements make that exactly the recurrence above:
//!
//! - **The query sits on the last micro-step** (`q = 0` elsewhere), so the
//!   readout happens after all `u` writes and the intervening micro-steps emit
//!   nothing. Their outputs are sliced away.
//! - **The forget gate sits on the first** (`g = 0` elsewhere), so `α` is
//!   applied once per *token*, before the Householder product, rather than once
//!   per micro-step.
//!
//! Both are `cat` of a zero block with the real tensor — there is no masking
//! and no branch in the core.
//!
//! ## Notation
//!
//! See the [`delta`](crate::delta) module header for the dimension keys; `u` is
//! `n_householder`.

use burn::module::Module;
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::prelude::*;
use burn_stack::modules::sanity as san;

use crate::common::gate::{ForgetGate, ForgetGateConfig};
use crate::common::norm::{OutNorm, QkActivation, QkNorm};
use crate::common::qkv::{QkvProjection, QkvProjectionConfig, WriteGate};
use crate::delta::path::{DeltaInput, DeltaPath};
use crate::common::cache::{DeltaCache, DeltaCacheConfig, DeltaCaches, DeltaCachesConfig};

// ---------------------------------------------------------------------------
// DeltaProduct  (the block)
// ---------------------------------------------------------------------------

/// The DeltaProduct block.
///
/// - [`Self::forward`] — chunkwise, for training and prefill.
/// - [`Self::step`] — recurrent, for token-by-token decoding.
#[derive(Module, Debug)]
pub struct DeltaProduct {
    /// Fused projection: `q` once, `k`/`v`/`β` `n_householder` times, plus the
    /// optional output gate and the forget gate's raw `Δ`.
    pub qkv: QkvProjection,

    /// The per-head scalar forget gate, applied once per token. `None` runs the
    /// ungated product (`α ≡ 1`), i.e. DeltaNet's transition raised to `u`
    /// factors.
    pub gate: Option<ForgetGate>,

    /// Per-head output RMSNorm, gated iff the projection produces a gate.
    pub norm: OutNorm,

    /// `value_dim → d_model`.
    pub out_proj: Linear,
}

impl DeltaProduct {
    /// Householder factors per transition.
    pub fn n_householder(&self) -> usize {
        self.qkv.n_householder
    }
    /// Number of heads the recurrent state carries (the *value* head count).
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
    ///
    /// The state is the same size as any other family's: `u` buys transition
    /// expressiveness, not memory.
    pub fn zero_caches(&self, batch: usize, n_virtual: usize, device: &Device) -> DeltaCaches {
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

    /// Process a full sequence with the chunkwise WY algorithm, over the
    /// micro-step-unrolled sequence.
    ///
    /// # Shapes
    /// - `input_bsd`: `[batch, sequence, d_model]`
    /// - output: `[batch, sequence, d_model]`
    #[allow(non_snake_case)]
    pub fn forward(
        &self,
        input_bsd: Tensor<3>,
        cache: Option<DeltaCache>,
        path: DeltaPath,
    ) -> (Tensor<3>, DeltaCache) {
        let [batch, sequence, d_model] = input_bsd.dims();
        assert_eq!(d_model, self.d_model());
        let (u, nheads, head_v_dim) = (self.n_householder(), self.nheads(), self.head_v_dim());
        san(&input_bsd);

        let cache = cache.unwrap_or_else(|| self.zero_cache(batch, &input_bsd.device()));
        cache.sanity();
        let DeltaCache {
            conv_bwc,
            state_bhkv,
        } = cache;

        let (qkv, next_conv_bwc) = self.qkv.forward(input_bsd, conv_bwc);
        let device = qkv.q_bshk.device();

        // ── q on the last micro-step; zero on the others ────────────────────
        let q_bShk = if u == 1 {
            qkv.q_bshk
        } else {
            let [_b, _s, _h, head_k_dim] = qkv.q_bshk.dims();
            let quiet = Tensor::<5>::zeros(
                Shape::new([batch, sequence, u - 1, nheads, head_k_dim]),
                &device,
            );
            Tensor::cat(vec![quiet, qkv.q_bshk.unsqueeze_dim(2)], 2)
                .reshape([batch, sequence * u, nheads, head_k_dim])
        };

        // ── the forget gate on the first micro-step; zero on the others ─────
        let g_bSh = self.gate.as_ref().map(|gate| {
            let dt_raw_bsh = qkv
                .extra_bsx
                .clone()
                .expect("a gated block always projects Δ");
            let g_bsh = gate.log_decay(dt_raw_bsh);
            if u == 1 {
                g_bsh
            } else {
                let quiet = Tensor::<4>::zeros(Shape::new([batch, sequence, u - 1, nheads]), &device);
                Tensor::cat(vec![g_bsh.unsqueeze_dim(2), quiet], 2)
                    .reshape([batch, sequence * u, nheads])
            }
        });

        let (y_bShv, next_state_bhkv) = DeltaInput {
            q_bshk: q_bShk,
            k_bshk: qkv.k_bShk,
            v_bshv: qkv.v_bShv,
            erase_bshK: qkv.erase_bShK,
            write_bshV: qkv.write_bShV,
            g_bshK: g_bSh.map(|g| g.unsqueeze_dim::<4>(3)),
            state_bhkv,
            scale: None,
        }
        .run(path);

        // ── Keep only the last micro-step of each token ─────────────────────
        let y_bshv = if u == 1 {
            y_bShv
        } else {
            y_bShv
                .reshape([batch, sequence, u, nheads, head_v_dim])
                .narrow(2, u - 1, 1)
                .squeeze_dim(2)
        };
        assert_eq!([batch, sequence, nheads, head_v_dim], y_bshv.dims());

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

    /// Process a single token: `u` micro-steps of the recurrence, one readout.
    ///
    /// # Shapes
    /// - `input_bd`: `[batch, d_model]`
    /// - output: `[batch, d_model]`
    #[allow(non_snake_case)]
    pub fn step(
        &self,
        input_bd: Tensor<2>,
        cache: Option<DeltaCache>,
    ) -> (Tensor<2>, DeltaCache) {
        let [batch, d_model] = input_bd.dims();
        assert_eq!(d_model, self.d_model());
        let (u, nheads) = (self.n_householder(), self.nheads());

        let cache = cache.unwrap_or_else(|| self.zero_cache(batch, &input_bd.device()));
        let DeltaCache {
            conv_bwc,
            state_bhkv,
        } = cache;

        let (qkv, next_conv_bwc) = self.qkv.step(input_bd, conv_bwc);
        let device = qkv.q_bhk.device();

        // One token's `u` micro-steps are a length-`u` sequence for the core,
        // built exactly as `forward` builds them.
        let head_k_dim = self.head_k_dim();
        let q_buhk = if u == 1 {
            qkv.q_bhk.unsqueeze_dim(1)
        } else {
            let quiet = Tensor::<4>::zeros(Shape::new([batch, u - 1, nheads, head_k_dim]), &device);
            Tensor::cat(vec![quiet, qkv.q_bhk.unsqueeze_dim(1)], 1)
        };
        let g_buh = self.gate.as_ref().map(|gate| {
            let g_bh = gate.log_decay(
                qkv.extra_bx
                    .clone()
                    .expect("a gated block always projects Δ"),
            );
            if u == 1 {
                g_bh.unsqueeze_dim(1)
            } else {
                let quiet = Tensor::<3>::zeros(Shape::new([batch, u - 1, nheads]), &device);
                Tensor::cat(vec![g_bh.unsqueeze_dim(1), quiet], 1)
            }
        });

        let (y_buhv, next_state_bhkv) = DeltaInput {
            q_bshk: q_buhk,
            k_bshk: qkv.k_buhk,
            v_bshv: qkv.v_buhv,
            erase_bshK: qkv.erase_buhK,
            write_bshV: qkv.write_buhV,
            g_bshK: g_buh.map(|g| g.unsqueeze_dim::<4>(3)),
            state_bhkv,
            scale: None,
        }
        .run(DeltaPath::Recurrent);

        let y_bhv = y_buhv.narrow(1, u - 1, 1).squeeze_dim(1);
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
// DeltaProductConfig
// ---------------------------------------------------------------------------

/// Hyperparameters for [`DeltaProduct`].
///
/// The defaults are the reference's: two Householders, a forget gate, an output
/// gate, and — unlike the other two families — `allow_neg_eigval` **on**. A
/// product of reflections is the point; restricting `β` to `(0, 1)` would leave
/// each factor a contraction and throw away exactly what `u > 1` buys.
#[derive(Config, Debug)]
pub struct DeltaProductConfig {
    /// Model width.
    pub d_model: usize,

    /// Householder factors per transition. `1` is
    /// [Gated DeltaNet](crate::gated_deltanet_1) exactly.
    #[config(default = 2)]
    pub n_householder: usize,

    /// Number of query/key heads.
    #[config(default = 4)]
    pub nheads: usize,

    /// Number of value heads; `0` means "same as `nheads`".
    #[config(default = 0)]
    pub n_value_heads: usize,

    /// Query/key width per head — the state's row rank.
    #[config(default = 64)]
    pub head_k_dim: usize,

    /// `head_v_dim = expand_v · head_k_dim`.
    #[config(default = 1.0)]
    pub expand_v: f64,

    /// Apply the scalar forget gate (once per token, before the product).
    #[config(default = true)]
    pub use_forget_gate: bool,

    /// Project an output gate and use a gated output RMSNorm.
    #[config(default = true)]
    pub use_gate: bool,

    /// Let `β` reach `(0, 2)`, so each factor can reflect. On by default here.
    #[config(default = true)]
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

    /// Range `[lo, hi]` for the uniform initialisation of `|A|`.
    #[config(default = "(0., 16.)")]
    pub a_init_range: (f64, f64),

    /// Minimum of the initial `Δ` spread.
    #[config(default = 1e-3)]
    pub dt_min: f64,

    /// Maximum of the initial `Δ` spread.
    #[config(default = 0.1)]
    pub dt_max: f64,

    /// Floor clamped onto the sampled initial `Δ`.
    #[config(default = 1e-4)]
    pub dt_init_floor: f64,

    /// Hard clamp on `Δ` at runtime.
    #[config(default = "(0., 6.5504e+4)")]
    pub dt_limit: (f64, f64),

    /// Whether `in_proj`/`out_proj` carry biases.
    #[config(default = false)]
    pub has_proj_bias: bool,
}

impl DeltaProductConfig {
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
    pub fn init(&self, device: &Device) -> DeltaProduct {
        assert!(self.n_householder > 0, "n_householder must be at least 1");
        let nheads_v = self.n_value_heads_resolved();
        let value_dim = self.value_dim();

        let qkv = QkvProjectionConfig::new(
            self.d_model,
            self.nheads,
            self.head_k_dim,
            self.head_v_dim(),
        )
        .with_n_value_heads(self.n_value_heads)
        .with_n_householder(self.n_householder)
        .with_write_gate(WriteGate::Scalar)
        .with_has_gate(self.use_gate)
        .with_allow_neg_eigval(self.allow_neg_eigval)
        .with_use_short_conv(self.use_short_conv)
        .with_conv_kernel(self.conv_kernel)
        .with_conv_bias(self.conv_bias)
        .with_qk_activation(QkActivation::Silu)
        .with_qk_norm(QkNorm::L2)
        // The forget gate is per *token*, so its Δ segment is not widened by u.
        .with_extra_channels(if self.use_forget_gate { nheads_v } else { 0 })
        .with_has_proj_bias(self.has_proj_bias)
        .init(device);

        let gate = self.use_forget_gate.then(|| {
            ForgetGateConfig::new(nheads_v)
                .with_a_init_range(self.a_init_range)
                .with_dt_min(self.dt_min)
                .with_dt_max(self.dt_max)
                .with_dt_init_floor(self.dt_init_floor)
                .with_dt_limit(self.dt_limit)
                .init(device)
        });

        let bound = 1.0 / (value_dim as f64).sqrt();
        let out_proj = LinearConfig::new(value_dim, self.d_model)
            .with_bias(self.has_proj_bias)
            .with_initializer(Initializer::Uniform {
                min: -bound,
                max: bound,
            })
            .init(device);

        DeltaProduct {
            qkv,
            gate,
            norm: OutNorm::init(self.head_v_dim(), self.use_gate, device),
            out_proj,
        }
    }

    /// The block's 2-D weights Muon may own, and where their fused columns
    /// split.
    ///
    /// `k`/`v` are `u` stacked maps; each is listed separately, because a
    /// micro-step's key projection is its own matrix and orthogonalising the
    /// `u` of them jointly would tie factors that are meant to be independent.
    /// See [`burn_stack::optim`].
    #[cfg(feature = "optim")]
    pub fn muon_projections(&self) -> Vec<burn_stack::optim::ProjSpec> {
        use burn_stack::optim::{ProjSegment as Seg, ProjSpec};
        let nheads_v = self.n_value_heads_resolved();
        let mut segments = vec![Seg::muon("q", self.key_dim())];
        for _ in 0..self.n_householder {
            segments.push(Seg::muon("k", self.key_dim()));
        }
        for _ in 0..self.n_householder {
            segments.push(Seg::muon("v", self.value_dim()));
        }
        segments.push(Seg::adamw("beta", self.n_householder * nheads_v));
        if self.use_gate {
            segments.push(Seg::muon("gate", self.value_dim()));
        }
        if self.use_forget_gate {
            segments.push(Seg::adamw("dt", nheads_v));
        }
        vec![
            ProjSpec::block("qkv.in_proj.weight", segments),
            ProjSpec::block_whole("out_proj.weight", self.d_model),
        ]
    }
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
