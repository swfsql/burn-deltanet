//! The projection front-end every family in this crate shares.
//!
//! A delta-rule block is "produce `(q, k, v, β)` from a token stream, run the
//! [delta rule](crate::delta), norm and project back". Everything before the
//! recurrence is identical across the three families — so it lives here once,
//! and each family's own file is left saying only what actually differs (the
//! forget gate, the Householder count).
//!
//! ```text
//!   x → in_proj → [ q | k | v | β? | gate? | extra? ]
//!                   └──── short conv ────┘
//!                              ↓ activation, head split, QK-norm, σ(β)
//!                        q, k, v, β  (+ gate, extra)
//! ```
//!
//! The projection is **fused**: one `Linear` produces every segment, as
//! Mamba-2 does, rather than the reference's five separate `nn.Linear`s. The
//! maps stay independent — [`crate::unified`]'s Muon plan splits the weight
//! back at exactly these column seams — but the forward is one GEMM.
//!
//! ## The `n_householder` axis
//!
//! [DeltaProduct](crate::delta_product) applies `u = n_householder` delta-rule
//! micro-steps per token, so it projects `u` copies of `k`, `v` and `β` (but
//! one `q`: the readout happens once, after all `u` writes). Rather than carry
//! a `u` axis everywhere, the micro-steps are **folded into the sequence**:
//! `k`, `v`, `β` come back with length `sequence · u`, in micro-step order.
//! For `u = 1` — the other two families — that is the plain sequence and there
//! is no special case anywhere.

use burn::module::Module;
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::prelude::*;
use burn_stack::modules::{Silu, sanity as san};

use super::conv::{ConvActivation, ShortConv, ShortConvConfig};
use super::norm::{QkActivation, QkNorm};

/// The projected, activated, normalised inputs to the delta rule, for a full
/// sequence.
///
/// `k`/`v`/`β` carry the micro-step axis folded into the sequence (`S = s · u`,
/// which is just `s` for `n_householder = 1`).
#[allow(non_snake_case)]
pub struct Qkv {
    /// Queries, one per token. `[batch, sequence, nheads, head_k_dim]`
    pub q_bshk: Tensor<4>,
    /// Keys. `[batch, sequence·n_householder, nheads, head_k_dim]`
    pub k_bShk: Tensor<4>,
    /// Values. `[batch, sequence·n_householder, nheads, head_v_dim]`
    pub v_bShv: Tensor<4>,
    /// Write strengths. `[batch, sequence·n_householder, nheads]`
    pub beta_bSh: Tensor<3>,
    /// Output gate, when the block has one. `[batch, sequence, nheads, head_v_dim]`
    pub gate_bshv: Option<Tensor<4>>,
    /// Raw trailing channels the family asked for (Gated DeltaNet's forget-gate
    /// projection). `[batch, sequence, extra_channels]`
    pub extra_bsx: Option<Tensor<3>>,
}

/// [`Qkv`] for a single token: the sequence axis is gone, but the micro-step
/// axis remains explicit.
pub struct QkvStep {
    /// Query. `[batch, nheads, head_k_dim]`
    pub q_bhk: Tensor<3>,
    /// Keys. `[batch, n_householder, nheads, head_k_dim]`
    pub k_buhk: Tensor<4>,
    /// Values. `[batch, n_householder, nheads, head_v_dim]`
    pub v_buhv: Tensor<4>,
    /// Write strengths. `[batch, n_householder, nheads]`
    pub beta_buh: Tensor<3>,
    /// Output gate, when the block has one. `[batch, nheads, head_v_dim]`
    pub gate_bhv: Option<Tensor<3>>,
    /// Raw trailing channels. `[batch, extra_channels]`
    pub extra_bx: Option<Tensor<2>>,
}

/// The fused in-projection + short convolution + activation + QK-norm.
#[derive(Module, Debug)]
pub struct QkvProjection {
    /// `d_model → [q | k | v | β? | gate? | extra?]`.
    pub in_proj: Linear,
    /// Causal short convolution over the `[q | k | v]` channels. Absent when
    /// the block runs without one — which the reference warns against, since
    /// the convolution is what supplies the local context the recurrence is
    /// deliberately bad at.
    pub conv: Option<ShortConv>,
    /// Activation for `q`/`k`. A non-parameter constant.
    #[module(skip)]
    pub qk_activation: QkActivation,
    /// Normalisation for `q`/`k`. A non-parameter constant.
    #[module(skip)]
    pub qk_norm: QkNorm,
    /// Number of query/key heads — how many are *projected*.
    pub nheads: usize,
    /// Number of value heads; a multiple of [`Self::nheads`]. Equal to it
    /// unless the block uses grouped values, in which case `q`/`k` are
    /// replicated up to this count.
    pub n_value_heads: usize,
    /// Query/key width per head.
    pub head_k_dim: usize,
    /// Value width per head.
    pub head_v_dim: usize,
    /// Delta-rule micro-steps per token (1 for everything but DeltaProduct).
    pub n_householder: usize,
    /// Whether `β` is projected (otherwise `β ≡ 1`: a pure projection write).
    pub has_beta: bool,
    /// Whether an output gate is projected.
    pub has_gate: bool,
    /// `β ∈ (0, 2)` rather than `(0, 1)`, admitting negative eigenvalues.
    pub allow_neg_eigval: bool,
    /// Width of the trailing family-owned segment.
    pub extra_channels: usize,
}

impl QkvProjection {
    /// `nheads · head_k_dim`.
    pub fn key_dim(&self) -> usize {
        self.nheads * self.head_k_dim
    }

    /// `n_value_heads · head_v_dim`.
    pub fn value_dim(&self) -> usize {
        self.n_value_heads * self.head_v_dim
    }

    /// Value heads per query/key head (1 without grouped values).
    pub fn heads_per_group(&self) -> usize {
        self.n_value_heads / self.nheads
    }

    /// The head count everything downstream of this projection sees: the delta
    /// rule's state, the caches, the output norm. Equals
    /// [`Self::n_value_heads`], since `q`/`k` are replicated up to it here.
    pub fn state_heads(&self) -> usize {
        self.n_value_heads
    }

    /// Channels entering the short convolution: one `q`, then `n_householder`
    /// copies each of `k` and `v`.
    pub fn conv_dim(&self) -> usize {
        self.key_dim() + self.n_householder * (self.key_dim() + self.value_dim())
    }

    /// The `in_proj` output width.
    pub fn d_in_proj(&self) -> usize {
        self.conv_dim()
            + if self.has_beta {
                self.n_householder * self.n_value_heads
            } else {
                0
            }
            + if self.has_gate { self.value_dim() } else { 0 }
            + self.extra_channels
    }

    /// The convolution window length, or 0 when there is no convolution.
    pub fn conv_kernel(&self) -> usize {
        self.conv.as_ref().map_or(0, |c| c.conv_kernel())
    }

    /// A zero convolution window, or `None` when there is no convolution.
    pub fn zero_conv_window(&self, batch: usize, device: &Device) -> Option<Tensor<3>> {
        self.conv.as_ref().map(|c| c.zero_window(batch, device))
    }

    /// Whether the convolution already applied the `q`/`k`/`v` activation.
    ///
    /// A depthwise convolution's fused SiLU covers all three at once, which is
    /// the default (`qk_activation = Silu`, matching the reference's SiLU on
    /// every one of its three convolutions). Any other `q`/`k` activation
    /// forces the split form: the convolution stays linear and `q`/`k` and `v`
    /// are activated separately afterwards. Identical values either way.
    fn activation_fused_into_conv(&self) -> bool {
        self.qk_activation == QkActivation::Silu
            && self
                .conv
                .as_ref()
                .is_some_and(|c| c.activation == ConvActivation::Silu)
    }

    /// The column widths of `in_proj`, in order, as `(name, width)`.
    ///
    /// The single source of truth for both the forward's split and the Muon
    /// plan's seams. `q`, `k` and `v` are listed separately even though the
    /// convolution treats them as one block: they are independent maps that
    /// merely share an allocation.
    pub fn segments(&self) -> Vec<(&'static str, usize)> {
        let mut segments = vec![
            ("q", self.key_dim()),
            ("k", self.n_householder * self.key_dim()),
            ("v", self.n_householder * self.value_dim()),
        ];
        if self.has_beta {
            segments.push(("beta", self.n_householder * self.n_value_heads));
        }
        if self.has_gate {
            segments.push(("gate", self.value_dim()));
        }
        if self.extra_channels > 0 {
            segments.push(("extra", self.extra_channels));
        }
        segments
    }

    /// Split `[q|k|v]`, `β`, `gate` and `extra` out of a projection output.
    ///
    /// Zero-width segments are never requested (Burn drops a zero-length split
    /// part), so the optional pieces are appended only when present.
    fn split_projection<const D: usize>(
        &self,
        projected: Tensor<D>,
        dim: usize,
    ) -> (Tensor<D>, Option<Tensor<D>>, Option<Tensor<D>>, Option<Tensor<D>>) {
        let mut sizes = vec![self.conv_dim()];
        if self.has_beta {
            sizes.push(self.n_householder * self.n_value_heads);
        }
        if self.has_gate {
            sizes.push(self.value_dim());
        }
        if self.extra_channels > 0 {
            sizes.push(self.extra_channels);
        }
        let mut parts = projected.split_with_sizes(sizes, dim).into_iter();
        let qkv = parts.next().expect("qkv segment");
        let beta = self.has_beta.then(|| parts.next().expect("beta segment"));
        let gate = self.has_gate.then(|| parts.next().expect("gate segment"));
        let extra = (self.extra_channels > 0).then(|| parts.next().expect("extra segment"));
        (qkv, beta, gate, extra)
    }

    /// Replicate `q`/`k` across the value heads sharing each query/key head.
    ///
    /// The head axis is the second-to-last; `DP1` must be `D + 1`. A no-op
    /// (not merely cheap — the same tensor) without grouped values.
    fn expand_to_value_heads<const D: usize, const DP1: usize>(&self, t: Tensor<D>) -> Tensor<D> {
        if self.heads_per_group() == 1 {
            return t;
        }
        burn_stack::modules::gqa_expand_to_heads::<D, DP1>(t, D - 2, self.n_value_heads)
    }

    /// `σ(β)`, doubled when negative eigenvalues are allowed, or `1` when the
    /// block projects no `β`.
    ///
    /// `β = 2` makes `I − β k kᵀ` an exact reflection, so the doubling is what
    /// turns the transition from "forget along `k`" into "negate along `k`".
    fn squash_beta<const D: usize>(&self, raw: Option<Tensor<D>>, ones_like: Tensor<D>) -> Tensor<D> {
        let beta = match raw {
            Some(raw) => burn::tensor::activation::sigmoid(raw),
            None => ones_like.ones_like(),
        };
        if self.allow_neg_eigval { beta * 2.0 } else { beta }
    }

    /// Full-sequence projection.
    ///
    /// Takes the convolution's rolling window (`None` iff the block has no
    /// convolution) and returns the updated one alongside the projections.
    #[allow(non_snake_case)]
    pub fn forward(
        &self,
        x_bsd: Tensor<3>,
        conv_window_bwc: Option<Tensor<3>>,
    ) -> (Qkv, Option<Tensor<3>>) {
        let [batch, sequence, _d_model] = x_bsd.dims();
        let (nheads, head_k_dim, head_v_dim) = (self.nheads, self.head_k_dim, self.head_v_dim);
        let u = self.n_householder;
        san(&x_bsd);

        let projected = self.in_proj.forward(x_bsd);
        assert_eq!([batch, sequence, self.d_in_proj()], projected.dims());
        let (qkv_bsw, beta_raw, gate_raw, extra_bsx) = self.split_projection(projected, 2);

        // ── Short convolution over the [q | k | v] channels ──────────────────
        let (qkv_bsw, next_window) = match (&self.conv, conv_window_bwc) {
            (Some(conv), Some(window)) => {
                let (y, next) = conv.forward(qkv_bsw, window);
                (y, Some(next))
            }
            (None, None) => (qkv_bsw, None),
            _ => panic!("convolution window presence must match the block's convolution"),
        };

        // ── Split into q, k, v and fold the micro-step axis into the sequence ─
        let [q_bsI, k_bsUI, v_bsUJ] = burn_stack::modules::split_into(
            qkv_bsw,
            [self.key_dim(), u * self.key_dim(), u * self.value_dim()],
            2,
        );
        let nheads_v = self.n_value_heads;
        let q_bshk = self.expand_to_value_heads::<4, 5>(
            q_bsI.reshape([batch, sequence, nheads, head_k_dim]),
        );
        let k_bShk = self.expand_to_value_heads::<4, 5>(
            k_bsUI.reshape([batch, sequence * u, nheads, head_k_dim]),
        );
        let v_bShv = v_bsUJ.reshape([batch, sequence * u, nheads_v, head_v_dim]);

        // ── Activation (unless the convolution already applied it) ───────────
        let (q_bshk, k_bShk, v_bShv) = if self.activation_fused_into_conv() {
            (q_bshk, k_bShk, v_bShv)
        } else {
            (
                self.qk_activation.apply(q_bshk),
                self.qk_activation.apply(k_bShk),
                Silu::new().forward(v_bShv),
            )
        };

        // ── QK normalisation ─────────────────────────────────────────────────
        let q_bshk = self.qk_norm.apply(q_bshk);
        let k_bShk = self.qk_norm.apply(k_bShk);

        let beta_bSh = self.squash_beta(
            beta_raw.map(|b| b.reshape([batch, sequence * u, nheads_v])),
            Tensor::zeros(Shape::new([batch, sequence * u, nheads_v]), &q_bshk.device()),
        );
        let gate_bshv = gate_raw.map(|g| g.reshape([batch, sequence, nheads_v, head_v_dim]));

        san(&q_bshk);
        san(&k_bShk);
        san(&v_bShv);
        san(&beta_bSh);

        (
            Qkv {
                q_bshk,
                k_bShk,
                v_bShv,
                beta_bSh,
                gate_bshv,
                extra_bsx,
            },
            next_window,
        )
    }

    /// Single-token projection. Mirrors [`Self::forward`] exactly, with the
    /// convolution evaluated against its window instead of over a sequence.
    #[allow(non_snake_case)]
    pub fn step(
        &self,
        x_bd: Tensor<2>,
        conv_window_bwc: Option<Tensor<3>>,
    ) -> (QkvStep, Option<Tensor<3>>) {
        let [batch, _d_model] = x_bd.dims();
        let (nheads, head_k_dim, head_v_dim) = (self.nheads, self.head_k_dim, self.head_v_dim);
        let u = self.n_householder;

        let projected = self.in_proj.forward(x_bd);
        assert_eq!([batch, self.d_in_proj()], projected.dims());
        let (qkv_bw, beta_raw, gate_raw, extra_bx) = self.split_projection(projected, 1);

        let (qkv_bw, next_window) = match (&self.conv, conv_window_bwc) {
            (Some(conv), Some(window)) => {
                let (y, next) = conv.step(qkv_bw, window);
                (y, Some(next))
            }
            (None, None) => (qkv_bw, None),
            _ => panic!("convolution window presence must match the block's convolution"),
        };

        let [q_bI, k_bUI, v_bUJ] = burn_stack::modules::split_into(
            qkv_bw,
            [self.key_dim(), u * self.key_dim(), u * self.value_dim()],
            1,
        );
        let nheads_v = self.n_value_heads;
        let q_bhk = self
            .expand_to_value_heads::<3, 4>(q_bI.reshape([batch, nheads, head_k_dim]));
        let k_buhk = self
            .expand_to_value_heads::<4, 5>(k_bUI.reshape([batch, u, nheads, head_k_dim]));
        let v_buhv = v_bUJ.reshape([batch, u, nheads_v, head_v_dim]);

        let (q_bhk, k_buhk, v_buhv) = if self.activation_fused_into_conv() {
            (q_bhk, k_buhk, v_buhv)
        } else {
            (
                self.qk_activation.apply(q_bhk),
                self.qk_activation.apply(k_buhk),
                Silu::new().forward(v_buhv),
            )
        };

        let q_bhk = self.qk_norm.apply(q_bhk);
        let k_buhk = self.qk_norm.apply(k_buhk);

        let beta_buh = self.squash_beta(
            beta_raw.map(|b| b.reshape([batch, u, nheads_v])),
            Tensor::zeros(Shape::new([batch, u, nheads_v]), &q_bhk.device()),
        );
        let gate_bhv = gate_raw.map(|g| g.reshape([batch, nheads_v, head_v_dim]));

        (
            QkvStep {
                q_bhk,
                k_buhk,
                v_buhv,
                beta_buh,
                gate_bhv,
                extra_bx,
            },
            next_window,
        )
    }
}

/// Configuration for [`QkvProjection`].
#[derive(Config, Debug)]
pub struct QkvProjectionConfig {
    /// Model width.
    pub d_model: usize,
    /// Number of query/key heads.
    pub nheads: usize,
    /// Query/key width per head.
    pub head_k_dim: usize,
    /// Value width per head.
    pub head_v_dim: usize,
    /// Number of value heads; `0` means "same as `nheads`". Must otherwise be
    /// a multiple of `nheads` (grouped values).
    #[config(default = 0)]
    pub n_value_heads: usize,
    /// Delta-rule micro-steps per token.
    #[config(default = 1)]
    pub n_householder: usize,
    /// Project `β` (rather than fixing `β ≡ 1`).
    #[config(default = true)]
    pub has_beta: bool,
    /// Project an output gate for the block's gated RMSNorm.
    #[config(default = false)]
    pub has_gate: bool,
    /// Let `β` reach `(0, 2)`.
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
    /// Activation for `q`/`k`.
    #[config(default = "QkActivation::Silu")]
    pub qk_activation: QkActivation,
    /// Normalisation for `q`/`k`.
    #[config(default = "QkNorm::L2")]
    pub qk_norm: QkNorm,
    /// Trailing family-owned channels (Gated DeltaNet's forget-gate projection).
    #[config(default = 0)]
    pub extra_channels: usize,
    /// Whether `in_proj` carries a bias.
    #[config(default = false)]
    pub has_proj_bias: bool,
}

impl QkvProjectionConfig {
    /// Allocate the projection on `device`.
    pub fn init(&self, device: &Device) -> QkvProjection {
        assert!(self.nheads > 0, "nheads must be at least 1");
        assert!(self.head_k_dim > 0 && self.head_v_dim > 0);
        assert!(self.n_householder > 0, "n_householder must be at least 1");
        assert!(
            self.n_value_heads == 0 || self.n_value_heads % self.nheads == 0,
            "n_value_heads ({}) must be a multiple of nheads ({})",
            self.n_value_heads,
            self.nheads,
        );
        assert!(
            self.qk_norm != QkNorm::Sum || self.qk_activation != QkActivation::Silu,
            "QkNorm::Sum needs a strictly-positive activation (Relu / EluPlusOne)",
        );

        // A skeleton carrying every derived width, so `d_in_proj`/`conv_dim`
        // are computed in exactly one place.
        let shape = QkvProjection {
            in_proj: LinearConfig::new(1, 1).init(device),
            conv: None,
            qk_activation: self.qk_activation,
            qk_norm: self.qk_norm,
            nheads: self.nheads,
            n_value_heads: if self.n_value_heads == 0 {
                self.nheads
            } else {
                self.n_value_heads
            },
            head_k_dim: self.head_k_dim,
            head_v_dim: self.head_v_dim,
            n_householder: self.n_householder,
            has_beta: self.has_beta,
            has_gate: self.has_gate,
            allow_neg_eigval: self.allow_neg_eigval,
            extra_channels: self.extra_channels,
        };

        // PyTorch's default `Linear` initialiser: U(-1/√fan_in, 1/√fan_in).
        let bound = 1.0 / (self.d_model as f64).sqrt();
        let in_proj = LinearConfig::new(self.d_model, shape.d_in_proj())
            .with_bias(self.has_proj_bias)
            .with_initializer(Initializer::Uniform {
                min: -bound,
                max: bound,
            })
            .init(device);

        let conv = self.use_short_conv.then(|| {
            // SiLU is folded in only when it is what `q`/`k` want too; see
            // `activation_fused_into_conv`.
            let activation = if self.qk_activation == QkActivation::Silu {
                ConvActivation::Silu
            } else {
                ConvActivation::Identity
            };
            ShortConvConfig::new(shape.conv_dim())
                .with_conv_kernel(self.conv_kernel)
                .with_has_bias(self.conv_bias)
                .with_activation(activation)
                .init(device)
        });

        QkvProjection {
            in_proj,
            conv,
            ..shape
        }
    }
}
