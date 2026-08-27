//! The scalar forget gate, parameterised exactly as Mamba-2's decay is.
//!
//! ```text
//!   Δₜ = softplus(dt_rawₜ + dt_bias)      per head, > 0
//!   A  = −exp(a_log)                       per head, < 0 by construction
//!   gₜ = Δₜ · A ≤ 0                        the log decay
//!   αₜ = exp(gₜ) ∈ (0, 1]
//! ```
//!
//! Two parameterisation choices carry their weight here:
//!
//! - Storing `log|A|` and negating makes `A < 0` unconditional. Descent can
//!   move `a_log` anywhere at all and the gate still cannot amplify the state —
//!   no sign constraint, no clamp, no way to get it wrong.
//! - `dt_bias` is set by inverting the softplus over a **log-uniform** spread
//!   of `Δ`, so heads start with timescales spread over `[dt_min, dt_max]`
//!   rather than all at one. Which head ends up fast and which slow is then
//!   learned, but the spread is there from step zero.
//!
//! Shared by [Gated DeltaNet](crate::gated_deltanet) and
//! [DeltaProduct](crate::delta_product); [DeltaNet](crate::deltanet) has no
//! gate at all, and [GDN-2](crate::gdn2) widens it onto the key channels with
//! [`ChannelForgetGate`].

use burn::module::{Module, Param};
use burn::prelude::*;
use burn_stack::modules::{sanity as san, softplus};

/// The per-head decay parameters.
#[derive(Module, Debug)]
pub struct ForgetGate {
    /// Per-head bias for `Δ`, shape `[nheads]`.
    pub dt_bias_h: Param<Tensor<1>>,
    /// Per-head `log|A|`, shape `[nheads]`. The decay rate is `A = −exp(a_log)`.
    pub a_log_h: Param<Tensor<1>>,
    /// Hard clamp on `Δ` after the softplus.
    pub dt_limit: (f64, f64),
}

impl ForgetGate {
    /// Number of heads the gate covers.
    pub fn nheads(&self) -> usize {
        let [nheads] = self.a_log_h.dims();
        nheads
    }

    /// The log decay `g = Δ·A ≤ 0` from raw projected `Δ` channels.
    ///
    /// Shapes: `[.., nheads] → [.., nheads]`; the per-head parameters broadcast
    /// over every leading axis.
    pub fn log_decay<const D: usize>(&self, dt_raw: Tensor<D>) -> Tensor<D> {
        assert_eq!(self.nheads(), dt_raw.dims()[D - 1]);
        let dt_bias: Tensor<D> = self.dt_bias_h.val().unsqueeze();
        let a_decay: Tensor<D> = (-self.a_log_h.val().exp()).unsqueeze();
        let dt = softplus(dt_raw + dt_bias).clamp(self.dt_limit.0, self.dt_limit.1);
        let g = dt * a_decay;
        san(&g);
        g
    }
}

/// Configuration for [`ForgetGate`].
#[derive(Config, Debug)]
pub struct ForgetGateConfig {
    /// Number of heads.
    pub nheads: usize,
    /// Range `[lo, hi]` for the uniform initialisation of `|A|`, stored as
    /// `a_log = log(Uniform(lo, hi))`.
    #[config(default = "(0., 16.)")]
    pub a_init_range: (f64, f64),
    /// Minimum of the initial `Δ` spread.
    #[config(default = 1e-3)]
    pub dt_min: f64,
    /// Maximum of the initial `Δ` spread.
    #[config(default = 0.1)]
    pub dt_max: f64,
    /// Floor clamped onto the sampled initial `Δ` before inverting the softplus.
    #[config(default = 1e-4)]
    pub dt_init_floor: f64,
    /// Hard clamp on `Δ` at runtime. The default only clamps at zero (the
    /// upper bound is f16's maximum).
    #[config(default = "(0., 6.5504e+4)")]
    pub dt_limit: (f64, f64),
}

impl ForgetGateConfig {
    /// Allocate the gate on `device`.
    pub fn init(&self, device: &Device) -> ForgetGate {
        assert!(
            self.a_init_range.0 >= 0.0 && self.a_init_range.0 < self.a_init_range.1,
            "a_init_range must satisfy 0 <= lo < hi",
        );
        assert!(
            self.dt_min > 0.0 && self.dt_min < self.dt_max,
            "dt_min must satisfy 0 < dt_min < dt_max",
        );

        // softplus⁻¹(y) = y + log(1 − e^{−y}), the numerically well-behaved
        // form of log(e^y − 1) for small y.
        let dt_h = Tensor::<1>::random(
            [self.nheads],
            burn::tensor::Distribution::Uniform(self.dt_min.ln(), self.dt_max.ln()),
            device,
        )
        .exp()
        .clamp(self.dt_init_floor, f64::INFINITY);
        let expm1 = |t: Tensor<1>| t.exp() - 1.0;
        let dt_bias_h = Param::from_tensor(dt_h.clone() + (-expm1(-dt_h)).log());

        let a_h = Tensor::<1>::random(
            [self.nheads],
            burn::tensor::Distribution::Uniform(self.a_init_range.0, self.a_init_range.1),
            device,
        );

        ForgetGate {
            dt_bias_h,
            a_log_h: Param::from_tensor(a_h.log()),
            dt_limit: self.dt_limit,
        }
    }
}

// ---------------------------------------------------------------------------
// ChannelForgetGate
// ---------------------------------------------------------------------------

/// The same decay, one value per **key channel** instead of one per head.
///
/// [GDN-2](crate::gdn2) (following KDA) gives every row of the state its own
/// timescale, so `Δ` must be projected at `nheads · head_k_dim` width. At the
/// deployed shape that equals `d_model`, which would make a dense `d_model →
/// key_dim` map as large as the rest of the block put together — so the
/// reference routes it through a rank-`bottleneck` bottleneck, and so does
/// this. The bottleneck's *first* factor is a segment of the block's fused
/// `in_proj` (it is a `d_model → r` map like any other); [`Self::up`] is the
/// second.
///
/// ```text
///   gₜ = lower_bound · σ(exp(a_log) · (up(extraₜ) + dt_bias))  ∈ (lower_bound, 0)
/// ```
///
/// `a_log` is **per head** — it sets how sharply that head's gate saturates —
/// while the projected `Δ` supplies the per-channel, per-token modulation.
///
/// ## Why this is bounded and [`ForgetGate`] is not
///
/// The scalar gate's `g = softplus(Δ + dt_bias)·(−exp(a_log))` is unbounded
/// below, and harmlessly so: the chunk path only ever forms `e^{Gᵢ−Gⱼ}` for
/// `i ≥ j`, which is at most 1 however fast the decay is. A per-channel gate
/// does not factor that way — [`decay`](crate::delta::decay) has to split the
/// same quantity into `e^{Gᵢ−G_ref}·e^{G_ref−Gⱼ}`, and the second factor grows
/// with the accumulated decay. Bounding `g` below bounds it. This is the
/// reference's own `safe_gate` / `lower_bound` parameterisation, with the same
/// `[-5, 0)` admissible range.
#[derive(Module, Debug)]
pub struct ChannelForgetGate {
    /// `bottleneck → nheads · head_k_dim`, the second half of the low-rank `Δ`
    /// projection.
    pub up: burn::nn::Linear,
    /// Per-channel bias for `Δ`, shape `[nheads · head_k_dim]`.
    pub dt_bias_i: Param<Tensor<1>>,
    /// Per-head sigmoid slope, shape `[nheads]`.
    pub a_log_h: Param<Tensor<1>>,
    /// Number of heads.
    pub nheads: usize,
    /// Query/key width per head.
    pub head_k_dim: usize,
    /// The most negative `g` the gate can reach, in `[-5, 0)`.
    pub lower_bound: f64,
}

impl ChannelForgetGate {
    /// The log decay `g ∈ (lower_bound, 0)` from the bottleneck's raw channels.
    ///
    /// Shapes: `[.., bottleneck] → [.., nheads, head_k_dim]`, with one leading
    /// axis (`sequence`) or none.
    pub fn log_decay<const D: usize, const DP1: usize>(&self, raw: Tensor<D>) -> Tensor<DP1> {
        let dt_bias: Tensor<D> = self.dt_bias_i.val().unsqueeze();
        let dt_i = self.up.forward(raw) + dt_bias;

        // Split per head *before* applying the slope, so the per-head parameter
        // broadcasts over the (nheads, head_k_dim) tail rather than being
        // materialised at `key_dim` width and reshaped straight back.
        let mut shape = dt_i.dims().to_vec();
        shape.pop();
        shape.extend_from_slice(&[self.nheads, self.head_k_dim]);
        let dt_hk: Tensor<DP1> = dt_i.reshape::<DP1, _>(
            <[usize; DP1]>::try_from(shape.as_slice()).expect("DP1 == D + 1"),
        );

        let slope_h1: Tensor<DP1> = self.a_log_h.val().exp().unsqueeze_dim::<2>(1).unsqueeze();
        let g = burn::tensor::activation::sigmoid(dt_hk * slope_h1) * self.lower_bound;
        san(&g);
        g
    }
}

/// Configuration for [`ChannelForgetGate`].
#[derive(Config, Debug)]
pub struct ChannelForgetGateConfig {
    /// Width of the bottleneck segment `in_proj` supplies.
    pub bottleneck: usize,
    /// Number of heads.
    pub nheads: usize,
    /// Query/key width per head.
    pub head_k_dim: usize,
    /// Range `[lo, hi]` for the uniform initialisation of the per-head sigmoid
    /// slope, stored as `a_log = log(Uniform(lo, hi))`.
    #[config(default = "(1., 16.)")]
    pub a_init_range: (f64, f64),
    /// Minimum of the initial `Δ` spread.
    #[config(default = 1e-3)]
    pub dt_min: f64,
    /// Maximum of the initial `Δ` spread.
    #[config(default = 0.1)]
    pub dt_max: f64,
    /// Floor clamped onto the sampled initial `Δ` before inverting the softplus.
    #[config(default = 1e-4)]
    pub dt_init_floor: f64,
    /// The most negative `g` the gate can reach. Must lie in `[-5, 0)`, the
    /// reference's own admissible range.
    #[config(default = -5.0)]
    pub lower_bound: f64,
}

impl ChannelForgetGateConfig {
    /// Allocate the gate on `device`.
    pub fn init(&self, device: &Device) -> ChannelForgetGate {
        assert!(
            self.a_init_range.0 > 0.0 && self.a_init_range.0 < self.a_init_range.1,
            "a_init_range must satisfy 0 < lo < hi",
        );
        assert!(
            self.dt_min > 0.0 && self.dt_min < self.dt_max,
            "dt_min must satisfy 0 < dt_min < dt_max",
        );
        assert!(
            (-5.0..0.0).contains(&self.lower_bound),
            "lower_bound must lie in [-5, 0), got {}",
            self.lower_bound,
        );
        let key_dim = self.nheads * self.head_k_dim;

        // The `Δ` spread is per channel here, not per head. Inverting the
        // softplus over a log-uniform spread starts the channels well into the
        // sigmoid's left tail, i.e. at `α ≈ 1` — no decay until it is learned.
        let dt_i = Tensor::<1>::random(
            [key_dim],
            burn::tensor::Distribution::Uniform(self.dt_min.ln(), self.dt_max.ln()),
            device,
        )
        .exp()
        .clamp(self.dt_init_floor, f64::INFINITY);
        let expm1 = |t: Tensor<1>| t.exp() - 1.0;
        let dt_bias_i = Param::from_tensor(dt_i.clone() + (-expm1(-dt_i)).log());

        let a_h = Tensor::<1>::random(
            [self.nheads],
            burn::tensor::Distribution::Uniform(self.a_init_range.0, self.a_init_range.1),
            device,
        );

        let bound = 1.0 / (self.bottleneck as f64).sqrt();
        let up = burn::nn::LinearConfig::new(self.bottleneck, key_dim)
            .with_bias(false)
            .with_initializer(burn::nn::Initializer::Uniform {
                min: -bound,
                max: bound,
            })
            .init(device);

        ChannelForgetGate {
            up,
            dt_bias_i,
            a_log_h: Param::from_tensor(a_h.log()),
            nheads: self.nheads,
            head_k_dim: self.head_k_dim,
            lower_bound: self.lower_bound,
        }
    }
}
