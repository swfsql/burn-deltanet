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
//! gate at all.

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
