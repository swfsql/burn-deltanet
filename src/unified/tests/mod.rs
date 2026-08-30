//! The `burn-stack` containers, exercised through real delta-rule blocks.
//!
//! `burn-stack` has its own `RefBlock` suite proving the containers need
//! nothing family-specific. What these tests add is the other direction: that
//! *this crate's* blocks hold up the end of the bargain the containers assume —
//! above all that `forward` over a sequence equals `step` unrolled, still true
//! once a stack, a residual, an MLP and a class token are wrapped around them.

mod bidi;
mod layers;
mod network;
#[cfg(feature = "optim")]
mod optim;

use burn::prelude::*;
use burn::tensor::Distribution;

pub(crate) type Device = burn::prelude::Device;

/// A deliberately tiny Gated DeltaNet: the tests are about the containers, not
/// about capacity.
pub(crate) fn tiny_block(d_model: usize) -> crate::gated_deltanet_1::prelude::GatedDeltaNet1Config {
    crate::gated_deltanet_1::prelude::GatedDeltaNet1Config::new(d_model)
        .with_nheads(2)
        .with_head_k_dim(4)
        .with_expand_v(1.0)
}

pub(crate) fn random_input(
    batch: usize,
    sequence: usize,
    d_model: usize,
    device: &Device,
) -> Tensor<3> {
    Tensor::random(
        [batch, sequence, d_model],
        Distribution::Normal(0.0, 1.0),
        device,
    )
}
