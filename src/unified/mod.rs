//! # The runtime-selectable API
//!
//! Two things live here:
//!
//! - **Where the families plug in** ([`cache`]): the
//!   [`Block`](burn_stack::modules::Block) /
//!   [`BlockConfig`](burn_stack::modules::BlockConfig) /
//!   [`CacheStack`](burn_stack::modules::CacheStack) impls that hand each block
//!   to the generic containers in [`burn_stack`]. Once those exist, every
//!   container — the Pre-LN layer, the virtual-layer stack, bidirectional
//!   pairs, latent/vocab networks, multi-gate residuals, class tokens, the Muon
//!   plan — applies to all three families unchanged.
//! - **Runtime dispatch** ([`network`], [`bidi`]): enums that pick a family at
//!   *construction* time rather than compile time, for callers that read the
//!   choice out of a config file. They wrap the generic containers; they do not
//!   reimplement them.
//!
//! ## What Muon sees
//!
//! Each family's `muon_projections()` lists the column seams of its fused
//! `in_proj` so [`burn_stack::optim`] can orthogonalise each sub-matrix on its
//! own while the forward keeps one GEMM. Two kinds of channel are deliberately
//! *excluded*:
//!
//! - **Per-head scalars** — `β` and the scalar forget gate's `Δ` project one
//!   number per head. A `[d_model, nheads]` slice is a stack of independent
//!   linear functionals, not a matrix whose singular values mean anything;
//!   Muon's orthogonalisation would mix heads that share nothing. They stay on
//!   AdamW, as do the 1-D `a_log`/`dt_bias`/`γ` and the 3-D convolution weight.
//!   [GDN-2](crate::gdn2) has no such channel: its erase, write and `Δ` maps
//!   all produce feature *vectors*, so they are Muon's.
//! - **Nothing else.** `q`, `k`, `v` and the output gate *are* matrices, and
//!   under [DeltaProduct](crate::delta_product) each of the `u` key and value
//!   maps is listed separately — orthogonalising the `u` of them jointly would
//!   couple Householder factors that are meant to be chosen independently.

pub mod cache;
pub mod family;
pub mod bidi;
pub mod network;

// The container suite runs every family through one enum, so it needs them all
// compiled in.
#[cfg(all(
    test,
    feature = "_dev-test",
    feature = "deltanet",
    feature = "gated-deltanet",
    feature = "delta-product",
    feature = "gdn2"
))]
mod tests;

pub use cache::DeltaCaches;
pub use family::DeltaFamily;
pub use bidi::{DeltaBidiLayers, DeltaBidiLayersConfig, DeltaBidiShape};
pub use network::{
    DeltaLatentNet, DeltaLatentNetConfig, DeltaLatentShape, DeltaNetworkShape, DeltaVocabNet,
    DeltaVocabNetConfig, DeltaVocabShape,
};
