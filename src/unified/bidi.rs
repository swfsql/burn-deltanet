//! Bidirectional stacks: `burn-stack`'s [`BidiLayers`] at [`DeltaBlock`], plus
//! the serialisable config that builds one.
//!
//! A bidirectional pair runs the same block straight (→) and reversed (← via
//! `flip`), merging the two passes per pair. That is a *non-causal* reading of
//! the sequence, so there is no `step`: the reversed pass cannot be decoded
//! token by token, because its state depends on tokens not yet seen. Encoders
//! and classifiers want this; generators do not.
//!
//! As in [`network`](super::network), the family is chosen inside the block, so
//! this is an alias rather than an enum.

use burn::prelude::*;
use burn_stack::modules::{BidiLayers, BidiShape};

use crate::unified::block::{DeltaBlock, DeltaBlockConfig};

/// A bidirectional layer stack over any delta-rule family.
pub type DeltaBidiLayers = BidiLayers<DeltaBlock>;

/// The block-independent knobs of a bidirectional stack — `burn-stack`'s
/// [`BidiShape`], alongside the unidirectional
/// [`DeltaNetworkShape`](crate::unified::network::DeltaNetworkShape).
pub type DeltaBidiShape = BidiShape;

/// Config for [`DeltaBidiLayers`].
#[derive(Config, Debug)]
pub struct DeltaBidiLayersConfig {
    /// Stack-level knobs.
    pub shape: DeltaBidiShape,
    /// Block config — this is where the family is chosen.
    pub block: DeltaBlockConfig,
}

impl DeltaBidiLayersConfig {
    /// Allocate and initialise the stack on `device`.
    pub fn init(&self, device: &Device) -> DeltaBidiLayers {
        self.shape.init(self.block.clone(), device)
    }

    /// The [`MuonPlan`](burn_stack::optim::MuonPlan) for this stack.
    #[cfg(feature = "optim")]
    pub fn muon_plan(&self) -> burn_stack::optim::MuonPlan {
        self.shape.muon_plan(&self.block)
    }
}
