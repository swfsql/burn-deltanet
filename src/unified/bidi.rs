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
use burn_stack::modules::{
    BidiLayers, BidiLayersBuilder, BlockConfig, OutputMergeConfig, ResidualsConfig,
};
use burn_stack::utils::{BidiSchedule, ClassLatent};

use crate::unified::block::{DeltaBlock, DeltaBlockConfig};

/// A bidirectional layer stack over any delta-rule family.
pub type DeltaBidiLayers = BidiLayers<DeltaBlock>;

/// The block-independent knobs of a bidirectional stack.
#[derive(Config, Debug)]
pub struct DeltaBidiShape {
    /// Number of real (weight-bearing) layers. Must be even — they pair up.
    pub n_real_layers: usize,
    /// One merge config per pair; length `n_real_layers / 2`.
    pub outputs_merge: Vec<OutputMergeConfig>,
    /// Optional virtual-layer scheduling over the real pairs.
    #[config(default = "None")]
    pub n_virtual_layers: Option<(usize, BidiSchedule)>,
    /// Zero the first virtual pair's residual.
    #[config(default = false)]
    pub ignore_first_residual: bool,
    /// Zero the last virtual pair's residual.
    #[config(default = false)]
    pub ignore_last_residual: bool,
    /// Stack-level class latents, spliced once before the first pair.
    #[config(default = "Vec::new()")]
    pub class_latents: Vec<ClassLatent>,
    /// Inter-pair residual scheme.
    #[config(default = "ResidualsConfig::Standard")]
    pub residuals: ResidualsConfig,
}

impl DeltaBidiShape {
    /// The builder for this shape around a given block config — the entry point
    /// for a statically named family.
    pub fn build<C: BlockConfig>(&self, block: C) -> BidiLayersBuilder<C> {
        BidiLayersBuilder {
            n_real_layers: self.n_real_layers,
            n_virtual_layers: self.n_virtual_layers.clone(),
            block,
            ignore_first_residual: self.ignore_first_residual,
            ignore_last_residual: self.ignore_last_residual,
            outputs_merge: self.outputs_merge.clone(),
            class_latents: self.class_latents.clone(),
            residuals: self.residuals.clone(),
        }
    }
}

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
        self.shape.build(self.block.clone()).init(device)
    }

    /// The [`MuonPlan`](burn_stack::optim::MuonPlan) for this stack.
    #[cfg(feature = "optim")]
    pub fn muon_plan(&self) -> burn_stack::optim::MuonPlan {
        burn_stack::optim::MuonPlan::new(self.block.muon_projections())
    }
}
