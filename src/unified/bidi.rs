//! Runtime-selectable bidirectional stacks: one enum over the three families'
//! [`BidiLayers`] monomorphisations.
//!
//! A bidirectional pair runs the same block straight (→) and reversed (← via
//! `flip`), merging the two passes per pair. That is a *non-causal* reading of
//! the sequence, so there is no `step`: the reversed pass cannot be decoded
//! token by token, because its state depends on tokens not yet seen. Encoders
//! and classifiers want this; generators do not.
//!
//! As in [`network`](super::network), the arms are one line each — see that
//! module's header for why.

use burn::prelude::*;
use burn_stack::modules::{BidiLayers, BidiLayersBuilder, OutputMergeConfig, ResidualsConfig};
use burn_stack::utils::{BidiSchedule, ClassCursors, ClassLatent};

use crate::delta::path::DeltaPath;
use crate::unified::cache::DeltaCaches;
use crate::unified::family::DeltaFamily;

fn bidi_forward<M: DeltaFamily>(
    layers: &BidiLayers<M>,
    x: Tensor<3>,
    caches: Option<DeltaCaches>,
    path: DeltaPath,
    class: Option<&mut ClassCursors>,
) -> (Tensor<3>, DeltaCaches) {
    let (y, caches) = layers.forward(x, caches.map(M::unwrap_caches), path, class);
    (y, M::wrap_caches(caches))
}

/// A runtime-selectable bidirectional layer stack.
#[derive(Module, Debug)]
pub enum DeltaBidiLayers {
    /// DeltaNet bidirectional stack.
    #[cfg(feature = "deltanet")]
    DeltaNet(BidiLayers<crate::deltanet::prelude::DeltaNet>),
    /// Gated DeltaNet bidirectional stack.
    #[cfg(feature = "gated-deltanet")]
    GatedDeltaNet(BidiLayers<crate::gated_deltanet::prelude::GatedDeltaNet>),
    /// DeltaProduct bidirectional stack.
    #[cfg(feature = "delta-product")]
    DeltaProduct(BidiLayers<crate::delta_product::prelude::DeltaProduct>),
    /// GDN-2 bidirectional stack.
    #[cfg(feature = "gdn2")]
    GatedDeltaNet2(BidiLayers<crate::gdn2::prelude::GatedDeltaNet2>),
}

impl DeltaBidiLayers {
    /// Full-sequence pass. The caches must belong to this stack's family; a
    /// mismatch is a caller error and panics.
    pub fn forward(
        &self,
        x: Tensor<3>,
        caches: Option<DeltaCaches>,
        path: DeltaPath,
        class: Option<&mut ClassCursors>,
    ) -> (Tensor<3>, DeltaCaches) {
        match self {
            #[cfg(feature = "deltanet")]
            Self::DeltaNet(layers) => bidi_forward(layers, x, caches, path, class),
            #[cfg(feature = "gated-deltanet")]
            Self::GatedDeltaNet(layers) => bidi_forward(layers, x, caches, path, class),
            #[cfg(feature = "delta-product")]
            Self::DeltaProduct(layers) => bidi_forward(layers, x, caches, path, class),
            #[cfg(feature = "gdn2")]
            Self::GatedDeltaNet2(layers) => bidi_forward(layers, x, caches, path, class),
        }
    }
}

/// The family-independent knobs of a bidirectional stack.
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
    fn build<C: burn_stack::modules::BlockConfig>(&self, block: C) -> BidiLayersBuilder<C> {
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
pub enum DeltaBidiLayersConfig {
    /// Build a DeltaNet bidirectional stack.
    #[cfg(feature = "deltanet")]
    DeltaNet {
        /// Stack-level knobs.
        shape: DeltaBidiShape,
        /// Block config.
        block: crate::deltanet::prelude::DeltaNetConfig,
    },
    /// Build a Gated DeltaNet bidirectional stack.
    #[cfg(feature = "gated-deltanet")]
    GatedDeltaNet {
        /// Stack-level knobs.
        shape: DeltaBidiShape,
        /// Block config.
        block: crate::gated_deltanet::prelude::GatedDeltaNetConfig,
    },
    /// Build a DeltaProduct bidirectional stack.
    #[cfg(feature = "delta-product")]
    DeltaProduct {
        /// Stack-level knobs.
        shape: DeltaBidiShape,
        /// Block config.
        block: crate::delta_product::prelude::DeltaProductConfig,
    },
    /// Build a GDN-2 bidirectional stack.
    #[cfg(feature = "gdn2")]
    GatedDeltaNet2 {
        /// Stack-level knobs.
        shape: DeltaBidiShape,
        /// Block config.
        block: crate::gdn2::prelude::GatedDeltaNet2Config,
    },
}

impl DeltaBidiLayersConfig {
    /// Allocate and initialise the selected stack on `device`.
    pub fn init(&self, device: &Device) -> DeltaBidiLayers {
        match self {
            #[cfg(feature = "deltanet")]
            Self::DeltaNet { shape, block } => {
                DeltaBidiLayers::DeltaNet(shape.build(block.clone()).init(device))
            }
            #[cfg(feature = "gated-deltanet")]
            Self::GatedDeltaNet { shape, block } => {
                DeltaBidiLayers::GatedDeltaNet(shape.build(block.clone()).init(device))
            }
            #[cfg(feature = "delta-product")]
            Self::DeltaProduct { shape, block } => {
                DeltaBidiLayers::DeltaProduct(shape.build(block.clone()).init(device))
            }
            #[cfg(feature = "gdn2")]
            Self::GatedDeltaNet2 { shape, block } => {
                DeltaBidiLayers::GatedDeltaNet2(shape.build(block.clone()).init(device))
            }
        }
    }

    /// The [`MuonPlan`](burn_stack::optim::MuonPlan) for this stack.
    #[cfg(feature = "optim")]
    pub fn muon_plan(&self) -> burn_stack::optim::MuonPlan {
        let specs = match self {
            #[cfg(feature = "deltanet")]
            Self::DeltaNet { block, .. } => block.muon_projections(),
            #[cfg(feature = "gated-deltanet")]
            Self::GatedDeltaNet { block, .. } => block.muon_projections(),
            #[cfg(feature = "delta-product")]
            Self::DeltaProduct { block, .. } => block.muon_projections(),
            #[cfg(feature = "gdn2")]
            Self::GatedDeltaNet2 { block, .. } => block.muon_projections(),
        };
        burn_stack::optim::MuonPlan::new(specs)
    }
}
