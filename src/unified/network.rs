//! The networks: `burn-stack`'s [`LatentNetwork`] / [`VocabNetwork`] at
//! [`DeltaBlock`], plus the serialisable configs that build them.
//!
//! There is no per-family network type here, and no dispatch: the families
//! agree on their cache, their options and their interface, so the runtime
//! choice is made once — inside the block — and every container above it is the
//! block-generic one. [`DeltaLatentNet`] and [`DeltaVocabNet`] are therefore
//! plain aliases.
//!
//! When the family is known statically, name it: [`DeltaLatentShape::build`]
//! takes any [`BlockConfig`](burn_stack::modules::BlockConfig), so
//! `shape.build(GatedDeltaNet1Config::new(..)).init(&device)` gives a
//! `LatentNetwork<GatedDeltaNet1>` with no enum in the way.
//!
//! The shapes themselves are **`burn-stack`'s** — nothing about a layer stack
//! is delta-specific, so [`DeltaNetworkShape`] & co. are aliases of
//! [`burn_stack::modules::shape`], where every knob is declared and documented
//! once. What this module adds is the family-tagged `{ shape, block }` pair.

use burn::prelude::*;
use burn_stack::modules::{LatentNetwork, LatentShape, NetworkShape, VocabNetwork, VocabShape};

use crate::unified::block::{DeltaBlock, DeltaBlockConfig};

// ===========================================================================
// The knobs every network config shares
// ===========================================================================

/// The block-independent half of a network config: everything about the *stack*
/// rather than the block — `burn-stack`'s [`NetworkShape`], which is where the
/// knobs (and the reference architecture's settings for them) are documented.
pub type DeltaNetworkShape = NetworkShape;

// ===========================================================================
// DeltaLatentNet
// ===========================================================================

/// A feature/regression network on latents over any delta-rule family:
/// `in_proj → Layers → [norm_f] → out_proj`.
pub type DeltaLatentNet = LatentNetwork<DeltaBlock>;

/// The latent network's own knobs, on top of [`DeltaNetworkShape`] —
/// `burn-stack`'s [`LatentShape`].
pub type DeltaLatentShape = LatentShape;

/// Config for [`DeltaLatentNet`]: the stack's shape, and the family-tagged
/// block config it is built around.
#[derive(Config, Debug)]
pub struct DeltaLatentNetConfig {
    /// Stack-level knobs.
    pub shape: DeltaLatentShape,
    /// Block config — this is where the family is chosen.
    pub block: DeltaBlockConfig,
}

impl DeltaLatentNetConfig {
    /// The [`MuonPlan`](burn_stack::optim::MuonPlan) for this network: the
    /// block's fused projections plus the optional MLP's.
    ///
    /// The network's own boundary weights — `in_proj`/`out_proj`, class-token
    /// tables — are deliberately left out; see [`burn_stack::optim`].
    #[cfg(feature = "optim")]
    pub fn muon_plan(&self) -> burn_stack::optim::MuonPlan {
        self.shape.stack.muon_plan(&self.block)
    }

    /// Allocate and initialise the network on `device`.
    pub fn init(&self, device: &Device) -> DeltaLatentNet {
        self.shape.init(self.block.clone(), device)
    }
}

// ===========================================================================
// DeltaVocabNet
// ===========================================================================

/// A complete autoregressive language model over any delta-rule family:
/// `Embedding → Layers → norm_f → LM head`.
pub type DeltaVocabNet = VocabNetwork<DeltaBlock>;

/// The vocab network's own knobs, on top of [`DeltaNetworkShape`] —
/// `burn-stack`'s [`VocabShape`].
pub type DeltaVocabShape = VocabShape;

/// Config for [`DeltaVocabNet`].
#[derive(Config, Debug)]
pub struct DeltaVocabNetConfig {
    /// Vocabulary + stack knobs.
    pub shape: DeltaVocabShape,
    /// Block config — this is where the family is chosen.
    pub block: DeltaBlockConfig,
}

impl DeltaVocabNetConfig {
    /// The [`MuonPlan`](burn_stack::optim::MuonPlan) for this model. The
    /// embedding and LM head stay on the fallback optimizer.
    #[cfg(feature = "optim")]
    pub fn muon_plan(&self) -> burn_stack::optim::MuonPlan {
        self.shape.stack.muon_plan(&self.block)
    }

    /// Allocate and initialise the model on `device`.
    pub fn init(&self, device: &Device) -> DeltaVocabNet {
        self.shape.init(self.block.clone(), device)
    }
}

// ===========================================================================
// The `config → module` seam
// ===========================================================================

/// The [`ModelConfigExt`] impls: what a model-agnostic driver (an example's
/// training loop, artifact loading) needs from a whole-model config, namely how
/// to build it on a device and which weights Muon may own.
///
/// Both methods forward to the inherent ones above — `self.init(..)` resolves to
/// [`DeltaLatentNetConfig::init`] (inherent methods win over trait methods in
/// method-call syntax), so this delegates rather than recurses.
mod model_config_ext {
    use super::*;
    use burn_stack::modules::ModelConfigExt;

    impl ModelConfigExt for DeltaLatentNetConfig {
        type Model = DeltaLatentNet;
        fn init(&self, device: &Device) -> Self::Model {
            self.init(device)
        }
        #[cfg(feature = "optim")]
        fn muon_plan(&self) -> burn_stack::optim::MuonPlan {
            self.muon_plan()
        }
    }

    impl ModelConfigExt for DeltaVocabNetConfig {
        type Model = DeltaVocabNet;
        fn init(&self, device: &Device) -> Self::Model {
            self.init(device)
        }
        #[cfg(feature = "optim")]
        fn muon_plan(&self) -> burn_stack::optim::MuonPlan {
            self.muon_plan()
        }
    }
}
