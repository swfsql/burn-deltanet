//! The example model surface: just the [`ModelConfigExt`] factory trait (the
//! seam the generic training loop uses to stay model-agnostic) plus its impl for
//! the library's unified [`DeltaLatentNetConfig`] / [`DeltaVocabNetConfig`].
//!
//! The examples define no networks of their own. They build directly from the
//! library's family-generic containers (`in_proj → Layers → out_proj`, exposed
//! as the runtime-selectable [`DeltaLatentNet`]); each example just picks the
//! family variant in its `model_config()`.

use burn::prelude::*;
use burn_deltanet::prelude::{
    DeltaLatentNet, DeltaLatentNetConfig, DeltaVocabNet, DeltaVocabNetConfig,
};
use burn_stack::optim::MuonPlan;

/// A model config that can build its module on a device — the seam the generic
/// training loop uses to stay model-agnostic.
pub trait ModelConfigExt: Config {
    /// The module type this config builds.
    type Model: Module;
    /// Allocate and initialise the model on `device`.
    fn init(&self, device: &Device) -> Self::Model;
    /// Which of the model's weights Muon may own, and where the fused
    /// projections split (see `burn_stack::optim`). Consumed by
    /// [`OptimizerConfig::init`](crate::common::training::OptimizerConfig::init);
    /// irrelevant when the training config leaves `muon` unset.
    fn muon_plan(&self) -> MuonPlan;
}

impl ModelConfigExt for DeltaLatentNetConfig {
    type Model = DeltaLatentNet;
    fn muon_plan(&self) -> MuonPlan {
        self.muon_plan()
    }
    fn init(&self, device: &Device) -> Self::Model {
        // `self.init(..)` resolves to the inherent `DeltaLatentNetConfig::init`
        // (inherent methods win over trait methods in method-call syntax), so
        // this delegates to the library builder rather than recursing.
        self.init(device)
    }
}

impl ModelConfigExt for DeltaVocabNetConfig {
    type Model = DeltaVocabNet;
    fn muon_plan(&self) -> MuonPlan {
        self.muon_plan()
    }
    fn init(&self, device: &Device) -> Self::Model {
        // Same inherent-over-trait resolution as the latent impl above.
        self.init(device)
    }
}
