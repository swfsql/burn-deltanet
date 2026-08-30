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
//! takes any [`BlockConfig`], so
//! `shape.build(GatedDeltaNet1Config::new(..)).init(&device)` gives a
//! `LatentNetwork<GatedDeltaNet1>` with no enum in the way.

use burn::prelude::*;
use burn_stack::modules::{
    BlockConfig, GatedMlpConfig, LatentNetwork, LatentNetworkBuilder, LayersBuilder,
    ResidualsConfig, VocabNetwork, VocabNetworkBuilder,
};
use burn_stack::utils::{ClassLatent, ClassToken, GradHorizon, InitPolicy, Schedule};

use crate::unified::block::{DeltaBlock, DeltaBlockConfig};

// ===========================================================================
// The knobs every network config shares
// ===========================================================================

/// The block-independent half of a network config: everything about the *stack*
/// rather than the block.
///
/// Split out because it is identical for both networks — a
/// [`DeltaLatentShape`] / [`DeltaVocabShape`] then adds only its own I/O
/// boundary.
#[derive(Config, Debug)]
pub struct DeltaNetworkShape {
    /// Number of real weight sets.
    pub n_real_layers: usize,

    /// Optional virtual-layer scheduling: run `n` logical layers over the real
    /// weight sets, mapped by a [`Schedule`].
    #[config(default = "None")]
    pub n_virtual_layers: Option<(usize, Schedule)>,

    /// Which virtual layers back-propagate; everything else runs on the inner
    /// backend (truncated BPTT for deep recursion). `None` ⇒ track the whole
    /// stack. See
    /// [`Layers::grad_horizon`](burn_stack::modules::Layers::grad_horizon).
    #[config(default = "None")]
    pub grad_horizon: Option<GradHorizon>,

    /// Stack-level class latents, spliced into the sequence before the first
    /// layer (width `d_model`).
    #[config(default = "Vec::new()")]
    pub class_latents: Vec<ClassLatent>,

    /// Suppress the first virtual layer's residual.
    #[config(default = false)]
    pub ignore_first_residual: bool,

    /// Suppress the last virtual layer's residual (the output is then the last
    /// layer's transform alone).
    #[config(default = false)]
    pub ignore_last_residual: bool,

    /// Inter-layer residual scheme (plain additive vs Multi-Gate).
    #[config(default = "ResidualsConfig::Standard")]
    pub residuals: ResidualsConfig,

    /// Optional per-layer SwiGLU feed-forward sub-block, with its own pre-norm
    /// and inner residual. `None` ⇒ mixer-only layers.
    ///
    /// Every reference delta-rule language model has one: the architecture is
    /// Llama's macro design with the delta rule in place of self-attention, so
    /// a token mixer is followed by a SwiGLU MLP of
    /// [`GatedMlpConfig::from_hidden_ratio`] width.
    #[config(default = "None")]
    pub mlp: Option<GatedMlpConfig>,

    /// Optional post-build re-initialisation of the whole network (the
    /// reference `initializer_range` + residual rescale). `None` ⇒ keep Burn's
    /// per-module defaults. See [`InitPolicy`].
    #[config(default = "None")]
    pub init: Option<InitPolicy>,
}

impl DeltaNetworkShape {
    /// The number of residual sub-blocks per layer this stack has, which is
    /// what an [`InitPolicy`] rescale is counted over: the mixer, plus the
    /// feed-forward when there is one.
    fn residuals_per_layer(&self) -> usize {
        if self.mlp.is_some() { 2 } else { 1 }
    }

    /// The [`InitPolicy`] to apply after building, with its rescale resolved
    /// against this stack's depth.
    fn init_policy(&self) -> Option<InitPolicy> {
        let n_virtual = self
            .n_virtual_layers
            .as_ref()
            .map(|(l, _)| *l)
            .unwrap_or(self.n_real_layers);
        self.init
            .clone()
            .map(|init| init.with_default_residual_depth(self.residuals_per_layer() * n_virtual))
    }

    /// Build the layer-stack builder for a given block config.
    pub fn layers<C: BlockConfig>(&self, block: C) -> LayersBuilder<C> {
        LayersBuilder::new(self.n_real_layers, block)
            .with_n_virtual_layers(self.n_virtual_layers.clone())
            .with_grad_horizon(self.grad_horizon.clone())
            .with_residuals(self.residuals.clone())
            .with_ignore_first_residual(self.ignore_first_residual)
            .with_ignore_last_residual(self.ignore_last_residual)
            .with_class_latents(self.class_latents.clone())
            .with_mlp(self.mlp.clone())
    }
}

// ===========================================================================
// DeltaLatentNet
// ===========================================================================

/// A feature/regression network on latents over any delta-rule family:
/// `in_proj → Layers → [norm_f] → out_proj`.
pub type DeltaLatentNet = LatentNetwork<DeltaBlock>;

/// The latent network's own knobs, on top of [`DeltaNetworkShape`].
#[derive(Config, Debug)]
pub struct DeltaLatentShape {
    /// Input feature width, fed to `in_proj`.
    pub input_size: usize,
    /// Output feature width, produced by `out_proj`.
    pub output_size: usize,
    /// The stack's knobs.
    pub stack: DeltaNetworkShape,
    /// Insert a final RMSNorm before `out_proj`.
    #[config(default = false)]
    pub final_norm: bool,
    /// Network-level class tokens, spliced into the input before `in_proj`
    /// (width `input_size`, unlike the stack's class latents).
    #[config(default = "Vec::new()")]
    pub class_tokens: Vec<ClassToken>,
}

impl DeltaLatentShape {
    /// The builder for this shape around a given block config — the entry point
    /// for a statically named family.
    pub fn build<C: BlockConfig>(&self, block: C) -> LatentNetworkBuilder<C> {
        LatentNetworkBuilder {
            input_size: self.input_size,
            layers: self.stack.layers(block),
            output_size: self.output_size,
            final_norm: self.final_norm,
            class_tokens: self.class_tokens.clone(),
        }
    }
}

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
        burn_stack::optim::MuonPlan::new(self.block.muon_projections())
            .with_mlp(self.shape.stack.mlp.as_ref())
    }

    /// Allocate and initialise the network on `device`.
    pub fn init(&self, device: &Device) -> DeltaLatentNet {
        let net = self.shape.build(self.block.clone()).init(device);
        match self.shape.stack.init_policy() {
            Some(init) => init.apply(net),
            None => net,
        }
    }
}

// ===========================================================================
// DeltaVocabNet
// ===========================================================================

/// A complete autoregressive language model over any delta-rule family:
/// `Embedding → Layers → norm_f → LM head`.
pub type DeltaVocabNet = VocabNetwork<DeltaBlock>;

/// The vocab network's own knobs, on top of [`DeltaNetworkShape`].
#[derive(Config, Debug)]
pub struct DeltaVocabShape {
    /// Unpadded vocabulary size.
    pub vocab_size: usize,
    /// The stack's knobs.
    pub stack: DeltaNetworkShape,
    /// Round `vocab_size` up to a multiple of this (1 disables rounding).
    #[config(default = 1)]
    pub pad_vocab_size_multiple: usize,
    /// Tie the LM head to the (transposed) embedding weights.
    #[config(default = true)]
    pub missing_lm_head: bool,
}

impl DeltaVocabShape {
    /// The builder for this shape around a given block config — the entry point
    /// for a statically named family.
    pub fn build<C: BlockConfig>(&self, block: C) -> VocabNetworkBuilder<C> {
        VocabNetworkBuilder {
            vocab_size: self.vocab_size,
            pad_vocab_size_multiple: self.pad_vocab_size_multiple,
            layers: self.stack.layers(block),
            missing_lm_head: self.missing_lm_head,
        }
    }
}

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
        burn_stack::optim::MuonPlan::new(self.block.muon_projections())
            .with_mlp(self.shape.stack.mlp.as_ref())
    }

    /// Allocate and initialise the model on `device`.
    pub fn init(&self, device: &Device) -> DeltaVocabNet {
        let net = self.shape.build(self.block.clone()).init(device);
        match self.shape.stack.init_policy() {
            Some(init) => init.apply(net),
            None => net,
        }
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
