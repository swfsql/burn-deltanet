//! Runtime-selectable networks: one enum (plus a serialisable `Config`) over
//! the three families' [`LatentNetwork`] /
//! [`VocabNetwork`] monomorphisations.
//!
//! Use these when the family comes out of a config file rather than a `use`
//! statement. When it does not, name the generic container directly —
//! `LatentNetwork<GatedDeltaNet>` — and skip the dispatch entirely; these enums
//! wrap those containers, they do not reimplement them.
//!
//! ## Why the arms are one line each
//!
//! Every family here takes the same per-call options ([`DeltaPath`]) and the
//! same *shape* of state, so the only per-family work is tagging the caches.
//! That lives in [`DeltaFamily`], which lets the real bodies below be generic
//! functions and each match arm a single call. The configs likewise share one
//! [`DeltaNetworkShape`] of common knobs, so a variant adds only its block
//! config.

use burn::prelude::*;
use burn_stack::modules::{
    GatedMlpConfig, LatentNetwork, LatentNetworkBuilder, LayersBuilder, ResidualsConfig,
    VocabNetwork, VocabNetworkBuilder,
};
use burn_stack::utils::{ClassCursors, ClassLatent, ClassToken, Schedule};

use crate::delta::path::DeltaPath;
use crate::unified::cache::DeltaCaches;
use crate::unified::family::DeltaFamily;

// ===========================================================================
// The generic bodies every arm delegates to
// ===========================================================================

fn latent_forward<M: DeltaFamily>(
    net: &LatentNetwork<M>,
    x: Tensor<3>,
    caches: Option<DeltaCaches>,
    path: DeltaPath,
    class: Option<&mut ClassCursors>,
) -> (Tensor<3>, DeltaCaches) {
    let (y, caches) = net.forward(x, caches.map(M::unwrap_caches), path, class);
    (y, M::wrap_caches(caches))
}

fn latent_step<M: DeltaFamily>(
    net: &LatentNetwork<M>,
    x: Tensor<2>,
    caches: Option<DeltaCaches>,
    class: Option<&mut ClassCursors>,
) -> (Tensor<2>, DeltaCaches) {
    let (y, caches) = net.step(x, caches.map(M::unwrap_caches), class);
    (y, M::wrap_caches(caches))
}

fn latent_prime<M: DeltaFamily>(
    net: &LatentNetwork<M>,
    batch: usize,
    caches: Option<DeltaCaches>,
    class: Option<&mut ClassCursors>,
) -> (Option<Tensor<2>>, Option<DeltaCaches>) {
    let (y, caches) = net.prime(batch, caches.map(M::unwrap_caches), class);
    (y, caches.map(M::wrap_caches))
}

fn vocab_forward<M: DeltaFamily>(
    net: &VocabNetwork<M>,
    x: Tensor<2, Int>,
    caches: Option<DeltaCaches>,
    path: DeltaPath,
    class: Option<&mut ClassCursors>,
) -> (Tensor<3>, DeltaCaches) {
    let (y, caches) = net.forward(x, caches.map(M::unwrap_caches), path, class);
    (y, M::wrap_caches(caches))
}

fn vocab_step<M: DeltaFamily>(
    net: &VocabNetwork<M>,
    x: Tensor<1, Int>,
    caches: Option<DeltaCaches>,
    class: Option<&mut ClassCursors>,
) -> (Tensor<2>, DeltaCaches) {
    let (y, caches) = net.step(x, caches.map(M::unwrap_caches), class);
    (y, M::wrap_caches(caches))
}

fn vocab_prime<M: DeltaFamily>(
    net: &VocabNetwork<M>,
    batch: usize,
    caches: Option<DeltaCaches>,
    class: Option<&mut ClassCursors>,
) -> (Option<Tensor<2>>, Option<DeltaCaches>) {
    let (y, caches) = net.prime(batch, caches.map(M::unwrap_caches), class);
    (y, caches.map(M::wrap_caches))
}

// ===========================================================================
// The knobs every network config shares
// ===========================================================================

/// The family-independent half of a network config: everything about the
/// *stack* rather than the block.
///
/// Split out because it is identical for all three families — a variant of
/// [`DeltaLatentNetConfig`] / [`DeltaVocabNetConfig`] then adds only its block
/// config.
#[derive(Config, Debug)]
pub struct DeltaNetworkShape {
    /// Number of real weight sets.
    pub n_real_layers: usize,

    /// Optional virtual-layer scheduling: run `n` logical layers over the real
    /// weight sets, mapped by a [`Schedule`].
    #[config(default = "None")]
    pub n_virtual_layers: Option<(usize, Schedule)>,

    /// Back-propagate only the last `K` virtual layers, running everything
    /// below on the inner backend (truncated BPTT for deep recursion).
    /// `None` ⇒ track the whole stack. See
    /// [`Layers::grad_horizon`](burn_stack::modules::Layers::grad_horizon).
    #[config(default = "None")]
    pub grad_horizon: Option<usize>,

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
    #[config(default = "None")]
    pub mlp: Option<GatedMlpConfig>,
}

impl DeltaNetworkShape {
    /// Build the layer-stack builder for a given block config.
    fn layers<C: burn_stack::modules::BlockConfig>(&self, block: C) -> LayersBuilder<C> {
        LayersBuilder::new(self.n_real_layers, block)
            .with_n_virtual_layers(self.n_virtual_layers.clone())
            .with_grad_horizon(self.grad_horizon)
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

/// A runtime-selectable latent network: the same `in_proj → Layers → out_proj`
/// shape over any delta-rule family, chosen at construction.
#[derive(Module, Debug)]
pub enum DeltaLatentNet {
    /// DeltaNet latent network.
    #[cfg(feature = "deltanet")]
    DeltaNet(LatentNetwork<crate::deltanet::prelude::DeltaNet>),
    /// Gated DeltaNet latent network.
    #[cfg(feature = "gated-deltanet")]
    GatedDeltaNet(LatentNetwork<crate::gated_deltanet::prelude::GatedDeltaNet>),
    /// DeltaProduct latent network.
    #[cfg(feature = "delta-product")]
    DeltaProduct(LatentNetwork<crate::delta_product::prelude::DeltaProduct>),
    /// GDN-2 latent network.
    #[cfg(feature = "gdn2")]
    GatedDeltaNet2(LatentNetwork<crate::gdn2::prelude::GatedDeltaNet2>),
}

impl DeltaLatentNet {
    /// Full-sequence pass. The caches must belong to this network's family; a
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
            Self::DeltaNet(net) => latent_forward(net, x, caches, path, class),
            #[cfg(feature = "gated-deltanet")]
            Self::GatedDeltaNet(net) => latent_forward(net, x, caches, path, class),
            #[cfg(feature = "delta-product")]
            Self::DeltaProduct(net) => latent_forward(net, x, caches, path, class),
            #[cfg(feature = "gdn2")]
            Self::GatedDeltaNet2(net) => latent_forward(net, x, caches, path, class),
        }
    }

    /// Single-token step. No path argument — decoding is recurrent for every
    /// family.
    pub fn step(
        &self,
        x: Tensor<2>,
        caches: Option<DeltaCaches>,
        class: Option<&mut ClassCursors>,
    ) -> (Tensor<2>, DeltaCaches) {
        match self {
            #[cfg(feature = "deltanet")]
            Self::DeltaNet(net) => latent_step(net, x, caches, class),
            #[cfg(feature = "gated-deltanet")]
            Self::GatedDeltaNet(net) => latent_step(net, x, caches, class),
            #[cfg(feature = "delta-product")]
            Self::DeltaProduct(net) => latent_step(net, x, caches, class),
            #[cfg(feature = "gdn2")]
            Self::GatedDeltaNet2(net) => latent_step(net, x, caches, class),
        }
    }

    /// Class-only step: emit the class markers waiting for the next user token
    /// without one, returning the last of them (`None` when none were waiting).
    pub fn prime(
        &self,
        batch: usize,
        caches: Option<DeltaCaches>,
        class: Option<&mut ClassCursors>,
    ) -> (Option<Tensor<2>>, Option<DeltaCaches>) {
        match self {
            #[cfg(feature = "deltanet")]
            Self::DeltaNet(net) => latent_prime(net, batch, caches, class),
            #[cfg(feature = "gated-deltanet")]
            Self::GatedDeltaNet(net) => latent_prime(net, batch, caches, class),
            #[cfg(feature = "delta-product")]
            Self::DeltaProduct(net) => latent_prime(net, batch, caches, class),
            #[cfg(feature = "gdn2")]
            Self::GatedDeltaNet2(net) => latent_prime(net, batch, caches, class),
        }
    }
}

/// Config for [`DeltaLatentNet`]: pick the family here, and the enum picks the
/// monomorphisation.
#[derive(Config, Debug)]
pub enum DeltaLatentNetConfig {
    /// Build a DeltaNet latent network.
    #[cfg(feature = "deltanet")]
    DeltaNet {
        /// Stack-level knobs.
        shape: DeltaLatentShape,
        /// Block config.
        block: crate::deltanet::prelude::DeltaNetConfig,
    },
    /// Build a Gated DeltaNet latent network.
    #[cfg(feature = "gated-deltanet")]
    GatedDeltaNet {
        /// Stack-level knobs.
        shape: DeltaLatentShape,
        /// Block config.
        block: crate::gated_deltanet::prelude::GatedDeltaNetConfig,
    },
    /// Build a DeltaProduct latent network.
    #[cfg(feature = "delta-product")]
    DeltaProduct {
        /// Stack-level knobs.
        shape: DeltaLatentShape,
        /// Block config.
        block: crate::delta_product::prelude::DeltaProductConfig,
    },
    /// Build a GDN-2 latent network.
    #[cfg(feature = "gdn2")]
    GatedDeltaNet2 {
        /// Stack-level knobs.
        shape: DeltaLatentShape,
        /// Block config.
        block: crate::gdn2::prelude::GatedDeltaNet2Config,
    },
}

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
    fn build<C: burn_stack::modules::BlockConfig>(&self, block: C) -> LatentNetworkBuilder<C> {
        LatentNetworkBuilder {
            input_size: self.input_size,
            layers: self.stack.layers(block),
            output_size: self.output_size,
            final_norm: self.final_norm,
            class_tokens: self.class_tokens.clone(),
        }
    }
}

impl DeltaLatentNetConfig {
    /// The [`MuonPlan`](burn_stack::optim::MuonPlan) for this network: the
    /// block's fused projections plus the optional MLP's.
    ///
    /// The network's own boundary weights — `in_proj`/`out_proj`, class-token
    /// tables — are deliberately left out; see [`burn_stack::optim`].
    #[cfg(feature = "optim")]
    pub fn muon_plan(&self) -> burn_stack::optim::MuonPlan {
        let (specs, mlp) = match self {
            #[cfg(feature = "deltanet")]
            Self::DeltaNet { shape, block } => {
                (block.muon_projections(), shape.stack.mlp.clone())
            }
            #[cfg(feature = "gated-deltanet")]
            Self::GatedDeltaNet { shape, block } => {
                (block.muon_projections(), shape.stack.mlp.clone())
            }
            #[cfg(feature = "delta-product")]
            Self::DeltaProduct { shape, block } => {
                (block.muon_projections(), shape.stack.mlp.clone())
            }
            #[cfg(feature = "gdn2")]
            Self::GatedDeltaNet2 { shape, block } => {
                (block.muon_projections(), shape.stack.mlp.clone())
            }
        };
        burn_stack::optim::MuonPlan::new(specs).with_mlp(mlp.as_ref())
    }

    /// Allocate and initialise the selected network on `device`.
    pub fn init(&self, device: &Device) -> DeltaLatentNet {
        match self {
            #[cfg(feature = "deltanet")]
            Self::DeltaNet { shape, block } => {
                DeltaLatentNet::DeltaNet(shape.build(block.clone()).init(device))
            }
            #[cfg(feature = "gated-deltanet")]
            Self::GatedDeltaNet { shape, block } => {
                DeltaLatentNet::GatedDeltaNet(shape.build(block.clone()).init(device))
            }
            #[cfg(feature = "delta-product")]
            Self::DeltaProduct { shape, block } => {
                DeltaLatentNet::DeltaProduct(shape.build(block.clone()).init(device))
            }
            #[cfg(feature = "gdn2")]
            Self::GatedDeltaNet2 { shape, block } => {
                DeltaLatentNet::GatedDeltaNet2(shape.build(block.clone()).init(device))
            }
        }
    }
}

// ===========================================================================
// DeltaVocabNet
// ===========================================================================

/// A runtime-selectable token language model: the same
/// `Embedding → Layers → norm_f → LM head` shape over any delta-rule family.
/// The vocabulary counterpart of [`DeltaLatentNet`].
#[derive(Module, Debug)]
pub enum DeltaVocabNet {
    /// DeltaNet language model.
    #[cfg(feature = "deltanet")]
    DeltaNet(VocabNetwork<crate::deltanet::prelude::DeltaNet>),
    /// Gated DeltaNet language model.
    #[cfg(feature = "gated-deltanet")]
    GatedDeltaNet(VocabNetwork<crate::gated_deltanet::prelude::GatedDeltaNet>),
    /// DeltaProduct language model.
    #[cfg(feature = "delta-product")]
    DeltaProduct(VocabNetwork<crate::delta_product::prelude::DeltaProduct>),
    /// GDN-2 language model.
    #[cfg(feature = "gdn2")]
    GatedDeltaNet2(VocabNetwork<crate::gdn2::prelude::GatedDeltaNet2>),
}

impl DeltaVocabNet {
    /// Full-sequence pass: token IDs `[batch, sequence]` → logits
    /// `[batch, sequence, padded_vocab]`.
    pub fn forward(
        &self,
        x: Tensor<2, Int>,
        caches: Option<DeltaCaches>,
        path: DeltaPath,
        class: Option<&mut ClassCursors>,
    ) -> (Tensor<3>, DeltaCaches) {
        match self {
            #[cfg(feature = "deltanet")]
            Self::DeltaNet(net) => vocab_forward(net, x, caches, path, class),
            #[cfg(feature = "gated-deltanet")]
            Self::GatedDeltaNet(net) => vocab_forward(net, x, caches, path, class),
            #[cfg(feature = "delta-product")]
            Self::DeltaProduct(net) => vocab_forward(net, x, caches, path, class),
            #[cfg(feature = "gdn2")]
            Self::GatedDeltaNet2(net) => vocab_forward(net, x, caches, path, class),
        }
    }

    /// Single-token step: token IDs `[batch]` → logits `[batch, padded_vocab]`.
    pub fn step(
        &self,
        x: Tensor<1, Int>,
        caches: Option<DeltaCaches>,
        class: Option<&mut ClassCursors>,
    ) -> (Tensor<2>, DeltaCaches) {
        match self {
            #[cfg(feature = "deltanet")]
            Self::DeltaNet(net) => vocab_step(net, x, caches, class),
            #[cfg(feature = "gated-deltanet")]
            Self::GatedDeltaNet(net) => vocab_step(net, x, caches, class),
            #[cfg(feature = "delta-product")]
            Self::DeltaProduct(net) => vocab_step(net, x, caches, class),
            #[cfg(feature = "gdn2")]
            Self::GatedDeltaNet2(net) => vocab_step(net, x, caches, class),
        }
    }

    /// Step the class latents the stack has waiting, with no token of its own —
    /// the seedless-generation entry point (`prime` → sample → `step` → …).
    pub fn prime(
        &self,
        batch: usize,
        caches: Option<DeltaCaches>,
        class: Option<&mut ClassCursors>,
    ) -> (Option<Tensor<2>>, Option<DeltaCaches>) {
        match self {
            #[cfg(feature = "deltanet")]
            Self::DeltaNet(net) => vocab_prime(net, batch, caches, class),
            #[cfg(feature = "gated-deltanet")]
            Self::GatedDeltaNet(net) => vocab_prime(net, batch, caches, class),
            #[cfg(feature = "delta-product")]
            Self::DeltaProduct(net) => vocab_prime(net, batch, caches, class),
            #[cfg(feature = "gdn2")]
            Self::GatedDeltaNet2(net) => vocab_prime(net, batch, caches, class),
        }
    }
}

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
    fn build<C: burn_stack::modules::BlockConfig>(&self, block: C) -> VocabNetworkBuilder<C> {
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
pub enum DeltaVocabNetConfig {
    /// Build a DeltaNet language model.
    #[cfg(feature = "deltanet")]
    DeltaNet {
        /// Vocabulary + stack knobs.
        shape: DeltaVocabShape,
        /// Block config.
        block: crate::deltanet::prelude::DeltaNetConfig,
    },
    /// Build a Gated DeltaNet language model.
    #[cfg(feature = "gated-deltanet")]
    GatedDeltaNet {
        /// Vocabulary + stack knobs.
        shape: DeltaVocabShape,
        /// Block config.
        block: crate::gated_deltanet::prelude::GatedDeltaNetConfig,
    },
    /// Build a DeltaProduct language model.
    #[cfg(feature = "delta-product")]
    DeltaProduct {
        /// Vocabulary + stack knobs.
        shape: DeltaVocabShape,
        /// Block config.
        block: crate::delta_product::prelude::DeltaProductConfig,
    },
    /// Build a GDN-2 language model.
    #[cfg(feature = "gdn2")]
    GatedDeltaNet2 {
        /// Vocabulary + stack knobs.
        shape: DeltaVocabShape,
        /// Block config.
        block: crate::gdn2::prelude::GatedDeltaNet2Config,
    },
}

impl DeltaVocabNetConfig {
    /// The [`MuonPlan`](burn_stack::optim::MuonPlan) for this model. The
    /// embedding and LM head stay on the fallback optimizer.
    #[cfg(feature = "optim")]
    pub fn muon_plan(&self) -> burn_stack::optim::MuonPlan {
        let (specs, mlp) = match self {
            #[cfg(feature = "deltanet")]
            Self::DeltaNet { shape, block } => {
                (block.muon_projections(), shape.stack.mlp.clone())
            }
            #[cfg(feature = "gated-deltanet")]
            Self::GatedDeltaNet { shape, block } => {
                (block.muon_projections(), shape.stack.mlp.clone())
            }
            #[cfg(feature = "delta-product")]
            Self::DeltaProduct { shape, block } => {
                (block.muon_projections(), shape.stack.mlp.clone())
            }
            #[cfg(feature = "gdn2")]
            Self::GatedDeltaNet2 { shape, block } => {
                (block.muon_projections(), shape.stack.mlp.clone())
            }
        };
        burn_stack::optim::MuonPlan::new(specs).with_mlp(mlp.as_ref())
    }

    /// Allocate and initialise the selected model on `device`.
    pub fn init(&self, device: &Device) -> DeltaVocabNet {
        match self {
            #[cfg(feature = "deltanet")]
            Self::DeltaNet { shape, block } => {
                DeltaVocabNet::DeltaNet(shape.build(block.clone()).init(device))
            }
            #[cfg(feature = "gated-deltanet")]
            Self::GatedDeltaNet { shape, block } => {
                DeltaVocabNet::GatedDeltaNet(shape.build(block.clone()).init(device))
            }
            #[cfg(feature = "delta-product")]
            Self::DeltaProduct { shape, block } => {
                DeltaVocabNet::DeltaProduct(shape.build(block.clone()).init(device))
            }
            #[cfg(feature = "gdn2")]
            Self::GatedDeltaNet2 { shape, block } => {
                DeltaVocabNet::GatedDeltaNet2(shape.build(block.clone()).init(device))
            }
        }
    }
}
