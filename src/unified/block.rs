//! The runtime-selectable *block*: one enum over the four families, itself a
//! [`Block`].
//!
//! This is where a family chosen at run time enters the stack. Every container
//! in `burn-stack` is generic over one block type, so a network whose family
//! comes out of a config file needs a single type that can be any of them —
//! and the delta families make that cheap, because they agree on everything a
//! container can see:
//!
//! - the same per-call options ([`DeltaPath`]),
//! - the same cache ([`DeltaCache`]: a convolution window and one
//!   `[batch, nheads, head_k_dim, head_v_dim]` associative memory),
//! - the same `[batch, sequence, d_model]` interface.
//!
//! So `LatentNetwork<DeltaBlock>` *is* the runtime-selectable network — there is
//! nothing left for a per-family network enum to dispatch. What a family
//! actually differs in (a scalar `β`, a per-head forget gate, per-channel
//! erase/write gates, `u` Householder factors) is settled inside
//! [`DeltaBlock::forward`] and never reaches the container.
//!
//! When the family *is* known statically, name the block directly —
//! `LatentNetwork<GatedDeltaNet1>` — and skip the dispatch: each family
//! implements [`Block`] on its own account (see [`super::cache`]).

use burn::prelude::*;
use burn_stack::modules::{Block, BlockConfig};

use crate::common::cache::{DeltaCache, DeltaCaches};
use crate::delta::path::DeltaPath;

/// A delta-rule block whose family is chosen at construction.
#[derive(Module, Debug)]
pub enum DeltaBlock {
    /// DeltaNet: no forget gate.
    #[cfg(feature = "deltanet")]
    DeltaNet(crate::deltanet::prelude::DeltaNet),
    /// Gated DeltaNet: the scalar forget gate.
    #[cfg(feature = "gated-deltanet-1")]
    GatedDeltaNet1(crate::gated_deltanet_1::prelude::GatedDeltaNet1),
    /// DeltaProduct: `u` Householder factors per transition.
    #[cfg(feature = "delta-product")]
    DeltaProduct(crate::delta_product::prelude::DeltaProduct),
    /// GDN-2: erase/write/decay per channel.
    #[cfg(feature = "gated-deltanet-2")]
    GatedDeltaNet2(crate::gated_deltanet_2::prelude::GatedDeltaNet2),
}

/// Run `$body` on the selected family's block, bound as `$inner`.
macro_rules! dispatch {
    ($self:expr, $inner:ident => $body:expr) => {
        match $self {
            #[cfg(feature = "deltanet")]
            Self::DeltaNet($inner) => $body,
            #[cfg(feature = "gated-deltanet-1")]
            Self::GatedDeltaNet1($inner) => $body,
            #[cfg(feature = "delta-product")]
            Self::DeltaProduct($inner) => $body,
            #[cfg(feature = "gated-deltanet-2")]
            Self::GatedDeltaNet2($inner) => $body,
        }
    };
}

impl DeltaBlock {
    /// The family this block belongs to, for diagnostics.
    pub fn family_name(&self) -> &'static str {
        match self {
            #[cfg(feature = "deltanet")]
            Self::DeltaNet(_) => "DeltaNet",
            #[cfg(feature = "gated-deltanet-1")]
            Self::GatedDeltaNet1(_) => "Gated DeltaNet",
            #[cfg(feature = "delta-product")]
            Self::DeltaProduct(_) => "DeltaProduct",
            #[cfg(feature = "gated-deltanet-2")]
            Self::GatedDeltaNet2(_) => "GDN-2",
        }
    }

    /// Model width.
    pub fn d_model(&self) -> usize {
        dispatch!(self, block => block.d_model())
    }

    /// Number of state heads.
    pub fn nheads(&self) -> usize {
        dispatch!(self, block => block.nheads())
    }

    /// Query/key width per head.
    pub fn head_k_dim(&self) -> usize {
        dispatch!(self, block => block.head_k_dim())
    }

    /// Value width per head.
    pub fn head_v_dim(&self) -> usize {
        dispatch!(self, block => block.head_v_dim())
    }

    /// Full-sequence pass — see the selected family's `forward`.
    pub fn forward(
        &self,
        x_bsd: Tensor<3>,
        cache: Option<DeltaCache>,
        path: DeltaPath,
    ) -> (Tensor<3>, DeltaCache) {
        dispatch!(self, block => block.forward(x_bsd, cache, path))
    }

    /// Single-token recurrent step — see the selected family's `step`.
    pub fn step(&self, x_bd: Tensor<2>, cache: Option<DeltaCache>) -> (Tensor<2>, DeltaCache) {
        dispatch!(self, block => block.step(x_bd, cache))
    }

    /// Zero caches for `n_virtual` layers at this batch size.
    pub fn zero_caches(&self, batch: usize, n_virtual: usize, device: &Device) -> DeltaCaches {
        dispatch!(self, block => block.zero_caches(batch, n_virtual, device))
    }
}

impl Block for DeltaBlock {
    type Cache = DeltaCache;
    type Caches = DeltaCaches;
    /// Which delta-rule evaluation runs, and at what chunk length.
    type Options = DeltaPath;

    fn block_forward(
        &self,
        x: Tensor<3>,
        cache: Option<DeltaCache>,
        options: DeltaPath,
    ) -> (Tensor<3>, DeltaCache) {
        self.forward(x, cache, options)
    }

    fn block_step(&self, x: Tensor<2>, cache: Option<DeltaCache>) -> (Tensor<2>, DeltaCache) {
        self.step(x, cache)
    }

    fn zero_caches_3d(&self, x: &Tensor<3>, n_virtual: usize) -> DeltaCaches {
        let [batch, _sequence, _d_model] = x.dims();
        self.zero_caches(batch, n_virtual, &x.device())
    }

    fn zero_caches_2d(&self, x: &Tensor<2>, n_virtual: usize) -> DeltaCaches {
        let [batch, _d_model] = x.dims();
        self.zero_caches(batch, n_virtual, &x.device())
    }
}

/// Config for [`DeltaBlock`]: pick the family here, and the enum picks the
/// block.
#[derive(Config, Debug)]
pub enum DeltaBlockConfig {
    /// Build a DeltaNet block.
    #[cfg(feature = "deltanet")]
    DeltaNet(crate::deltanet::prelude::DeltaNetConfig),
    /// Build a Gated DeltaNet block.
    #[cfg(feature = "gated-deltanet-1")]
    GatedDeltaNet1(crate::gated_deltanet_1::prelude::GatedDeltaNet1Config),
    /// Build a DeltaProduct block.
    #[cfg(feature = "delta-product")]
    DeltaProduct(crate::delta_product::prelude::DeltaProductConfig),
    /// Build a GDN-2 block.
    #[cfg(feature = "gated-deltanet-2")]
    GatedDeltaNet2(crate::gated_deltanet_2::prelude::GatedDeltaNet2Config),
}

impl DeltaBlockConfig {
    /// Model width.
    pub fn d_model(&self) -> usize {
        dispatch!(self, config => config.d_model)
    }

    /// Allocate and initialise the selected block on `device`.
    pub fn init(&self, device: &Device) -> DeltaBlock {
        match self {
            #[cfg(feature = "deltanet")]
            Self::DeltaNet(config) => DeltaBlock::DeltaNet(config.init(device)),
            #[cfg(feature = "gated-deltanet-1")]
            Self::GatedDeltaNet1(config) => DeltaBlock::GatedDeltaNet1(config.init(device)),
            #[cfg(feature = "delta-product")]
            Self::DeltaProduct(config) => DeltaBlock::DeltaProduct(config.init(device)),
            #[cfg(feature = "gated-deltanet-2")]
            Self::GatedDeltaNet2(config) => DeltaBlock::GatedDeltaNet2(config.init(device)),
        }
    }

    /// The selected family's Muon allowlist.
    ///
    /// Independent of the enum: a [`ProjSpec`](burn_stack::optim::ProjSpec)
    /// names a weight *under* a block container, and the variant name the enum
    /// adds to the parameter path sits between the two — which is why the spec
    /// matches its container and its weight as separate substrings.
    #[cfg(feature = "optim")]
    pub fn muon_projections(&self) -> Vec<burn_stack::optim::ProjSpec> {
        dispatch!(self, config => config.muon_projections())
    }
}

impl BlockConfig for DeltaBlockConfig {
    type Block = DeltaBlock;

    fn d_model(&self) -> usize {
        self.d_model()
    }

    fn init_block(&self, device: &Device) -> DeltaBlock {
        self.init(device)
    }

    #[cfg(feature = "optim")]
    fn muon_projections(&self) -> Vec<burn_stack::optim::ProjSpec> {
        self.muon_projections()
    }
}
