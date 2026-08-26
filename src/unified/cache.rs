//! Where each family plugs into the block-generic stack: `impl Block`,
//! `impl BlockConfig`, `impl CacheStack` — plus the runtime-tagged
//! [`DeltaCaches`] the runtime-selectable networks thread through.
//!
//! The three impls are near-identical, because the three families differ in
//! what they *project*, not in what they *carry*: one convolution window and
//! one `[batch, nheads, head_k_dim, head_v_dim]` associative memory, whatever
//! the transition does to it.

use burn::prelude::*;
use burn_stack::modules::{Block, BlockConfig, CacheStack};

use crate::delta::path::DeltaPath;

/// Runtime-tagged caches: one variant per family, matching
/// [`DeltaLatentNet`](crate::unified::DeltaLatentNet).
///
/// Plain runtime state, not a `Module`: caches are threaded through
/// `forward`/`step`, never recorded or optimised.
#[derive(Debug, Clone)]
pub enum DeltaCaches {
    /// DeltaNet caches.
    #[cfg(feature = "deltanet")]
    DeltaNet(crate::deltanet::prelude::DeltaNetCaches),
    /// Gated DeltaNet caches.
    #[cfg(feature = "gated-deltanet")]
    GatedDeltaNet(crate::gated_deltanet::prelude::GatedDeltaNetCaches),
    /// DeltaProduct caches.
    #[cfg(feature = "delta-product")]
    DeltaProduct(crate::delta_product::prelude::DeltaProductCaches),
}

/// Emit the four impls a family owes the stack.
///
/// Each family's cache is the same two fields under a different name, and each
/// block's `forward`/`step` have the same signature, so the wiring is the same
/// text three times over — written once here rather than copied.
macro_rules! impl_block_for_family {
    (
        block: $block:ty,
        config: $config:ty,
        cache: $cache:ty,
        caches: $caches:ty,
    ) => {
        impl CacheStack for $caches {
            type Cache = $cache;

            fn slot_count(&self) -> usize {
                self.caches.len()
            }

            fn into_slots(self) -> Vec<Option<$cache>> {
                self.into_options()
            }

            fn from_slots(slots: Vec<Option<$cache>>) -> Self {
                Self::from_options(slots)
            }

            // Spelled out field by field: `Module::map` is a no-op on the bare
            // `Tensor`s a cache holds, so a derive would silently skip them.
            fn cache_to_inner(cache: $cache) -> $cache {
                <$cache>::from_parts(
                    cache.conv_bwc.map(|t| t.inner()),
                    cache.state_bhkv.inner(),
                )
            }

            fn cache_from_inner(cache: $cache) -> $cache {
                <$cache>::from_parts(
                    cache.conv_bwc.map(Tensor::from_inner),
                    Tensor::from_inner(cache.state_bhkv),
                )
            }
        }

        impl Block for $block {
            type Cache = $cache;
            type Caches = $caches;
            /// Which delta-rule evaluation runs, and at what chunk length.
            type Options = DeltaPath;

            fn block_forward(
                &self,
                x: Tensor<3>,
                cache: Option<$cache>,
                options: DeltaPath,
            ) -> (Tensor<3>, $cache) {
                self.forward(x, cache, options)
            }

            fn block_step(&self, x: Tensor<2>, cache: Option<$cache>) -> (Tensor<2>, $cache) {
                self.step(x, cache)
            }

            fn zero_caches_3d(&self, x: &Tensor<3>, n_virtual: usize) -> $caches {
                let [batch, _sequence, _d_model] = x.dims();
                self.zero_caches(batch, n_virtual, &x.device())
            }

            fn zero_caches_2d(&self, x: &Tensor<2>, n_virtual: usize) -> $caches {
                let [batch, _d_model] = x.dims();
                self.zero_caches(batch, n_virtual, &x.device())
            }
        }

        impl BlockConfig for $config {
            type Block = $block;

            fn d_model(&self) -> usize {
                self.d_model
            }

            fn init_block(&self, device: &Device) -> $block {
                self.init(device)
            }

            #[cfg(feature = "optim")]
            fn muon_projections(&self) -> Vec<burn_stack::optim::ProjSpec> {
                self.muon_projections()
            }
        }
    };
}

#[cfg(feature = "deltanet")]
mod impl_deltanet {
    use super::*;
    use crate::deltanet::prelude::{DeltaNet, DeltaNetCache, DeltaNetCaches, DeltaNetConfig};

    impl_block_for_family! {
        block: DeltaNet,
        config: DeltaNetConfig,
        cache: DeltaNetCache,
        caches: DeltaNetCaches,
    }
}

#[cfg(feature = "gated-deltanet")]
mod impl_gated_deltanet {
    use super::*;
    use crate::gated_deltanet::prelude::{
        GatedDeltaNet, GatedDeltaNetCache, GatedDeltaNetCaches, GatedDeltaNetConfig,
    };

    impl_block_for_family! {
        block: GatedDeltaNet,
        config: GatedDeltaNetConfig,
        cache: GatedDeltaNetCache,
        caches: GatedDeltaNetCaches,
    }
}

#[cfg(feature = "delta-product")]
mod impl_delta_product {
    use super::*;
    use crate::delta_product::prelude::{
        DeltaProduct, DeltaProductCache, DeltaProductCaches, DeltaProductConfig,
    };

    impl_block_for_family! {
        block: DeltaProduct,
        config: DeltaProductConfig,
        cache: DeltaProductCache,
        caches: DeltaProductCaches,
    }
}

impl DeltaCaches {
    /// The family these caches belong to, for mismatch messages.
    pub fn family_name(&self) -> &'static str {
        match self {
            #[cfg(feature = "deltanet")]
            Self::DeltaNet(_) => "DeltaNet",
            #[cfg(feature = "gated-deltanet")]
            Self::GatedDeltaNet(_) => "Gated DeltaNet",
            #[cfg(feature = "delta-product")]
            Self::DeltaProduct(_) => "DeltaProduct",
        }
    }

    /// Number of per-(virtual-)layer slots.
    pub fn slot_count(&self) -> usize {
        match self {
            #[cfg(feature = "deltanet")]
            Self::DeltaNet(c) => c.caches_len(),
            #[cfg(feature = "gated-deltanet")]
            Self::GatedDeltaNet(c) => c.caches_len(),
            #[cfg(feature = "delta-product")]
            Self::DeltaProduct(c) => c.caches_len(),
        }
    }
}
