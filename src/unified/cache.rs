//! Where each family plugs into the block-generic stack: `impl Block`,
//! `impl BlockConfig`, and the single `impl CacheStack` all four share.
//!
//! The impls are near-identical, because the families differ in what they
//! *project*, not in what they *carry*: one convolution window and one
//! `[batch, nheads, head_k_dim, head_v_dim]` associative memory, whatever the
//! transition does to it. That is also why the cache itself is one type
//! ([`DeltaCaches`]) rather than one per family — see
//! [`crate::common::cache`].

use burn::prelude::*;
use burn_stack::modules::{Block, BlockConfig, CacheStack};

use crate::common::cache::{DeltaCache, DeltaCaches};
use crate::delta::path::DeltaPath;

impl CacheStack for DeltaCaches {
    type Cache = DeltaCache;

    fn slot_count(&self) -> usize {
        self.caches.len()
    }

    fn into_slots(self) -> Vec<Option<DeltaCache>> {
        self.into_options()
    }

    fn from_slots(slots: Vec<Option<DeltaCache>>) -> Self {
        Self::from_options(slots)
    }

    // Spelled out field by field: `Module::map` is a no-op on the bare
    // `Tensor`s a cache holds, so a derive would silently skip them.
    fn cache_to_inner(cache: DeltaCache) -> DeltaCache {
        DeltaCache::from_parts(
            cache.conv_bwc.map(|t| t.inner()),
            cache.state_bhkv.inner(),
        )
    }

    fn cache_from_inner(cache: DeltaCache) -> DeltaCache {
        DeltaCache::from_parts(
            cache.conv_bwc.map(Tensor::from_inner),
            Tensor::from_inner(cache.state_bhkv),
        )
    }
}

/// Emit the two impls a family owes the stack.
///
/// Each block's `forward`/`step` have the same signature over the same cache,
/// so the wiring is the same text once per family — written once here rather
/// than copied.
macro_rules! impl_block_for_family {
    (
        block: $block:ty,
        config: $config:ty,
    ) => {
        impl Block for $block {
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
    use crate::deltanet::prelude::{DeltaNet, DeltaNetConfig};

    impl_block_for_family! {
        block: DeltaNet,
        config: DeltaNetConfig,
    }
}

#[cfg(feature = "gated-deltanet-1")]
mod impl_gated_deltanet_1 {
    use super::*;
    use crate::gated_deltanet_1::prelude::{GatedDeltaNet1, GatedDeltaNet1Config};

    impl_block_for_family! {
        block: GatedDeltaNet1,
        config: GatedDeltaNet1Config,
    }
}

#[cfg(feature = "delta-product")]
mod impl_delta_product {
    use super::*;
    use crate::delta_product::prelude::{DeltaProduct, DeltaProductConfig};

    impl_block_for_family! {
        block: DeltaProduct,
        config: DeltaProductConfig,
    }
}

#[cfg(feature = "gated-deltanet-2")]
mod impl_gdn2 {
    use super::*;
    use crate::gated_deltanet_2::prelude::{GatedDeltaNet2, GatedDeltaNet2Config};

    impl_block_for_family! {
        block: GatedDeltaNet2,
        config: GatedDeltaNet2Config,
    }
}
