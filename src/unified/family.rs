//! The bridge from a concrete family block to the runtime-tagged
//! [`DeltaCaches`].
//!
//! All three families take the same per-call options ([`DeltaPath`]) and carry
//! the same *shape* of state, so the only thing a runtime-dispatching container
//! needs from a family is how to tag and untag its caches. That is this trait —
//! and it is what keeps [`network`](super::network) and [`bidi`](super::bidi)
//! to one line per family instead of a copied block each.

use burn_stack::modules::Block;

use crate::delta::path::DeltaPath;
use crate::unified::cache::DeltaCaches;

/// A block family reachable through the runtime-selectable enums.
pub trait DeltaFamily: Block<Options = DeltaPath> {
    /// The family's name, used in cache-mismatch panics.
    const NAME: &'static str;

    /// Tag this family's caches for transport through a runtime enum.
    fn wrap_caches(caches: Self::Caches) -> DeltaCaches;

    /// Untag caches, panicking if they belong to a different family — a caller
    /// error (the runtime enums cannot check it at compile time, which is the
    /// price of runtime selection).
    fn unwrap_caches(caches: DeltaCaches) -> Self::Caches;
}

macro_rules! impl_family {
    ($block:ty, $variant:ident, $name:literal) => {
        impl DeltaFamily for $block {
            const NAME: &'static str = $name;

            fn wrap_caches(caches: Self::Caches) -> DeltaCaches {
                DeltaCaches::$variant(caches)
            }

            fn unwrap_caches(caches: DeltaCaches) -> Self::Caches {
                match caches {
                    DeltaCaches::$variant(caches) => caches,
                    #[allow(unreachable_patterns)]
                    other => panic!(
                        "cache family mismatch: a {} network was given {} caches",
                        $name,
                        other.family_name(),
                    ),
                }
            }
        }
    };
}

#[cfg(feature = "deltanet")]
impl_family!(crate::deltanet::prelude::DeltaNet, DeltaNet, "DeltaNet");
#[cfg(feature = "gated-deltanet")]
impl_family!(
    crate::gated_deltanet::prelude::GatedDeltaNet,
    GatedDeltaNet,
    "Gated DeltaNet"
);
#[cfg(feature = "delta-product")]
impl_family!(
    crate::delta_product::prelude::DeltaProduct,
    DeltaProduct,
    "DeltaProduct"
);
