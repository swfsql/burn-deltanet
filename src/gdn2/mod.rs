//! # GDN-2 (Gated DeltaNet 2)
//!
//! [Gated DeltaNet](crate::gated_deltanet) with its two per-head gates widened
//! onto channel axes: an **erase** gate `b ∈ ℝ^head_k_dim` and a **write** gate
//! `w ∈ ℝ^head_v_dim` in place of the single `β`, and a per-key-channel forget
//! gate in place of the scalar `α`. Deleting a fact and accumulating into one
//! become single-token operations, which a shared `β` cannot express. See
//! [`gdn2`] for the block and the argument, and [`crate::delta`] for the
//! recurrence they all share.
//!
//! - [`gdn2`] — the block and its config.
//! - [`cache`] — the convolution window + recurrent state carried between
//!   calls.
//!
//! The residual layer, the layer stack, the language model and the
//! bidirectional wrappers are the family-generic types in
//! [`burn_stack::modules`], reached through [`crate::unified`].

pub mod cache;
pub mod gdn2;

/// Public re-exports for GDN-2.
pub mod prelude {
    pub use super::cache::{
        GatedDeltaNet2Cache, GatedDeltaNet2CacheConfig, GatedDeltaNet2Caches,
        GatedDeltaNet2CachesConfig,
    };
    pub use super::gdn2::{GatedDeltaNet2, GatedDeltaNet2Config};
}
