//! # DeltaNet
//!
//! The delta rule with no forget gate: the state changes only when a token
//! deliberately writes to it. See [`deltanet`](crate::deltanet::deltanet) for
//! the block and the architecture diagram, and [`crate::delta`] for the
//! recurrence itself.
//!
//! - [`deltanet`](crate::deltanet::deltanet) — the block and its config.
//! - [`cache`](crate::deltanet::cache) — the convolution window + recurrent
//!   state carried between calls.
//!
//! The residual layer, the layer stack, the language model and the
//! bidirectional wrappers are the family-generic types in
//! [`burn_stack::modules`], reached through [`crate::unified`].

pub mod cache;
pub mod deltanet;

/// Public re-exports for DeltaNet.
pub mod prelude {
    pub use super::cache::{
        DeltaNetCache, DeltaNetCacheConfig, DeltaNetCaches, DeltaNetCachesConfig,
    };
    pub use super::deltanet::{DeltaNet, DeltaNetConfig};
}
