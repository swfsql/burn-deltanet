//! # Gated DeltaNet
//!
//! The delta rule plus a Mamba-2-style scalar forget gate: targeted overwrites
//! *and* indiscriminate decay, which the paper shows are complementary rather
//! than redundant. See
//! [`gated_deltanet`] for the block and
//! the gate's parameterisation, and [`crate::delta`] for the recurrence.
//!
//! - [`gated_deltanet`] — the block and
//!   its config.
//! - [`cache`] — the convolution window +
//!   recurrent state carried between calls.
//!
//! The residual layer, the layer stack, the language model and the
//! bidirectional wrappers are the family-generic types in
//! [`burn_stack::modules`], reached through [`crate::unified`].

pub mod cache;
pub mod gated_deltanet;

/// Public re-exports for Gated DeltaNet.
pub mod prelude {
    pub use super::cache::{
        GatedDeltaNetCache, GatedDeltaNetCacheConfig, GatedDeltaNetCaches,
        GatedDeltaNetCachesConfig,
    };
    pub use super::gated_deltanet::{GatedDeltaNet, GatedDeltaNetConfig};
}
