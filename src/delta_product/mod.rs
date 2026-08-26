//! # DeltaProduct
//!
//! `n_householder` delta-rule micro-steps per token, so one transition is a
//! *product* of generalised Householder reflections rather than a single one —
//! a direct dial on how much group structure the recurrence can track, at no
//! extra state. See
//! [`delta_product`](crate::delta_product::delta_product) for why that follows
//! from Cartan–Dieudonné and how it is evaluated without a new kernel.
//!
//! - [`delta_product`](crate::delta_product::delta_product) — the block and its
//!   config.
//! - [`cache`](crate::delta_product::cache) — the convolution window +
//!   recurrent state carried between calls.
//!
//! The residual layer, the layer stack, the language model and the
//! bidirectional wrappers are the family-generic types in
//! [`burn_stack::modules`], reached through [`crate::unified`].

pub mod cache;
pub mod delta_product;

/// Public re-exports for DeltaProduct.
pub mod prelude {
    pub use super::cache::{
        DeltaProductCache, DeltaProductCacheConfig, DeltaProductCaches, DeltaProductCachesConfig,
    };
    pub use super::delta_product::{DeltaProduct, DeltaProductConfig};
}
