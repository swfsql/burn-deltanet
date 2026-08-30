//! # Gated DeltaNet
//!
//! The delta rule plus a Mamba-2-style scalar forget gate: targeted overwrites
//! *and* indiscriminate decay, which the paper shows are complementary rather
//! than redundant. See
//! [`gated_deltanet_1`] for the block and
//! the gate's parameterisation, and [`crate::delta`] for the recurrence.
//!
//! - [`gated_deltanet_1`] — the block and
//!   its config.
//!
//! The convolution window + recurrent state carried between calls is the
//! shared [`DeltaCache`](crate::common::cache::DeltaCache); the residual layer,
//! the layer stack, the language model and the bidirectional wrappers are the
//! family-generic types in [`burn_stack::modules`], reached through
//! [`crate::unified`].

pub mod gated_deltanet_1;

/// Public re-exports for Gated DeltaNet.
pub mod prelude {
    pub use super::gated_deltanet_1::{GatedDeltaNet1, GatedDeltaNet1Config};
}
