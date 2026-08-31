//! Chunkwise WY with a custom, memory-efficient backward.
//!
//! The third arm of [`DeltaPath`](crate::delta::path::DeltaPath):
//!
//! ```text
//!   Recurrent            the definition, one token at a time
//!   Chunk { .. }         the chunkwise WY forward, backward via autodiff
//!   ChunkRecalculated    the same forward, backward written out by hand
//! ```
//!
//! The three compute the same function. What this one changes is *what reaches
//! the tape*: a single custom autodiff node retains the seven leaf inputs and
//! nothing else, and its backward replays the forward's three stages before
//! taking the gradient analytically. Under plain autodiff every one of those
//! stages leaves `[batch, nchunks, nheads, chunk_len, chunk_len]` tensors alive
//! — the `⌈log₂ L⌉` levels of the WY inverse alone are ~4 of them per level —
//! and at the shapes this crate trains at that *is* the training memory.
//!
//! This is the reference kernel's own split: `chunk_delta_rule_fwd` in
//! `fla/ops/delta_rule/chunk.py` saves `(q, k, v, β, A, h₀)` and
//! `chunk_delta_rule_bwd` recomputes `w`, `u` and the whole per-chunk state
//! stream before it differentiates anything.
//!
//! ## Layout
//!
//! - `chunk_recalculated` — the entry point and the [`DeltaChunkBackendExt`]
//!   trait whose default body is the plain forward.
//! - `forward` — the forward's three stages on backend primitives, shared by
//!   the default body and the backward's recompute.
//! - [`backward`] — the registered autodiff node.
//! - [`combined_backward`] — the recompute-based gradient math.
//!
//! Nothing here reaches below Burn's portable tensor ops.

#[cfg(feature = "autodiff")]
pub mod backward;
mod chunk_recalculated;
pub mod combined_backward;
pub(crate) mod forward;
pub(crate) mod prim;

pub use chunk_recalculated::DeltaChunkBackendExt;

#[cfg(feature = "autodiff")]
pub use chunk_recalculated::DeltaChunkAutodiffBackendExt;

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
