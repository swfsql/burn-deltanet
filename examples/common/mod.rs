//! Shared infrastructure for the burn-deltanet examples.
//!
//! Each concrete example (`register-majority`, `register-carousel`) wires
//! together the pieces here: runtime [`device`] selection, CLI + artifact
//! handling ([`cli`]), the [`model`] factory seam, and the generic [`training`]
//! config.
//!
//! With the Dispatch-based architecture, no module here carries a backend type
//! generic — `Tensor`/`Device`/`Module` are pinned to the global `Dispatch`
//! backend, and the device chooses the concrete runtime backend.

#![allow(dead_code)]

/// CLI parsing, artifact directory management, and the train/infer flow.
pub mod cli;
/// Runtime [`Device`] selection + optional dtype configuration.
pub mod device;
/// The [`ModelConfigExt`](model::ModelConfigExt) seam bridging example configs
/// to the library's unified network types.
pub mod model;
/// The shared training configuration.
pub mod training;
