//! # Inference caches
//!
//! What must survive between calls during autoregressive decoding — and, just
//! as importantly, between two `forward`s over consecutive chunks of one
//! sequence. Two pieces per layer:
//!
//! 1. **The convolution window** — the last `conv_kernel` inputs to the
//!    depthwise short convolution, so a decode step can apply the causal filter
//!    without re-reading the past. Absent when the block has no convolution.
//! 2. **The recurrent state `S ∈ ℝ^{head_k_dim × head_v_dim}`** per head — the
//!    associative memory the delta rule writes into. It is fixed-size: the
//!    whole point of the family is that decoding costs `O(head_k_dim ·
//!    head_v_dim)` per token however long the context, where a Transformer's
//!    KV cache grows with it.
//!
//! **One cache type serves all four families.** They differ in what they
//! *project* — a scalar `β`, a per-head forget gate, per-channel erase/write
//! gates, `u` Householder factors — but every one of those is computed from the
//! current token and consumed within the step. What crosses a step boundary is
//! the same convolution window and the same associative memory, so a shared
//! [`DeltaCache`] is not a simplification of four types but a statement about
//! the recurrence. It is also what lets [`DeltaBlock`] dispatch a family at
//! runtime without a cache tag.
//!
//! Both fields are plain (non-parameter) tensors, so the [`CacheStack`] impl in
//! [`crate::unified`] converts them to and from the inner backend by hand.
//!
//! [`CacheStack`]: burn_stack::modules::CacheStack
//! [`DeltaBlock`]: crate::unified::block::DeltaBlock

use burn::module::Module;
use burn::prelude::*;
use burn_stack::modules::sanity as san;

// ---------------------------------------------------------------------------
// DeltaCache  (one layer)
// ---------------------------------------------------------------------------

/// The state carried between calls for a **single** delta-rule layer, whichever
/// family the block belongs to.
#[derive(Module, Debug)]
pub struct DeltaCache {
    /// Short-convolution rolling window, `[batch, conv_dim, conv_kernel]`.
    /// `None` iff the block runs without a convolution.
    pub conv_bwc: Option<Tensor<3>>,

    /// The delta rule's associative memory `S`,
    /// `[batch, nheads, head_k_dim, head_v_dim]`.
    pub state_bhkv: Tensor<4>,
}

impl DeltaCache {
    /// Rebuild a cache from its two tensors.
    ///
    /// Used by the [`CacheStack`](burn_stack::modules::CacheStack) impl, which
    /// has to convert each field by hand.
    pub fn from_parts(conv_bwc: Option<Tensor<3>>, state_bhkv: Tensor<4>) -> Self {
        Self {
            conv_bwc,
            state_bhkv,
        }
    }

    /// Run the [`NaN`/`Inf` guards](burn_stack::modules::misc::sanity) on every
    /// cached tensor.
    pub fn sanity(&self) {
        if let Some(conv_bwc) = &self.conv_bwc {
            san(conv_bwc);
        }
        san(&self.state_bhkv);
    }
}

/// Configuration / factory for a single [`DeltaCache`].
#[derive(Config, Debug)]
pub struct DeltaCacheConfig {
    /// Batch size.
    pub batch: usize,
    /// Number of heads.
    pub nheads: usize,
    /// Query/key width per head.
    pub head_k_dim: usize,
    /// Value width per head.
    pub head_v_dim: usize,
    /// Channels entering the short convolution.
    pub conv_dim: usize,
    /// Convolution window length; `0` means the block has no convolution.
    pub conv_kernel: usize,
}

impl DeltaCacheConfig {
    /// Allocate zero-initialised cache tensors on `device`.
    ///
    /// Zero is the correct empty state on both counts: an all-zero convolution
    /// window is "no previous tokens", and `S = 0` is an associative memory
    /// that returns zero for every key.
    pub fn init(&self, device: &Device) -> DeltaCache {
        DeltaCache {
            conv_bwc: (self.conv_kernel > 0).then(|| {
                Tensor::zeros(
                    Shape::new([self.batch, self.conv_dim, self.conv_kernel]),
                    device,
                )
            }),
            state_bhkv: Tensor::zeros(
                Shape::new([self.batch, self.nheads, self.head_k_dim, self.head_v_dim]),
                device,
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// DeltaCaches  (one entry per virtual layer)
// ---------------------------------------------------------------------------

/// Per-layer caches for a complete network — one slot per *virtual* layer,
/// which may exceed the number of real weight sets.
#[derive(Module, Debug)]
pub struct DeltaCaches {
    /// Per-layer caches.
    pub caches: Vec<DeltaCache>,
}

impl DeltaCaches {
    /// Number of per-layer caches.
    pub fn caches_len(&self) -> usize {
        self.caches.len()
    }

    /// Wrap a vector of per-layer caches.
    pub fn from_vec(vec: Vec<DeltaCache>) -> Self {
        Self { caches: vec }
    }

    /// Wrap each per-layer cache in `Some` so the layer loop can `take` it
    /// without cloning (Burn tensors are reference-counted).
    pub fn into_options(self) -> Vec<Option<DeltaCache>> {
        self.caches.into_iter().map(Some).collect()
    }

    /// Inverse of [`Self::into_options`].
    pub fn from_options(options: Vec<Option<DeltaCache>>) -> Self {
        Self::from_vec(options.into_iter().map(Option::unwrap).collect())
    }
}

/// Configuration / factory for [`DeltaCaches`].
#[derive(Config, Debug)]
pub struct DeltaCachesConfig {
    /// Number of cache slots (= virtual layers).
    pub n_caches: usize,
    /// Shape of each individual cache.
    pub cache: DeltaCacheConfig,
}

impl DeltaCachesConfig {
    /// Allocate all cache tensors (zero-initialised) on `device`.
    pub fn init(&self, device: &Device) -> DeltaCaches {
        DeltaCaches {
            caches: (0..self.n_caches)
                .map(|_| self.cache.clone().init(device))
                .collect(),
        }
    }
}
