//! The model configuration for the `mnist-class` example — a small GDN-2
//! classifier (2 real layers stretched to 16 virtual layers); see
//! [`model_config`].

use burn_deltanet::prelude::{
    DeltaBlockConfig, DeltaLatentNetConfig, DeltaLatentShape, DeltaNetworkShape,
    GatedDeltaNet2Config,
};
use burn_stack::utils::{ClassLatent, GradHorizon, Schedule};

/// Depth of the (virtual) layer stack.
const N_VIRTUAL_LAYERS: usize = 16;

/// Number of real weight sets the virtual stack cycles over.
const N_REAL_LAYERS: usize = 2;

/// Back-propagate only the last `K` applications of **each real layer** —
/// [`GradHorizon::Depth`], counted per weight set — with everything below
/// running on the inner backend; `None` tracks the whole stack.
///
/// This stack — 16 virtual layers over 2 real weight sets — is TRM/HRM-style
/// deep recursion, and its activations dominate the vram figure below, so a
/// small `K` is what lets the stack grow deeper. `Depth(2)` tracks 4 of the 16
/// virtual layers, two per real layer, so neither weight set goes untrained.
const GRAD_HORIZON: Option<GradHorizon> = Some(GradHorizon::Depth(2));

/// Stack-level class latents prepended to every image's pixel sequence:
/// learnable `[CLS]`-style registers (width `d_model`) that let the model settle
/// into a trained initial state before the first pixel arrives. They lengthen
/// the output sequence, which the readout accounts for — see
/// [`OUTPUT_SEQUENCE_EXTRA`].
pub const N_CLASS_LATENTS: usize = 0;

/// How much longer the model's output is than its pixel input, in timesteps.
/// The class latents all sit at the **front** (`Start`), so the classification
/// readout is still the sequence's last position — just not index `784 - 1`.
pub const OUTPUT_SEQUENCE_EXTRA: usize = N_CLASS_LATENTS;

/// This model configuration uses ~36K params (~146KB on disk in FP32).
///
/// The task is the one the `register-*` examples are *not*: a real dataset, no
/// hand-built solution, and the block used as a sequence mixer rather than as a
/// register file. It is the smallest GDN-2 configuration that trains to a
/// useful accuracy on the 784-pixel sequence, so it doubles as the crate's
/// end-to-end smoke test for the whole stack (fused projection, short conv,
/// chunkwise WY path, virtual layers, truncated BPTT).
///
/// Why GDN-2 rather than DeltaNet or Gated DeltaNet 1: reading one pixel at a
/// time, most of the sequence is background that the state should hold through,
/// while strokes should overwrite what a head is tracking. That is exactly the
/// erase/write split — `b` decides what to drop, `w` what to commit — and the
/// per-key-channel `α` lets one head keep some rows for hundreds of steps while
/// others lapse within a row of pixels.
pub fn model_config() -> DeltaLatentNetConfig {
    // d_model = 32 (intra/inter-layer expressivity, high impact on disk size)
    let d_model = 32;
    let block = GatedDeltaNet2Config::new(d_model)
        // nheads · head_k_dim = 4 · 8 = 32 = key_dim; expand_v = 1 ⇒
        // head_v_dim = 8, so each head's state is an 8×8 matrix.
        .with_nheads(4)
        .with_head_k_dim(8)
        .with_expand_v(1.0)
        // The `Δ` / output-gate bottleneck rank. `0` resolves to head_v_dim (8),
        // which at this width is already the whole thing — spell it out.
        .with_bottleneck(8)
        // The output gate: the block's own `RmsNormGated` readout.
        .with_use_gate(true)
        // A write replaces; nothing here reflects.
        .with_allow_neg_eigval(false)
        // A 4-wide causal depthwise conv over [q|k|v]: a pixel's immediate
        // neighbours along the scan line, which the recurrence would otherwise
        // have to rebuild.
        .with_use_short_conv(true)
        .with_conv_kernel(4)
        .with_has_proj_bias(true);

    // input  [batch_size, sequence_len = HEIGHT * WIDTH, input_size = 1]
    // output [batch_size, HEIGHT * WIDTH + OUTPUT_SEQUENCE_EXTRA, output_size = 10]
    // (later narrowed to the last timestep for the 10-bin classification)
    DeltaLatentNetConfig::new(
        DeltaLatentShape::new(
            1,
            10,
            // two real layers, virtually cycled (2×2×2×2) to 16 for more
            // expressivity at no parameter cost
            DeltaNetworkShape::new(N_REAL_LAYERS)
                .with_n_virtual_layers(Some((N_VIRTUAL_LAYERS, Schedule::Cyclic)))
                .with_grad_horizon(GRAD_HORIZON)
                .with_class_latents(vec![ClassLatent::Start; N_CLASS_LATENTS]),
        )
        .with_final_norm(false),
        DeltaBlockConfig::GatedDeltaNet2(block),
    )
}
// notes:
// - this small model requires quite a lot of vram because the whole 28*28
//   sequence for each image is processed in parallel, and a high amount of
//   virtual layers are used.
// - this should benefit from a bidi encoder since a single output is predicted
//   after the whole image is read.
