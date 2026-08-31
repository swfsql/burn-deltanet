//! The model configuration for the `tiny-stories` example — a small
//! character-level Gated DeltaNet language model (2 real layers, cycled to an
//! 8-deep virtual stack); see [`model_config`].

use crate::dataset::VOCAB_SIZE;
use burn_deltanet::prelude::{
    DeltaBlockConfig, DeltaNetworkShape, DeltaVocabNetConfig, DeltaVocabShape, GatedDeltaNet1Config,
    GatedMlpConfig, InitPolicy, ResidualsConfig,
};
use burn_stack::utils::{GradHorizon, Schedule};

/// Depth of the (virtual) layer stack: the 2 real weight sets applied four
/// times. Virtual depth is **free in parameters**, so it is the cheapest
/// capacity this budget can buy.
const N_VIRTUAL_LAYERS: usize = 8;

/// Number of real weight sets the virtual stack cycles over.
const N_REAL_LAYERS: usize = 2;

/// Back-propagate only the top `Depth(K)` applications of each real layer,
/// everything below running on the inner backend; `None` tracks the whole stack.
///
/// `None` here, unlike `mnist-class`: a language model is scored at *every*
/// position, so leaving most applications of a *shared* weight undifferentiated
/// biases every one of those readouts — unlike a task that reads out once, at
/// the end of the sequence. (`burn-mamba`'s `tiny-stories` README measures that
/// on the same corpus.)
const GRAD_HORIZON: Option<GradHorizon> = None;

/// The character-level LM: two Gated DeltaNet blocks between a **tied**
/// 48-character embedding and its transpose, each followed by the SwiGLU MLP the
/// reference architecture puts there.
///
/// 33,748 parameters — `burn-mamba`'s `tiny-stories` model (39,632 at the same
/// `d_model = 32`) to within 15%, so the two are a fair comparison on one
/// corpus, one tokenizer and one optimizer schedule, with the block as the
/// variable. The budget is not matched exactly because the block's sizing is
/// fixed by the reference's parameter allocation (below), not by the target.
///
/// Why Gated DeltaNet 1 rather than the other three families: it is the one the
/// deployed language models use (Qwen3-Next), and it is the smallest family that
/// has everything text needs — a keyed, *erasing* write (plain DeltaNet's, which
/// is what keeps one fact per key instead of accumulating them) plus the scalar
/// forget gate that lets a head drop the whole state at a document boundary.
/// Swapping the family is one line here: `DeltaBlockConfig::` picks it, and
/// nothing else in the example knows which one it got.
///
/// **What follows the reference and what does not.** The layer is the
/// reference's, down to the parameter allocation (`~6·d_model²`, below) and the
/// global init, inside Llama's macro architecture the paper states it uses:
/// Pre-LN mixer, Pre-LN SwiGLU MLP, a final norm before the head. The *stack*
/// is not: cycling 2 weight sets over 8 virtual layers, and mixing the layers
/// through [`ResidualsConfig::MultiGate`] instead of one additive skip, are
/// depth and capacity bought at (almost) no parameter cost, which the
/// reference, training 21 distinct layers, never needs. Both are measured wins
/// on this corpus at this budget — `burn-mamba`'s README has the numbers — and
/// both are stated explicitly below rather than left to a default.
pub fn model_config() -> DeltaVocabNetConfig {
    // d_model = 32 (intra/inter-layer expressivity, high impact on disk size)
    let d_model = 32;
    let block = GatedDeltaNet1Config::new(d_model)
        // The reference's parameter allocation for a gated block: with the
        // output gate on, `nheads · head_k_dim = 0.75 · d_model` and
        // `expand_v = 2`, so `q`/`k` cost 0.75·d² each and `v`/`gate`/`o` 1.5·d²
        // each — 6·d² in total, a Transformer layer's budget (the NOTE in
        // `fla/layers/gated_deltanet.py`, whose own defaults are 6 heads of 256
        // at `hidden_size = 2048`).
        //
        // Here: 3 · 8 = 24 = 0.75 · 32 = key_dim; expand_v = 2 ⇒ head_v_dim = 16
        // and value_dim = 48 = 1.5 · 32. Each head's state is an 8×16 matrix —
        // 384 state scalars per block, which is what has to carry a whole story.
        .with_nheads(3)
        .with_head_k_dim(8)
        .with_expand_v(2.0)
        // The block's own gated output RMSNorm (`use_gate`, on in the
        // reference — and what the 0.75 above pays for).
        .with_use_gate(true)
        // A write replaces; nothing here reflects.
        .with_allow_neg_eigval(false)
        // A 4-wide causal depthwise conv over [q|k|v]: character n-grams, which
        // the recurrence would otherwise have to rebuild every token.
        .with_use_short_conv(true)
        .with_conv_kernel(4)
        // The reference language models carry no projection bias (the global
        // init below zeroes biases anyway).
        .with_has_proj_bias(false);

    DeltaVocabNetConfig::new(
        DeltaVocabShape::new(
            // the 48 case-folded characters the corpus actually contains
            VOCAB_SIZE,
            // Every knob of `burn_stack::modules::NetworkShape` is stated here,
            // defaults included, so this one block describes the whole stack.
            DeltaNetworkShape::new(N_REAL_LAYERS)
                .with_n_virtual_layers(Some((N_VIRTUAL_LAYERS, Schedule::Cyclic)))
                .with_grad_horizon(GRAD_HORIZON)
                // No stack-level class latents: every position is a character.
                .with_class_latents(Vec::new())
                .with_ignore_first_residual(false)
                .with_ignore_last_residual(false)
                // Multi-Gate Residuals: `n_stream` pooled streams between layers
                // instead of the reference's one additive skip, at the cost of
                // three vectors per real layer. `per_virtual_layer: false` keeps
                // one set per *real* layer, reused across the virtual passes.
                .with_residuals(ResidualsConfig::MultiGate {
                    n_stream: 4,
                    // Start every stream on an equal, unbiased gate and let
                    // training break the symmetry.
                    init_bias: 0.0,
                    init_bias_step: 0.0,
                    per_virtual_layer: false,
                })
                // The channel mixer of the Llama macro architecture, which every
                // reference delta-rule LM keeps verbatim. `from_hidden_ratio`'s
                // `ratio = 4` is the reference figure; its 256-alignment is not
                // meaningful at `d_model = 32` (it would make the MLP eight
                // times the width the rule asks for), so the rounding drops to
                // 16 and the realised inner width is 96.
                .with_mlp(Some(
                    GatedMlpConfig::from_hidden_ratio(d_model, 4).with_multiple_of(16),
                ))
                // The reference LM init: every 2-D weight from `N(0, 0.02²)`,
                // biases zeroed. It is also what keeps the **tied** head sane at
                // this width — Burn initialises an `Embedding` from `N(0, 1)`,
                // which at `d_model = 32` starts the logits at variance ~32 and
                // costs the opening of the LR schedule.
                .with_init(Some(InitPolicy::new())),
        )
        // keep the softmax exactly `VOCAB_SIZE`-way: no padded class can ever be
        // sampled, so every logit is a character the decoder understands
        .with_pad_vocab_size_multiple(1)
        // true ⇒ the LM head is the (transposed) embedding: one table for "which
        // character is this" and "which character comes next"
        .with_missing_lm_head(true),
        DeltaBlockConfig::GatedDeltaNet1(block),
    )
}
