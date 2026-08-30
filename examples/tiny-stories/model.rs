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
/// Sized to `burn-mamba`'s `tiny-stories` model (~40K parameters at
/// `d_model = 32`) so the two are a fair comparison on one corpus, one
/// tokenizer and one optimizer schedule — the block is the variable.
///
/// Why Gated DeltaNet 1 rather than the other three families: it is the one the
/// deployed language models use (Qwen3-Next), and it is the smallest family that
/// has everything text needs — a keyed, *erasing* write (plain DeltaNet's, which
/// is what keeps one fact per key instead of accumulating them) plus the scalar
/// forget gate that lets a head drop the whole state at a document boundary.
/// Swapping the family is one line here: `DeltaBlockConfig::` picks it, and
/// nothing else in the example knows which one it got.
pub fn model_config() -> DeltaVocabNetConfig {
    // d_model = 32 (intra/inter-layer expressivity, high impact on disk size)
    let d_model = 32;
    let block = GatedDeltaNet1Config::new(d_model)
        // nheads · head_k_dim = 4 · 8 = 32 = key_dim (expand_k = 1); expand_v = 2
        // ⇒ head_v_dim = 16, so each head's state is an 8×16 matrix — 512 state
        // scalars per block, which is what has to carry a whole story.
        .with_nheads(4)
        .with_head_k_dim(8)
        .with_expand_v(2.0)
        // The block's own gated output RMSNorm.
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
            DeltaNetworkShape::new(N_REAL_LAYERS)
                .with_n_virtual_layers(Some((N_VIRTUAL_LAYERS, Schedule::Cyclic)))
                .with_grad_horizon(GRAD_HORIZON)
                // Multi-Gate Residuals: `n_stream` pooled streams between layers
                // instead of one additive skip, at the cost of three vectors per
                // real layer. `per_virtual_layer: false` keeps one set per
                // *real* layer, reused across the virtual passes.
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
