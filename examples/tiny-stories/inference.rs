//! Sampling from the trained character LM.
//!
//! [`infer`] loads the checkpoint and prints a few stories at different
//! temperatures. The sampler itself is
//! [`burn_stack`'s](burn_stack::examples::tiny_stories::sample::generate),
//! shared with `burn-mamba` — [`DeltaVocabNet`] is the block-generic
//! `VocabNetwork` at [`DeltaBlock`](burn_deltanet::prelude::DeltaBlock), so the
//! generic sampler applies to it directly; all this module adds is the
//! [`path`](crate::training::path).
//!
//! It shows the library's two execution modes back to back: the prompt is
//! consumed by one chunkwise `forward` (prefill), and every generated character
//! then costs one `step` against that same cache — O(state) per token, with no
//! growing KV cache.

use crate::AppArgs;
use crate::dataset::STORY_SEPARATOR;
use burn::prelude::*;
use burn_deltanet::prelude::{DeltaVocabNet, DeltaVocabNetConfig};

/// Temperatures sampled by [`infer`], from near-greedy to loose.
const TEMPERATURES: &[f64] = &[0.5, 0.8, 1.0];

/// Characters generated per sample by [`infer`].
const SAMPLE_CHARS: usize = 800;

/// Load the trained LM and print one story per temperature, plus one
/// continuation of a fixed prompt.
pub fn infer(model_config: DeltaVocabNetConfig, infer_device: Device, app_args: &AppArgs) {
    let model: DeltaVocabNet = app_args
        .load_model(&model_config, &infer_device)
        .expect("no trained model in the artifacts directory; run with --training first");

    let out_dir = app_args.artifacts_path.join("inference");
    std::fs::create_dir_all(&out_dir).expect("failed to create the inference directory");

    for (i, &temperature) in TEMPERATURES.iter().enumerate() {
        // The document boundary is the model's "start of story" prompt.
        let text = generate(
            &model,
            &infer_device,
            STORY_SEPARATOR,
            SAMPLE_CHARS,
            temperature,
            i as u64,
        );
        println!("\n--- unprompted, temperature {temperature} ---\n{text}");
        let path = out_dir.join(format!("sample-t{temperature}.txt"));
        std::fs::write(&path, &text).expect("failed to write the sample");
    }

    let prompt = "once upon a time, there was a little girl named lily. she";
    let text = generate(
        &model,
        &infer_device,
        prompt,
        SAMPLE_CHARS,
        0.8,
        TEMPERATURES.len() as u64,
    );
    println!("\n--- prompted, temperature 0.8 ---\n{prompt}{text}");
    let path = out_dir.join("sample-prompted.txt");
    std::fs::write(&path, format!("{prompt}{text}")).expect("failed to write the sample");

    println!("\nsaved {} samples to {out_dir:?}", TEMPERATURES.len() + 1);
}

/// Continue `prompt` with `n_chars` sampled characters, at this example's
/// [`path`](crate::training::path).
pub fn generate(
    model: &DeltaVocabNet,
    device: &Device,
    prompt: &str,
    n_chars: usize,
    temperature: f64,
    seed: u64,
) -> String {
    burn_stack::examples::tiny_stories::sample::generate(
        model,
        device,
        crate::training::path(),
        prompt,
        n_chars,
        temperature,
        seed,
    )
}
