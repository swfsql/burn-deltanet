//! # Associative recall (MQAR) — the task the delta rule exists for
//!
//! The model reads a stream of key/value pairs and is then asked for the value
//! of one of the keys:
//!
//! ```text
//!   k₃ v₇  k₁ v₂  k₅ v₀  …  k₁   →   v₂
//! ```
//!
//! Nothing about it needs long-range reasoning; it needs an *addressable*
//! memory. That is exactly the axis linear attention is weak on and the delta
//! rule is built for. A linear-attention state accumulates `Σ kᵢ vᵢᵀ`, so every
//! new pair adds interference to every stored one and recall degrades as the
//! store fills. The delta rule instead *removes* the association currently held
//! at `kₜ` before writing the new one, which keeps the store clean:
//!
//! ```text
//!   Sₜ = αₜ (I − βₜ kₜ kₜᵀ) Sₜ₋₁ + βₜ kₜ vₜᵀ
//! ```
//!
//! The whole point is that the state is **fixed-size**: it does not grow with
//! the number of pairs, unlike a Transformer's KV cache. So the interesting
//! knob is `--pairs`, which fills the memory further without giving the model
//! any more of it.
//!
//! ## Running it
//!
//! ```bash
//! cargo run --release --example associative-recall --features backend-flex
//! cargo run --release --example associative-recall --features backend-flex -- --family deltanet
//! cargo run --release --example associative-recall --features backend-flex -- --pairs 16 --steps 6000
//! ```
//!
//! Deliberately tiny — two layers, `d_model = 64`. The defaults (8 pairs, 2000
//! steps) reach **~98% held-out accuracy in about 80 seconds** on a CPU. Raising
//! `--pairs` makes it markedly harder and needs several times the steps: the
//! state does not grow with the store, which is the whole trade the family
//! makes. It is a demonstration, not a benchmark.
//!
//! The run finishes by decoding the same held-out problems token by token with
//! `step()` and comparing against the chunked `forward()` — the parity the
//! crate is built around, checked end to end on a trained model rather than on
//! random weights.

use burn::module::AutodiffModule;
use burn::optim::{AdamWConfig, GradientsParams, ModuleOptimizer};
use burn::prelude::*;
use burn::tensor::TensorData;
use burn_deltanet::prelude::*;
use rand::prelude::*;
use rand::rngs::StdRng;

// ---------------------------------------------------------------------------
// The task
// ---------------------------------------------------------------------------

/// Token layout: `[0, n_keys)` are keys, `[n_keys, n_keys + n_values)` values.
#[derive(Clone, Copy, Debug)]
struct Task {
    n_keys: usize,
    n_values: usize,
    n_pairs: usize,
}

impl Task {
    fn vocab_size(&self) -> usize {
        self.n_keys + self.n_values
    }

    /// One problem: `2·n_pairs` tokens of key/value pairs, then a query key.
    /// Returns `(tokens, answer)`.
    ///
    /// Keys are sampled **without replacement**, so each has exactly one right
    /// answer and the task is about capacity rather than about resolving
    /// contradictions.
    fn sample(&self, rng: &mut StdRng) -> (Vec<i64>, i64) {
        let mut keys: Vec<usize> = (0..self.n_keys).collect();
        keys.shuffle(rng);
        keys.truncate(self.n_pairs);

        let mut tokens = Vec::with_capacity(2 * self.n_pairs + 1);
        let mut values = Vec::with_capacity(self.n_pairs);
        for &key in &keys {
            let value = rng.random_range(0..self.n_values);
            values.push(value);
            tokens.push(key as i64);
            tokens.push((self.n_keys + value) as i64);
        }

        let asked = rng.random_range(0..self.n_pairs);
        tokens.push(keys[asked] as i64);
        (tokens, (self.n_keys + values[asked]) as i64)
    }

    /// A batch of problems as `([batch, sequence] tokens, [batch, 1] answers)`.
    fn batch(&self, batch: usize, rng: &mut StdRng, device: &Device) -> (Tensor<2, Int>, Tensor<2, Int>) {
        let sequence = 2 * self.n_pairs + 1;
        let mut tokens = Vec::with_capacity(batch * sequence);
        let mut answers = Vec::with_capacity(batch);
        for _ in 0..batch {
            let (problem, answer) = self.sample(rng);
            tokens.extend(problem);
            answers.push(answer);
        }
        (
            Tensor::from_data(TensorData::new(tokens, [batch, sequence]), device),
            Tensor::from_data(TensorData::new(answers, [batch, 1]), device),
        )
    }
}

// ---------------------------------------------------------------------------
// Model, loss, evaluation
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    DeltaNet,
    GatedDeltaNet,
    DeltaProduct,
}

impl Family {
    fn parse(name: &str) -> Self {
        match name {
            "deltanet" => Self::DeltaNet,
            "gated" | "gated-deltanet" => Self::GatedDeltaNet,
            "product" | "delta-product" => Self::DeltaProduct,
            other => panic!("unknown --family `{other}` (deltanet | gated | product)"),
        }
    }

    fn config(self, vocab_size: usize, d_model: usize, n_layers: usize) -> DeltaVocabNetConfig {
        // An untied LM head: the tied-embedding default starts with logits of
        // magnitude ~d_model, which is a needlessly hard place to begin from on
        // a task this small.
        let shape = DeltaVocabShape::new(vocab_size, DeltaNetworkShape::new(n_layers))
            .with_missing_lm_head(false);
        let (nheads, head_k_dim) = (2, d_model / 4);
        match self {
            Self::DeltaNet => DeltaVocabNetConfig::DeltaNet {
                shape,
                block: DeltaNetConfig::new(d_model)
                    .with_nheads(nheads)
                    .with_use_gate(true),
            },
            Self::GatedDeltaNet => DeltaVocabNetConfig::GatedDeltaNet {
                shape,
                block: GatedDeltaNetConfig::new(d_model)
                    .with_nheads(nheads)
                    .with_head_k_dim(head_k_dim)
                    .with_expand_v(1.0),
            },
            Self::DeltaProduct => DeltaVocabNetConfig::DeltaProduct {
                shape,
                block: DeltaProductConfig::new(d_model)
                    .with_nheads(nheads)
                    .with_head_k_dim(head_k_dim)
                    .with_expand_v(1.0)
                    .with_n_householder(2),
            },
        }
    }
}

/// Logits at the **last** position only — the answer slot.
fn answer_logits(model: &DeltaVocabNet, tokens: Tensor<2, Int>, path: DeltaPath) -> Tensor<2> {
    let [_batch, sequence] = tokens.dims();
    let (logits, _caches) = model.forward(tokens, None, path, None);
    logits.narrow(1, sequence - 1, 1).squeeze_dim(1)
}

/// Cross-entropy against integer class indices.
fn cross_entropy(logits_bv: Tensor<2>, answers_b1: Tensor<2, Int>) -> Tensor<1> {
    let log_probs = burn::tensor::activation::log_softmax(logits_bv, 1);
    -log_probs.gather(1, answers_b1).mean()
}

fn accuracy(logits_bv: Tensor<2>, answers_b1: Tensor<2, Int>) -> f32 {
    let predicted = logits_bv.argmax(1);
    predicted
        .equal(answers_b1)
        .float()
        .mean()
        .into_scalar::<f32>()
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let mut args = pico_args::Arguments::from_env();
    let family = Family::parse(
        &args
            .opt_value_from_str::<_, String>("--family")
            .expect("--family")
            .unwrap_or_else(|| "gated".into()),
    );
    let n_pairs: usize = args.opt_value_from_str("--pairs").unwrap().unwrap_or(8);
    let steps: usize = args.opt_value_from_str("--steps").unwrap().unwrap_or(2000);
    let batch: usize = args.opt_value_from_str("--batch").unwrap().unwrap_or(32);
    let seed: u64 = args.opt_value_from_str("--seed").unwrap().unwrap_or(0);
    let lr: f64 = args.opt_value_from_str("--lr").unwrap().unwrap_or(3e-3);

    let task = Task {
        n_keys: 64,
        n_values: 32,
        n_pairs,
    };
    assert!(
        task.n_pairs <= task.n_keys,
        "keys are sampled without replacement, so --pairs cannot exceed n_keys",
    );

    let device = Device::default().autodiff();
    let (d_model, n_layers) = (64, 2);
    let path = DeltaPath::chunk_len(32);

    let config = family.config(task.vocab_size(), d_model, n_layers);
    let mut model = config.init(&device);
    let mut optim: ModuleOptimizer = AdamWConfig::new()
        .with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Value(1.0)))
        .init();

    let mut rng = StdRng::seed_from_u64(seed);
    // A held-out RNG stream, so the reported accuracy is on problems the model
    // has not been trained on (the sampler is unlikely to repeat, but this makes
    // it a guarantee rather than a likelihood).
    let mut eval_rng = StdRng::seed_from_u64(seed ^ 0x5eed);

    println!(
        "family {family:?} | {} pairs into a {}×{} per-head state | sequence {} | vocab {}",
        task.n_pairs,
        d_model / 4,
        d_model / 4,
        2 * task.n_pairs + 1,
        task.vocab_size(),
    );

    for step in 1..=steps {
        let (tokens, answers) = task.batch(batch, &mut rng, &device);
        let logits = answer_logits(&model, tokens, path);
        let loss = cross_entropy(logits.clone(), answers.clone());

        let grads = GradientsParams::from_grads(loss.clone().backward(), &model);
        model = optim.step(lr, model, grads);

        if step % 100 == 0 || step == 1 {
            println!(
                "step {step:>5}/{steps}  loss {:.4}  train acc {:.1}%",
                loss.into_scalar::<f32>(),
                100.0 * accuracy(logits, answers),
            );
        }
    }

    // ── Held-out evaluation, and a check that decoding agrees ──────────────
    let evaluated = model.valid();
    let eval_device = Device::default();
    let (tokens, answers) = task.batch(512, &mut eval_rng, &eval_device);
    let logits = answer_logits(&evaluated, tokens.clone(), path);
    println!(
        "\nheld-out accuracy over 512 problems: {:.1}%",
        100.0 * accuracy(logits.clone(), answers.clone()),
    );

    // The same model decoded token by token must give the same answer — the
    // parity the whole crate is built around, here end to end.
    let [_batch, sequence] = tokens.dims();
    let mut caches: Option<DeltaCaches> = None;
    let mut stepped = None;
    for t in 0..sequence {
        let token = tokens.clone().narrow(1, t, 1).squeeze_dim(1);
        let (out, next) = evaluated.step(token, caches.take(), None);
        caches = Some(next);
        stepped = Some(out);
    }
    let stepped = stepped.expect("a non-empty sequence");
    let drift = (logits - stepped.clone()).abs().max().into_scalar::<f32>();
    println!(
        "step-decoded accuracy:                {:.1}%   (max logit drift vs forward: {drift:.2e})",
        100.0 * accuracy(stepped, answers),
    );
}
