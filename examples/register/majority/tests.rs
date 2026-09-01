//! The two claims this example rests on, measured.
//!
//! 1. A **hand-built** DeltaNet block solves the task exactly — no fitting,
//!    every weight written down in closed form from the recurrence.
//! 2. The **same recurrence with the erase switched off** cannot, for *any*
//!    decay. That is what makes the task a delta-rule task rather than a plain
//!    linear-attention one.
//!
//! The second claim is measured on a real block, not a simulation: GDN-2 is the
//! family in this crate whose erase and write gates are *independent*
//! ([`crate::common`] aside, see `burn_deltanet::gated_deltanet_2`), so forcing its erase
//! gate to zero leaves exactly `Sₜ = diag(α) Sₜ₋₁ + kₜ (w ⊙ vₜ)ᵀ` — gated
//! linear attention, with `α` free to be swept.
//!
//! The construction is the one derived in [`crate::model`]: `k` picks the
//! register, `β` is the write-enable, `v` carries the bit next to a constant
//! reference axis, and `q = (1,1,1)` reads the whole file so the block's output
//! direction *is* the majority.

use crate::common::model::ModelConfigExt;
use crate::dataset::{
    Family, IGNORE, NUM_CLASSES, NUM_EVAL, NUM_REGISTERS, NUM_SYMBOLS, QUERY, RegisterDataset,
    SEQ_LENGTH, labels, one_hot,
};
use crate::model::{D_MODEL, HEAD_V_DIM};
use crate::training::path;
use burn::data::dataset::Dataset;
use burn::module::Param;
use burn::prelude::*;
use burn_deltanet::prelude::*;

// ---------------------------------------------------------------------------
// the construction's constants
// ---------------------------------------------------------------------------

/// `v`'s bit channel, `±V`. Bounded by silu's floor (`min silu = −0.2785`),
/// which is what lets the two bits be exactly symmetric.
const V_MAG: f64 = 0.25;
/// `v`'s reference channel: constant and positive, so the block's per-head
/// RMSNorm sees a *direction* rather than a lone scalar it would flatten to a
/// hard sign.
const R_MAG: f64 = 0.25;
/// The saturating logit: `σ(±BIG)` is 1 / 0 to well under f32 resolution.
const BIG: f64 = 20.0;
/// Pre-activation magnitude for `q`/`k` when the block applies silu to them
/// (every family but DeltaNet does). Only the *direction* survives the L2 norm.
const QK_MAG: f64 = 4.0;
/// Class-logit gain on the majority axis.
const OUT_GAIN: f64 = 6.0;

/// The three register codes: an equilateral triangle in the embedding's first
/// two channels, so `k` — an affine functional of them — can reach each of the
/// three unit axes exactly (`kᵢ = ⅓ + ⅔·⟨pᵢ, code⟩`).
const REGISTER_CODE: [[f64; 2]; NUM_REGISTERS] = [
    [1.0, 0.0],
    [-0.5, 0.866_025_403_784_438_6],
    [-0.5, -0.866_025_403_784_438_6],
];

// ---------------------------------------------------------------------------
// scalar helpers
// ---------------------------------------------------------------------------

fn silu(t: f64) -> f64 {
    t / (1.0 + (-t).exp())
}

/// Inverse of `silu` on the branch containing 0 (`t > -1.2785`).
fn silu_inv(v: f64) -> f64 {
    assert!(v > -0.2784, "silu bottoms out at -0.2785, cannot reach {v}");
    let (mut lo, mut hi) = (-1.2785f64, v.max(0.0) + 1.0);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if silu(mid) < v { lo = mid } else { hi = mid }
    }
    0.5 * (lo + hi)
}

/// Inverse of the logistic sigmoid.
fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

fn t1<const D: usize>(v: &[f64], shape: [usize; D], device: &Device) -> Tensor<D> {
    let f: Vec<f32> = v.iter().map(|&x| x as f32).collect();
    Tensor::<1>::from_floats(f.as_slice(), device).reshape(shape)
}

/// Solve the `n × n` system `m·x = rhs` by Gaussian elimination with partial
/// pivoting. `m` is row-major.
fn solve(mut m: Vec<f64>, mut rhs: Vec<f64>, n: usize) -> Vec<f64> {
    for col in 0..n {
        let piv = (col..n)
            .max_by(|&a, &b| {
                m[a * n + col]
                    .abs()
                    .partial_cmp(&m[b * n + col].abs())
                    .unwrap()
            })
            .unwrap();
        for k in 0..n {
            m.swap(col * n + k, piv * n + k);
        }
        rhs.swap(col, piv);
        assert!(
            m[col * n + col].abs() > 1e-12,
            "singular design matrix at column {col}"
        );
        for row in 0..n {
            if row == col {
                continue;
            }
            let f = m[row * n + col] / m[col * n + col];
            for k in col..n {
                m[row * n + k] -= f * m[col * n + k];
            }
            rhs[row] -= f * rhs[col];
        }
    }
    (0..n).map(|i| rhs[i] / m[i * n + i]).collect()
}

// ---------------------------------------------------------------------------
// the symbol embedding
// ---------------------------------------------------------------------------

/// The four features a symbol carries, in the order the embedding stores them:
/// two channels of register code, the bit, and the write/query flag.
///
/// Every embedding has the same norm, chosen so the layer's pre-`RmsNorm`
/// (`γ = 1`, over `D_MODEL` channels) passes it through unchanged. The query's
/// flag is `−√3` rather than `−1` for exactly that reason: it is the only
/// channel it has.
fn embedding(symbol: usize) -> [f64; D_MODEL] {
    let scale = 2.0 / 3.0f64.sqrt();
    let feature = if symbol == QUERY {
        [0.0, 0.0, 0.0, -3.0f64.sqrt()]
    } else {
        let code = REGISTER_CODE[symbol / 2];
        let bit = if symbol % 2 == 0 { -1.0 } else { 1.0 };
        [code[0], code[1], bit, 1.0]
    };
    assert!(
        (feature.iter().map(|x| x * x).sum::<f64>() - 3.0).abs() < 1e-12,
        "every embedding must have the same norm"
    );
    std::array::from_fn(|i| feature[i] * scale)
}

/// The 4 features, undoing [`embedding`]'s scaling — what the target tables
/// below are written in terms of.
fn feature(symbol: usize) -> [f64; D_MODEL] {
    let e = embedding(symbol);
    std::array::from_fn(|i| e[i] * 3.0f64.sqrt() / 2.0)
}

/// Fit the block's fused projection: one affine functional of the embedding per
/// column, exactly reproducing `targets[symbol][column]` at **every** symbol.
///
/// The assert is the load-bearing part of the derivation: seven symbols do not
/// fit in a four-dimensional embedding by accident. Each column the block needs
/// — the register axis, the bit, the write-enable — is a function of *one*
/// feature, and so is affine in all four.
fn fit_affine(targets: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    let n = D_MODEL + 1;
    let cols = targets[0].len();
    // design matrix rows: [embedding | 1]
    let rows: Vec<Vec<f64>> = (0..NUM_SYMBOLS)
        .map(|s| {
            let mut r = embedding(s).to_vec();
            r.push(1.0);
            r
        })
        .collect();
    // normal equations XᵀX (shared by every column)
    let mut xtx = vec![0.0; n * n];
    for r in &rows {
        for i in 0..n {
            for j in 0..n {
                xtx[i * n + j] += r[i] * r[j];
            }
        }
    }

    let mut weight = vec![0.0; D_MODEL * cols];
    let mut bias = vec![0.0; cols];
    for c in 0..cols {
        let mut xty = vec![0.0; n];
        for (s, r) in rows.iter().enumerate() {
            for (i, ri) in r.iter().enumerate() {
                xty[i] += ri * targets[s][c];
            }
        }
        let z = solve(xtx.clone(), xty, n);
        for (s, r) in rows.iter().enumerate() {
            let got: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
            assert!(
                (got - targets[s][c]).abs() < 1e-9,
                "column {c} is not an affine functional of the embedding \
                 (symbol {s}: wanted {}, got {got})",
                targets[s][c],
            );
        }
        for (i, zi) in z.iter().take(D_MODEL).enumerate() {
            weight[i * cols + c] = *zi; // Linear weight is [d_in, d_out]
        }
        bias[c] = z[D_MODEL];
    }
    (weight, bias)
}

/// `kᵢ` before activation: `mag` on the written register's axis, `0` elsewhere.
fn key_targets(symbol: usize, mag: f64) -> [f64; NUM_REGISTERS] {
    let f = feature(symbol);
    std::array::from_fn(|i| {
        mag * (1.0 / 3.0 + (2.0 / 3.0) * (REGISTER_CODE[i][0] * f[0] + REGISTER_CODE[i][1] * f[1]))
    })
}

/// `v` before silu: the bit on one channel, the constant reference on the other.
fn value_targets(symbol: usize) -> [f64; HEAD_V_DIM] {
    let bit = feature(symbol)[2];
    let mid = (silu_inv(V_MAG) + silu_inv(-V_MAG)) / 2.0;
    let half = (silu_inv(V_MAG) - silu_inv(-V_MAG)) / 2.0;
    [mid + half * bit, silu_inv(R_MAG)]
}

// ---------------------------------------------------------------------------
// the shared network boundary
// ---------------------------------------------------------------------------

/// Set the pieces that are the same for both constructions: the symbol
/// embedding, the pass-through pre-norm, and the two-class head.
fn wire_boundary(net_in: &mut burn::nn::Linear, net_out: &mut burn::nn::Linear, device: &Device) {
    let table: Vec<f64> = (0..NUM_SYMBOLS).flat_map(embedding).collect();
    net_in.weight = Param::from_tensor(t1(&table, [NUM_SYMBOLS, D_MODEL], device));
    net_in.bias = Some(Param::from_tensor(Tensor::zeros(
        Shape::new([D_MODEL]),
        device,
    )));

    // logits [NEG, POS] = [−g·o₀, +g·o₀]; `ignore_last_residual` means `o` is
    // all the head sees. `o₁` (the reference axis) enters only through the
    // block's normalisation, which is what keeps the margin proportional to the
    // majority rather than saturating it.
    let mut w = vec![0.0; D_MODEL * NUM_CLASSES];
    w[0] = -OUT_GAIN;
    w[1] = OUT_GAIN;
    net_out.weight = Param::from_tensor(t1(&w, [D_MODEL, NUM_CLASSES], device));
    net_out.bias = Some(Param::from_tensor(Tensor::zeros(
        Shape::new([NUM_CLASSES]),
        device,
    )));
}

/// `out_proj` of a block: value channel 0 (the majority axis) onto `d_model`
/// channel 0, channel 1 onto channel 1.
fn block_out(device: &Device) -> burn::nn::Linear {
    let mut w = vec![0.0; HEAD_V_DIM * D_MODEL];
    w[0] = 1.0;
    w[D_MODEL + 1] = 1.0;
    burn::nn::Linear {
        weight: Param::from_tensor(t1(&w, [HEAD_V_DIM, D_MODEL], device)),
        bias: Some(Param::from_tensor(Tensor::zeros(
            Shape::new([D_MODEL]),
            device,
        ))),
    }
}

fn set_out_norm(norm: &mut OutNorm, device: &Device) {
    let ones = Param::from_tensor(Tensor::ones(Shape::new([HEAD_V_DIM]), device));
    match norm {
        OutNorm::Plain(rms) => rms.gamma = ones,
        OutNorm::Gated(rms) => rms.gamma = ones,
    }
}

// ---------------------------------------------------------------------------
// 1. the hand-built DeltaNet
// ---------------------------------------------------------------------------

/// Build the DeltaNet block by hand — every weight in closed form, nothing fit.
fn handmade(device: &Device) -> DeltaLatentNet {
    let cfg = crate::model::model_config();
    let mut model = ModelConfigExt::init(&cfg, device);
    wire_boundary(&mut model.in_proj, &mut model.out_proj, device);

    let layer = &mut model.layers.real_layers[0];
    layer.norm.gamma = Param::from_tensor(Tensor::ones(Shape::new([D_MODEL]), device));
    let DeltaBlock::DeltaNet(block) = &mut layer.block else {
        unreachable!("register-majority configures the DeltaNet variant")
    };

    // in_proj columns are `[q(3) | k(3) | v(2) | β(1)]` — see
    // `QkvProjection::segments()`. `qk_activation = Identity` here, so `q`/`k`
    // reach the L2 norm exactly as projected.
    let targets: Vec<Vec<f64>> = (0..NUM_SYMBOLS)
        .map(|s| {
            let mut row = vec![1.0; NUM_REGISTERS]; // q: read every register
            row.extend(key_targets(s, 1.0)); //         k: the written register's axis
            row.extend(value_targets(s)); //            v: the bit + the reference
            row.push(BIG * feature(s)[3]); //           β: write-enable
            row
        })
        .collect();
    let cols = targets[0].len();
    let (w, b) = fit_affine(&targets);
    block.qkv.in_proj.weight = Param::from_tensor(t1(&w, [D_MODEL, cols], device));
    block.qkv.in_proj.bias = Some(Param::from_tensor(t1(&b, [cols], device)));

    set_out_norm(&mut block.norm, device);
    block.out_proj = block_out(device);
    model
}

// ---------------------------------------------------------------------------
// 2. the same recurrence with the erase switched off
// ---------------------------------------------------------------------------

/// Build the accumulating baseline: GDN-2 with its **erase** gate pinned at 0
/// and its per-channel forget gate pinned at `alpha`, i.e.
/// `Sₜ = α Sₜ₋₁ + kₜ (w ⊙ vₜ)ᵀ` — gated linear attention, and nothing else
/// changed.
///
/// Note there is no readout gain to sweep alongside `alpha`: the block's
/// RMSNorm keeps only the output's *direction*, so any scalar gain on `q` or
/// `v` is invisible. `alpha` is the whole remaining degree of freedom.
fn accumulator(device: &Device, alpha: f64) -> DeltaLatentNet {
    const LOWER_BOUND: f64 = -5.0;
    assert!(
        alpha > LOWER_BOUND.exp() && alpha < 1.0,
        "alpha must lie inside the gate's reachable range"
    );

    let block_cfg = GatedDeltaNet2Config::new(D_MODEL)
        .with_nheads(1)
        .with_head_k_dim(NUM_REGISTERS)
        .with_expand_v(HEAD_V_DIM as f64 / NUM_REGISTERS as f64)
        .with_bottleneck(1)
        .with_use_gate(false)
        .with_allow_neg_eigval(Some(false))
        .with_use_short_conv(false)
        .with_lower_bound(LOWER_BOUND)
        .with_has_proj_bias(true);
    let cfg = DeltaLatentNetConfig::new(
        DeltaLatentShape::new(
            NUM_SYMBOLS,
            NUM_CLASSES,
            DeltaNetworkShape::new(1).with_ignore_last_residual(true),
        )
        .with_final_norm(false),
        DeltaBlockConfig::GatedDeltaNet2(block_cfg),
    );
    let mut model = ModelConfigExt::init(&cfg, device);
    wire_boundary(&mut model.in_proj, &mut model.out_proj, device);

    let layer = &mut model.layers.real_layers[0];
    layer.norm.gamma = Param::from_tensor(Tensor::ones(Shape::new([D_MODEL]), device));
    let DeltaBlock::GatedDeltaNet2(block) = &mut layer.block else {
        unreachable!("the baseline is the GDN-2 variant")
    };

    // in_proj columns are `[q(3) | k(3) | v(2) | erase(3) | write(2) | Δ_in(1)]`.
    // GDN-2 applies silu to `q`/`k` before the L2 norm, so the axes are asked
    // for at `QK_MAG` rather than 1 — only the direction survives either way.
    let targets: Vec<Vec<f64>> = (0..NUM_SYMBOLS)
        .map(|s| {
            let mut row = vec![QK_MAG; NUM_REGISTERS]; // q: read every register
            row.extend(key_targets(s, QK_MAG)); //        k: the written register
            row.extend(value_targets(s)); //              v: the bit + the reference
            row.extend([-BIG; NUM_REGISTERS]); //         erase: off. this is the ablation
            row.extend([BIG * feature(s)[3]; HEAD_V_DIM]); // write: enabled on writes
            row.push(0.0); //                             Δ_in: unused, the gate is constant
            row
        })
        .collect();
    let cols = targets[0].len();
    let (w, b) = fit_affine(&targets);
    block.qkv.in_proj.weight = Param::from_tensor(t1(&w, [D_MODEL, cols], device));
    block.qkv.in_proj.bias = Some(Param::from_tensor(t1(&b, [cols], device)));

    // g = σ(Δ · exp(a_log)) · lower_bound, and α = exp(g). With `up` zeroed the
    // gate reads nothing from the token, so α is the constant swept below.
    block.gate.up.weight = Param::from_tensor(Tensor::zeros(
        Shape::new([1, NUM_REGISTERS]),
        device,
    ));
    block.gate.a_log_h = Param::from_tensor(Tensor::zeros(Shape::new([1]), device));
    block.gate.dt_bias_i = Param::from_tensor(t1(
        &[logit(alpha.ln() / LOWER_BOUND); NUM_REGISTERS],
        [NUM_REGISTERS],
        device,
    ));

    set_out_norm(&mut block.norm, device);
    block.out_proj = block_out(device);
    model
}

// ---------------------------------------------------------------------------
// evaluation
// ---------------------------------------------------------------------------

/// Per-position accuracy of `model` on `count` sequences of one family.
fn accuracy(model: &DeltaLatentNet, family: Family, count: usize, device: &Device) -> f64 {
    let items: Vec<_> = RegisterDataset::new(count, SEQ_LENGTH, family, 0xE7A1)
        .iter()
        .map(|i| i.expect("dataset item"))
        .collect();
    let inputs = Tensor::stack(
        items
            .iter()
            .map(|i| one_hot(&i.symbols, device))
            .collect::<Vec<_>>(),
        0,
    );
    let (out, _c) = model.forward(inputs, None, path(), None);
    let n = count * SEQ_LENGTH;
    let pred = out
        .reshape([n, NUM_CLASSES])
        .argmax(1)
        .reshape([n])
        .into_data()
        .try_to_vec::<i32>()
        .unwrap();
    let want: Vec<i64> = items.iter().flat_map(|i| i.targets.clone()).collect();
    let scored: Vec<(i64, i64)> = pred
        .iter()
        .zip(&want)
        .filter(|(_, t)| **t != IGNORE)
        .map(|(p, t)| (i64::from(*p), *t))
        .collect();
    assert!(!scored.is_empty(), "no scored positions in {family:?}");
    let hits = scored.iter().filter(|(p, t)| p == t).count();
    hits as f64 / scored.len() as f64
}

const FAMILIES: [(&str, Family); 3] = [
    ("random", Family::Random),
    ("stale", Family::Stale),
    ("fading", Family::Fading),
];

// ---------------------------------------------------------------------------
// the tests
// ---------------------------------------------------------------------------

/// Every weight written down in closed form; no training anywhere.
#[test]
fn handmade_block_solves_every_family() {
    let device = Device::default();
    let model = handmade(&device);
    println!(
        "hand-built DeltaNet ({} params, {} state scalars):",
        model.num_params(),
        NUM_REGISTERS * HEAD_V_DIM
    );
    let mut worst = 1.0f64;
    for (name, family) in FAMILIES {
        let acc = accuracy(&model, family, 256, &device);
        println!("  {name:<8} {:6.2}%", 100.0 * acc);
        worst = worst.min(acc);
    }
    assert!(worst > 0.995, "hand-built solution is not exact: {worst}");
}

/// Sweep the decay of the same block with its **erase gate switched off** — the
/// one changed knob — and report the best any of them do.
///
/// Both adversarial families are unreachable at once, and from opposite sides: a
/// decay near 1 keeps every superseded write in the sum (`stale`), a decay away
/// from 1 fades the register written longest ago before it can vote (`fading`).
#[test]
fn no_accumulating_state_solves_the_task() {
    let device = Device::default();
    let alphas = [
        1.0 - 1e-6,
        0.99,
        0.95,
        0.9,
        0.8,
        0.7,
        0.5,
        0.3,
        0.1,
        0.02,
    ];

    println!("erase off (gated linear attention), accuracy per family:");
    println!("      α     random    stale    fading     worst");
    let mut best_worst = 0.0f64;
    for alpha in alphas {
        let model = accumulator(&device, alpha);
        let accs: Vec<f64> = FAMILIES
            .iter()
            .map(|(_, f)| accuracy(&model, *f, 128, &device))
            .collect();
        let worst = accs.iter().cloned().fold(1.0f64, f64::min);
        println!(
            "  {alpha:7.5}  {:6.2}%  {:6.2}%   {:6.2}%   {:6.2}%",
            100.0 * accs[0],
            100.0 * accs[1],
            100.0 * accs[2],
            100.0 * worst
        );
        best_worst = best_worst.max(worst);
    }
    println!(
        "best worst-family accuracy over the whole sweep: {:.2}%",
        100.0 * best_worst
    );
    assert!(
        best_worst < 0.75,
        "an accumulating state reached {best_worst:.4} — the task does not need the erase"
    );
}

/// `forward()` and `step()` are the same function — checked end to end on the
/// hand-built model, over a whole sequence of every family.
///
/// The chunkwise WY path and the token-by-token recurrence are two evaluations
/// of one recurrence; the library asserts that per block, and this asserts it
/// where a user would notice: the class the example actually reads off.
#[test]
fn decoding_one_token_at_a_time_agrees_with_the_chunked_pass() {
    let device = Device::default();
    let model = handmade(&device);
    let mut worst_agreement = 1.0f64;
    for (name, family) in FAMILIES {
        let items: Vec<_> = RegisterDataset::new(16, SEQ_LENGTH, family, 0x5157)
            .iter()
            .map(|i| i.expect("dataset item"))
            .collect();
        let batch = items.len();
        let inputs = Tensor::stack(
            items
                .iter()
                .map(|i| one_hot(&i.symbols, &device))
                .collect::<Vec<_>>(),
            0,
        );

        let (chunked, _) = model.forward(inputs.clone(), None, path(), None);

        // the same sequence, one token at a time, threading the caches
        let mut caches = None;
        let mut steps = Vec::with_capacity(SEQ_LENGTH);
        for t in 0..SEQ_LENGTH {
            let token = inputs.clone().narrow(1, t, 1).squeeze_dim(1);
            let (y, next) = model.step(token, caches, None);
            caches = Some(next);
            steps.push(y);
        }
        let stepped = Tensor::stack::<3>(steps, 1);
        assert_eq!(chunked.dims(), stepped.dims());

        // Compare the logits themselves: an argmax over an *unscored* position
        // — where the register file is still empty and both logits sit at zero
        // — flips on a difference of 1e-7, which says nothing about the paths.
        let gap = (chunked.clone() - stepped.clone())
            .abs()
            .max()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()[0] as f64;

        // ... and on the positions the example actually reads, the decision is
        // identical, not merely close.
        let n = batch * SEQ_LENGTH;
        let argmax = |t: Tensor<3>| {
            t.reshape([n, NUM_CLASSES])
                .argmax(1)
                .reshape([n])
                .into_data()
                .try_to_vec::<i32>()
                .unwrap()
        };
        let (a, b) = (argmax(chunked), argmax(stepped));
        let want: Vec<i64> = items.iter().flat_map(|i| i.targets.clone()).collect();
        let scored: Vec<(i32, i32)> = a
            .iter()
            .zip(&b)
            .zip(&want)
            .filter(|(_, t)| **t != IGNORE)
            .map(|((x, y), _)| (*x, *y))
            .collect();
        let agree = scored.iter().filter(|(x, y)| x == y).count() as f64 / scored.len() as f64;
        println!(
            "  {name:<8} max |forward − step| {gap:.2e}, scored decisions agree {:6.2}%",
            100.0 * agree
        );
        assert!(gap < 1e-4, "{name}: the two paths differ by {gap}");
        worst_agreement = worst_agreement.min(agree);
    }
    assert!(
        worst_agreement > 0.999,
        "the chunked and recurrent paths disagree: {worst_agreement}"
    );
}

/// The **windowed ceiling**: the best a model that sees only the last `w`
/// symbols can do, fit *on the evaluation data itself* so it is a genuine upper
/// bound rather than a trained baseline.
///
/// No local window gets near the state, which is why the block runs with
/// `use_short_conv = false` — there is nothing for a convolution to find.
#[test]
fn windowed_ceilings_are_far_below_the_state() {
    use std::collections::HashMap;
    println!("best in-sample lookup table over the last w symbols:");
    let mut ceilings = Vec::new();
    for w in [1usize, 2, 3, 4, 6] {
        let mut tally: HashMap<Vec<usize>, [u64; NUM_CLASSES]> = HashMap::new();
        for (_, family) in FAMILIES {
            for item in RegisterDataset::new(NUM_EVAL, SEQ_LENGTH, family, 0xE7A1).iter() {
                let item = item.expect("dataset item");
                for (t, &c) in labels(&item.symbols).iter().enumerate() {
                    if c == IGNORE {
                        continue;
                    }
                    let key = item.symbols[t.saturating_sub(w - 1)..=t].to_vec();
                    tally.entry(key).or_default()[c as usize] += 1;
                }
            }
        }
        let total: u64 = tally.values().flatten().sum();
        let best: u64 = tally.values().map(|r| r.iter().max().copied().unwrap()).sum();
        let ceiling = best as f64 / total as f64;
        println!(
            "  window {w}: {:6.2}%   ({} distinct windows memorised)",
            100.0 * ceiling,
            tally.len()
        );
        ceilings.push(ceiling);
    }
    println!(
        "chance {:.2}%, hand-built block 100%",
        100.0 / NUM_CLASSES as f64
    );
    assert!(
        ceilings[0] < 0.55,
        "the current symbol nearly gives the answer: {}",
        ceilings[0]
    );
    assert!(
        *ceilings.last().unwrap() < 0.8,
        "a six-symbol window nearly gives the answer: {}",
        ceilings.last().unwrap()
    );
}
