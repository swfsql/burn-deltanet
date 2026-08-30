//! The claims this example rests on, measured.
//!
//! 1. A **hand-built** DeltaProduct block solves both turns exactly — every
//!    weight in closed form, no fitting: a push is one factor at `β = 1`, a turn
//!    is one factor per transposition at `β = 2`.
//! 2. **One** Householder per token cannot reach the 3-cycle, for *any* axis and
//!    *any* `β` — swept over the sphere. It does reach the transposition, which
//!    is the contrast: the ladder is `β ≤ 1` → nothing, one reflection → a swap,
//!    two → a 3-cycle.
//! 3. And `β ≤ 1` (`allow_neg_eigval` off) cannot even swap: the transition's
//!    spectrum is then `[0, 1]` and no orbit of it oscillates.
//!
//! The sweep is over real blocks, not matrices: each candidate is a complete
//! model whose turn factor is `(axis, β)` and whose every other weight is the
//! exact construction, so what is measured is the best a single-Householder
//! block can do on this task, not a matrix-norm proxy.

use crate::common::model::ModelConfigExt;
use crate::dataset::{
    CarouselDataset, Family, IGNORE, NUM_CLASSES, NUM_REGISTERS, NUM_SYMBOLS, PUSH_NEG, PUSH_POS,
    QUERY, SEQ_LENGTH, TURN, Turn, one_hot,
};
use crate::model::{D_MODEL, HEAD_V_DIM, default_factors, model_config};
use crate::training::path;
use burn::data::dataset::Dataset;
use burn::module::Param;
use burn::prelude::*;
use burn_deltanet::prelude::*;

// ---------------------------------------------------------------------------
// the construction's constants
// ---------------------------------------------------------------------------

/// `v`'s bit channel, `±V`; bounded by silu's floor (`min silu = −0.2785`).
const V_MAG: f64 = 0.25;
/// `v`'s reference channel: constant and positive, so the block's per-head
/// RMSNorm sees a direction rather than a lone scalar it would flatten.
const R_MAG: f64 = 0.25;
/// `q`/`k` are asked for at this magnitude before the L2 norm. It must stay
/// inside silu's invertible range, since only the *direction* survives anyway.
const AXIS_MAG: f64 = 0.25;
/// The saturating logit: `2σ(±BIG)` is 2 / 0 to well under f32 resolution.
const BIG: f64 = 20.0;
/// Class-logit gain on the bit axis.
const OUT_GAIN: f64 = 6.0;

/// The four symbol embeddings: the vertices of a regular tetrahedron. Affinely
/// independent, so **every** per-symbol table below is a unique affine
/// functional of the embedding; equal-norm, and scaled so `mean(x²) = 1`, so the
/// layer's pre-`RmsNorm` (`γ = 1`) passes them through unchanged.
const TETRAHEDRON: [[f64; D_MODEL]; NUM_SYMBOLS] = [
    [1.0, 1.0, 1.0],
    [1.0, -1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
];

/// The transposition `(A B)` as a reflection axis: `I − 2wwᵀ` swaps rows 0 and 1.
const SWAP_AB: [f64; NUM_REGISTERS] = [1.0, -1.0, 0.0];
/// The transposition `(A C)`. `SWAP_AC ∘ SWAP_AB` is the 3-cycle `A ← C ← B ← A`.
const SWAP_AC: [f64; NUM_REGISTERS] = [1.0, 0.0, -1.0];

/// The factors a turn decomposes into, innermost (applied first) first.
fn turn_axes(turn: Turn) -> Vec<[f64; NUM_REGISTERS]> {
    match turn {
        Turn::Swap => vec![SWAP_AB],
        Turn::Rotate => vec![SWAP_AB, SWAP_AC],
    }
}

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

fn t1<const D: usize>(v: &[f64], shape: [usize; D], device: &Device) -> Tensor<D> {
    let f: Vec<f32> = v.iter().map(|&x| x as f32).collect();
    Tensor::<1>::from_floats(f.as_slice(), device).reshape(shape)
}

/// Solve `m·x = rhs` (`n × n`, row-major) by Gaussian elimination with pivoting.
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
        assert!(m[col * n + col].abs() > 1e-12, "singular design matrix");
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

/// One affine functional of the embedding per column, hitting `targets` exactly.
///
/// Four symbols at four affinely independent points and `D_MODEL + 1 = 4`
/// coefficients: the system is square and always solvable, which is the whole
/// reason the tetrahedron is the embedding.
fn fit_affine(targets: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    let n = D_MODEL + 1;
    assert_eq!(n, NUM_SYMBOLS, "the design matrix must be square");
    let cols = targets[0].len();
    let scale = 1.0 / (3.0f64 / D_MODEL as f64).sqrt(); // ⇒ mean(x²) = 1
    let mut design = vec![0.0; n * n];
    for s in 0..NUM_SYMBOLS {
        for d in 0..D_MODEL {
            design[s * n + d] = TETRAHEDRON[s][d] * scale;
        }
        design[s * n + D_MODEL] = 1.0;
    }

    let mut weight = vec![0.0; D_MODEL * cols];
    let mut bias = vec![0.0; cols];
    for c in 0..cols {
        let rhs: Vec<f64> = (0..NUM_SYMBOLS).map(|s| targets[s][c]).collect();
        let z = solve(design.clone(), rhs, n);
        for (d, zd) in z.iter().take(D_MODEL).enumerate() {
            weight[d * cols + c] = *zd; // Linear weight is [d_in, d_out]
        }
        bias[c] = z[D_MODEL];
    }
    (weight, bias)
}

/// The symbol embedding table, `[NUM_SYMBOLS, D_MODEL]`.
fn embedding_table() -> Vec<f64> {
    let scale = 1.0 / (3.0f64 / D_MODEL as f64).sqrt();
    TETRAHEDRON
        .iter()
        .flat_map(|row| row.iter().map(move |x| x * scale))
        .collect()
}

/// `q`/`k` before silu, so that after silu and the L2 norm they point along
/// `dir` exactly. Every component must stay inside silu's invertible range,
/// which is why `dir` is asked for at [`AXIS_MAG`] rather than unit length.
fn axis_targets(dir: [f64; NUM_REGISTERS]) -> [f64; NUM_REGISTERS] {
    let norm = dir.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(norm > 1e-9, "a zero axis has no direction");
    std::array::from_fn(|i| silu_inv(AXIS_MAG * dir[i] / norm))
}

/// `v` before silu: the bit on one channel, the constant reference on the other.
fn value_targets(bit: f64) -> [f64; HEAD_V_DIM] {
    let mid = (silu_inv(V_MAG) + silu_inv(-V_MAG)) / 2.0;
    let half = (silu_inv(V_MAG) - silu_inv(-V_MAG)) / 2.0;
    [mid + half * bit, silu_inv(R_MAG)]
}

/// The raw logit giving `β = 2σ(raw)`. `β = 0` and `β = 2` saturate.
fn beta_raw(beta: f64) -> f64 {
    assert!((0.0..=2.0).contains(&beta));
    let p = beta / 2.0;
    if p <= 1e-9 {
        -BIG
    } else if p >= 1.0 - 1e-9 {
        BIG
    } else {
        (p / (1.0 - p)).ln()
    }
}

// ---------------------------------------------------------------------------
// the hand-built model
// ---------------------------------------------------------------------------

/// One factor of one token: which axis it reflects about and how hard.
struct Factor {
    axis: [f64; NUM_REGISTERS],
    beta: f64,
    value: Option<f64>,
}

impl Factor {
    /// A factor that does nothing.
    fn idle() -> Self {
        Self {
            axis: [1.0, 0.0, 0.0],
            beta: 0.0,
            value: None,
        }
    }
}

/// Build the block by hand.
///
/// `turn_factors` is what the `R` symbol does, factor by factor — the *only*
/// thing the sweep below varies. Everything else (the push, the query, the
/// readout) is the same construction throughout.
fn handmade(device: &Device, factors: usize, turn_factors: &[Factor]) -> DeltaLatentNet {
    assert!(
        turn_factors.len() <= factors,
        "a turn cannot use more factors than the block has"
    );
    let cfg = model_config(Turn::Rotate, factors); // the turn shapes the data, not the block
    let mut model = ModelConfigExt::init(&cfg, device);
    let net = &mut model;

    net.in_proj.weight = Param::from_tensor(t1(
        &embedding_table(),
        [NUM_SYMBOLS, D_MODEL],
        device,
    ));
    net.in_proj.bias = Some(Param::from_tensor(Tensor::zeros(
        Shape::new([D_MODEL]),
        device,
    )));
    // logits [NEG, POS] = [−g·o₀, +g·o₀]; `ignore_last_residual` means `o` is
    // all the head sees.
    let mut head = vec![0.0; D_MODEL * NUM_CLASSES];
    head[0] = -OUT_GAIN;
    head[1] = OUT_GAIN;
    net.out_proj.weight = Param::from_tensor(t1(&head, [D_MODEL, NUM_CLASSES], device));
    net.out_proj.bias = Some(Param::from_tensor(Tensor::zeros(
        Shape::new([NUM_CLASSES]),
        device,
    )));

    let layer = &mut net.layers.real_layers[0];
    layer.norm.gamma = Param::from_tensor(Tensor::ones(Shape::new([D_MODEL]), device));
    let DeltaBlock::DeltaProduct(block) = &mut layer.block else {
        unreachable!("register-carousel configures the DeltaProduct variant")
    };

    // in_proj columns are `[q(3) | k(3·u) | v(2·u) | β(1·u)]`, the per-factor
    // segments in micro-step order — see `QkvProjection::segments()`.
    let targets: Vec<Vec<f64>> = (0..NUM_SYMBOLS)
        .map(|symbol| {
            // what each of this symbol's factors does
            let mut plan: Vec<Factor> = (0..factors).map(|_| Factor::idle()).collect();
            match symbol {
                PUSH_NEG | PUSH_POS => {
                    // the write goes on the *last* factor, which is where `q`
                    // lives, so the readout sees it in the same token
                    plan[factors - 1] = Factor {
                        axis: [1.0, 0.0, 0.0], // register A: the input port
                        beta: 1.0,             // a full replacement
                        value: Some(if symbol == PUSH_POS { 1.0 } else { -1.0 }),
                    };
                }
                TURN => {
                    for (slot, f) in turn_factors.iter().enumerate() {
                        plan[slot] = Factor {
                            axis: f.axis,
                            beta: f.beta,
                            value: f.value,
                        };
                    }
                }
                QUERY => {} // every factor idle: the file is only read
                _ => unreachable!(),
            }

            let mut row = axis_targets([1.0, 0.0, 0.0]).to_vec(); // q: the read port
            for f in &plan {
                row.extend(axis_targets(f.axis));
            }
            for f in &plan {
                // silu(0) = 0, so an idle factor writes nothing at all
                row.extend(match f.value {
                    Some(bit) => value_targets(bit),
                    None => [0.0; HEAD_V_DIM],
                });
            }
            for f in &plan {
                row.push(beta_raw(f.beta));
            }
            row
        })
        .collect();
    let cols = targets[0].len();
    let (w, b) = fit_affine(&targets);
    block.qkv.in_proj.weight = Param::from_tensor(t1(&w, [D_MODEL, cols], device));
    block.qkv.in_proj.bias = Some(Param::from_tensor(t1(&b, [cols], device)));

    let ones = Param::from_tensor(Tensor::ones(Shape::new([HEAD_V_DIM]), device));
    match &mut block.norm {
        OutNorm::Plain(rms) => rms.gamma = ones,
        OutNorm::Gated(rms) => rms.gamma = ones,
    }
    // out_proj: the bit axis onto d_model channel 0, the reference onto 1.
    let mut w = vec![0.0; HEAD_V_DIM * D_MODEL];
    w[0] = 1.0;
    w[D_MODEL + 1] = 1.0;
    block.out_proj.weight = Param::from_tensor(t1(&w, [HEAD_V_DIM, D_MODEL], device));
    block.out_proj.bias = Some(Param::from_tensor(Tensor::zeros(
        Shape::new([D_MODEL]),
        device,
    )));
    model
}

/// The exact solution for `turn`: one reflection per transposition.
fn exact(device: &Device, turn: Turn, factors: usize) -> DeltaLatentNet {
    let plan: Vec<Factor> = turn_axes(turn)
        .into_iter()
        .map(|axis| Factor {
            axis,
            beta: 2.0, // an exact reflection; anything less blends the registers
            value: None,
        })
        .collect();
    handmade(device, factors, &plan)
}

// ---------------------------------------------------------------------------
// evaluation
// ---------------------------------------------------------------------------

/// Per-position accuracy of `model` on `count` sequences of one family.
fn accuracy(
    model: &DeltaLatentNet,
    family: Family,
    turn: Turn,
    count: usize,
    device: &Device,
) -> f64 {
    let items: Vec<_> = CarouselDataset::new(count, SEQ_LENGTH, family, turn, 0xE7A1)
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

const FAMILIES: [(&str, Family); 2] = [("random", Family::Random), ("cycle", Family::Cycle)];

/// Worst-family accuracy of `model` under `turn`.
fn worst(model: &DeltaLatentNet, turn: Turn, count: usize, device: &Device) -> (f64, Vec<f64>) {
    let accs: Vec<f64> = FAMILIES
        .iter()
        .map(|(_, f)| accuracy(model, *f, turn, count, device))
        .collect();
    (accs.iter().cloned().fold(1.0f64, f64::min), accs)
}

/// A roughly uniform grid of directions on the unit 2-sphere.
///
/// Only the axis *line* matters (`I − βkkᵀ` is even in `k`), so a hemisphere
/// would do; the whole sphere is swept anyway, for the same cost.
fn sphere_grid(rings: usize) -> Vec<[f64; NUM_REGISTERS]> {
    let mut out = Vec::new();
    for i in 0..rings {
        let z = -1.0 + 2.0 * (i as f64 + 0.5) / rings as f64;
        let r = (1.0 - z * z).max(0.0).sqrt();
        let count = ((rings as f64 * r).round() as usize).max(1);
        for j in 0..count {
            let theta = std::f64::consts::TAU * j as f64 / count as f64;
            out.push([r * theta.cos(), r * theta.sin(), z]);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// the tests
// ---------------------------------------------------------------------------

/// Every weight written down in closed form; no training anywhere.
///
/// `Rotate` is run at both `u = 2` (what it needs) and `u = 3` (a factor to
/// spare), so the construction is shown not to depend on the block being sized
/// exactly to the permutation.
#[test]
fn handmade_block_solves_both_turns() {
    let device = Device::default();
    for turn in [Turn::Swap, Turn::Rotate] {
        let needed = default_factors(turn);
        for factors in [needed, needed + 1] {
            let model = exact(&device, turn, factors);
            let (w, accs) = worst(&model, turn, 256, &device);
            println!(
                "  turn {turn:?}, {factors} factor(s), {} params, {} state scalars:  \
                 random {:6.2}%  cycle {:6.2}%",
                model.num_params(),
                NUM_REGISTERS * HEAD_V_DIM,
                100.0 * accs[0],
                100.0 * accs[1],
            );
            assert!(w > 0.995, "the hand-built solution is not exact: {w}");
        }
    }
}

/// Sweep **every** single Householder — axis over the sphere, `β` over `(0, 2]`
/// — with the rest of the construction untouched, and report the best any of
/// them does.
///
/// A transposition is reached exactly. The 3-cycle is not, and cannot be: with
/// `‖k‖ = 1` the transition's eigenvalues are `{1, 1, 1−β}`, all real, so its
/// orbits are sums of real geometric terms — period 1 or 2, never 3. The best a
/// period-≤2 output can do against a period-3 target is two positions in three,
/// which is what the `cycle` column lands on.
#[test]
fn one_householder_swaps_but_cannot_rotate() {
    let device = Device::default();
    let grid = sphere_grid(8);
    let betas = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0];
    println!("sweeping {} axes × {} β, one factor per token:", grid.len(), betas.len());

    let mut best: [(f64, Vec<f64>, f64); 2] =
        std::array::from_fn(|_| (0.0, vec![0.0; 2], 0.0));
    for (idx, turn) in [Turn::Swap, Turn::Rotate].into_iter().enumerate() {
        for axis in &grid {
            for beta in betas {
                let plan = [Factor {
                    axis: *axis,
                    beta,
                    value: None,
                }];
                let model = handmade(&device, 1, &plan);
                let (w, accs) = worst(&model, turn, 48, &device);
                if w > best[idx].0 {
                    best[idx] = (w, accs, beta);
                }
            }
        }
        println!(
            "  turn {turn:?}: best worst-family {:6.2}%  (random {:6.2}%, cycle {:6.2}%, β = {})",
            100.0 * best[idx].0,
            100.0 * best[idx].1[0],
            100.0 * best[idx].1[1],
            best[idx].2,
        );
    }
    assert!(
        best[0].0 > 0.995,
        "one reflection should *be* a transposition, but only reached {:.4}",
        best[0].0
    );
    assert!(
        best[1].0 < 0.75,
        "a single Householder reached {:.4} on the 3-cycle — it should be capped \
         near ⅔ by having only real eigenvalues",
        best[1].0
    );
}

/// The same sweep restricted to `β ≤ 1` — what the block does with
/// `allow_neg_eigval` off. The transition's spectrum is then `{1, 1, 1−β} ⊂
/// [0, 1]`: non-negative, so nothing it does can oscillate, and even the
/// transposition is out of reach.
#[test]
fn a_contraction_cannot_even_swap() {
    let device = Device::default();
    let grid = sphere_grid(8);
    let mut best = 0.0f64;
    let mut detail = vec![0.0; 2];
    for axis in &grid {
        for beta in [0.25, 0.5, 0.75, 1.0] {
            let plan = [Factor {
                axis: *axis,
                beta,
                value: None,
            }];
            let model = handmade(&device, 1, &plan);
            let (w, accs) = worst(&model, Turn::Swap, 48, &device);
            if w > best {
                best = w;
                detail = accs;
            }
        }
    }
    println!(
        "  β ≤ 1 on the swap: best worst-family {:6.2}%  (random {:6.2}%, cycle {:6.2}%)",
        100.0 * best,
        100.0 * detail[0],
        100.0 * detail[1],
    );
    println!("  the same axes with β = 2 reach 100% — see `one_householder_swaps_but_cannot_rotate`");
    assert!(
        best < 0.75,
        "a contraction reached {best:.4} — negative eigenvalues are not what the swap needs"
    );
}

/// The **windowed ceiling**: the best a model seeing only the last `w` symbols
/// can do, fit *on the evaluation data itself* so it is a genuine upper bound.
///
/// The current symbol says nothing at all (a `?` is a `?`), and a window only
/// helps where it happens to reach back past the fill — which is why the number
/// creeps up with `w` and still leaves the state doing all the work. Nothing
/// here is within reach of a convolution, which is why the block runs with
/// `use_short_conv = false`.
#[test]
fn the_symbol_stream_carries_no_answer() {
    use std::collections::HashMap;
    for turn in [Turn::Swap, Turn::Rotate] {
        let mut ceilings = Vec::new();
        for w in [1usize, 4, 8] {
            let mut tally: HashMap<Vec<usize>, [u64; NUM_CLASSES]> = HashMap::new();
            for (_, family) in FAMILIES {
                for item in CarouselDataset::new(512, SEQ_LENGTH, family, turn, 0xE7A1).iter() {
                    let item = item.expect("dataset item");
                    for (t, &c) in item.targets.iter().enumerate() {
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
                "  turn {turn:?}, window {w}: best in-sample table {:6.2}%  ({} windows memorised)",
                100.0 * ceiling,
                tally.len()
            );
            ceilings.push(ceiling);
        }
        assert!(
            ceilings[0] < 0.55,
            "the current symbol nearly gives the answer under {turn:?}: {}",
            ceilings[0]
        );
        assert!(
            *ceilings.last().unwrap() < 0.8,
            "an eight-symbol window nearly gives the answer under {turn:?}: {}",
            ceilings.last().unwrap()
        );
    }
    println!("  chance 50.00%, hand-built block 100%");
}
