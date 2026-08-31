//! `ChunkRecalculated` against `Chunk`: the same forward, so the same values to
//! float noise, and a hand-written backward that must reproduce what autodiff
//! derives from that forward.
//!
//! The broader contract — every [`DeltaPath`] evaluating the same recurrence —
//! is asserted against [`DeltaPath::Recurrent`] in
//! [`delta::tests`](crate::delta::tests). What is pinned *here* is the pair:
//! the tolerance is tight because the two paths run the identical arithmetic.

#![allow(non_snake_case)]

use crate::common::norm::l2_normalize;
use crate::delta::path::{DeltaInput, DeltaPath};
use crate::delta::tri::TriSolve;
use burn::module::Param;
use burn::prelude::*;
use burn::tensor::Distribution;
use burn_stack::utils::test_helpers::max_abs_diff;

type Device = burn::prelude::Device;

/// A block's-eye view of one delta-rule call: `q`/`k` L2-normalised, gates
/// already squashed, `g ≤ 0`.
struct Raw {
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    erase: Tensor<4>,
    write: Tensor<4>,
    g: Option<Tensor<4>>,
    state: Tensor<4>,
}

/// How wide the three gates are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Gates {
    /// No forget gate at all (DeltaNet).
    Ungated,
    /// One number per head (Gated DeltaNet 1 / DeltaProduct).
    PerHead,
    /// One number per channel (GDN-2) — the branch that goes through
    /// [`decay`](crate::delta::decay).
    PerChannel,
}

fn random_raw(
    batch: usize,
    sequence: usize,
    nheads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    beta_max: f64,
    gates: Gates,
    device: &Device,
) -> Raw {
    let normal = Distribution::Normal(0.0, 1.0);
    let channel = gates == Gates::PerChannel;
    let (k_gate, v_gate) = if channel {
        (head_k_dim, head_v_dim)
    } else {
        (1, 1)
    };
    let erase = Tensor::<4>::random(
        [batch, sequence, nheads, k_gate],
        Distribution::Uniform(0.05, beta_max),
        device,
    );
    Raw {
        q: l2_normalize(Tensor::<4>::random(
            [batch, sequence, nheads, head_k_dim],
            normal,
            device,
        )),
        k: l2_normalize(Tensor::<4>::random(
            [batch, sequence, nheads, head_k_dim],
            normal,
            device,
        )),
        v: Tensor::<4>::random([batch, sequence, nheads, head_v_dim], normal, device),
        write: if channel {
            Tensor::<4>::random(
                [batch, sequence, nheads, v_gate],
                Distribution::Uniform(0.05, 1.0),
                device,
            )
        } else {
            erase.clone()
        },
        erase,
        g: (gates != Gates::Ungated).then(|| {
            Tensor::<4>::random(
                [batch, sequence, nheads, k_gate],
                Distribution::Uniform(-0.5, -0.01),
                device,
            )
        }),
        // A non-zero incoming state: the streaming case, and the only way
        // `d_state` gets exercised at all.
        state: Tensor::<4>::random(
            [batch, nheads, head_k_dim, head_v_dim],
            Distribution::Normal(0.0, 0.2),
            device,
        ),
    }
}

/// One autodiff graph over the same underlying values, plus what a backward
/// through it produces.
struct Run {
    y: Tensor<4>,
    state: Tensor<4>,
    d_q: Tensor<4>,
    d_k: Tensor<4>,
    d_v: Tensor<4>,
    d_erase: Tensor<4>,
    d_write: Tensor<4>,
    d_g: Option<Tensor<4>>,
    d_state: Tensor<4>,
}

fn run_path(path: DeltaPath, raw: &Raw, y_head: &Tensor<4>, s_head: &Tensor<4>) -> Run {
    let lift = |t: &Tensor<4>| Param::from_tensor(Tensor::from_inner(t.clone()));
    let (q, k, v) = (lift(&raw.q), lift(&raw.k), lift(&raw.v));
    let (erase, write) = (lift(&raw.erase), lift(&raw.write));
    let g = raw.g.as_ref().map(lift);
    let state = lift(&raw.state);

    let (y, final_state) = DeltaInput {
        q_bshk: q.val(),
        k_bshk: k.val(),
        v_bshv: v.val(),
        erase_bshK: erase.val(),
        write_bshV: write.val(),
        g_bshK: g.as_ref().map(|g| g.val()),
        state_bhkv: state.val(),
        scale: None,
    }
    .run(path);

    let y_inner = y.clone().inner();
    let state_inner = final_state.clone().inner();
    // Two distinct heads, so the `y` and `final_state` gradient paths are
    // exercised independently.
    let loss = (y * Tensor::from_inner(y_head.clone())).sum()
        + (final_state * Tensor::from_inner(s_head.clone())).sum();
    let grads = loss.backward();

    Run {
        y: y_inner,
        state: state_inner,
        d_q: q.val().grad(&grads).expect("grad q"),
        d_k: k.val().grad(&grads).expect("grad k"),
        d_v: v.val().grad(&grads).expect("grad v"),
        d_erase: erase.val().grad(&grads).expect("grad erase"),
        d_write: write.val().grad(&grads).expect("grad write"),
        d_g: g.as_ref().map(|g| g.val().grad(&grads).expect("grad g")),
        d_state: state.val().grad(&grads).expect("grad state"),
    }
}

fn assert_runs_match(taped: &Run, custom: &Run, label: &str, value_tol: f32, grad_tol: f32) {
    let mut failures = Vec::new();
    let mut check = |name: &str, diff: f32, tol: f32| {
        if !(diff < tol) {
            failures.push(format!("{name}: max abs diff {diff} (tol {tol})"));
        }
    };
    check("y", max_abs_diff(taped.y.clone(), custom.y.clone()), value_tol);
    check(
        "final_state",
        max_abs_diff(taped.state.clone(), custom.state.clone()),
        value_tol,
    );
    check("d_q", max_abs_diff(taped.d_q.clone(), custom.d_q.clone()), grad_tol);
    check("d_k", max_abs_diff(taped.d_k.clone(), custom.d_k.clone()), grad_tol);
    check("d_v", max_abs_diff(taped.d_v.clone(), custom.d_v.clone()), grad_tol);
    check(
        "d_erase",
        max_abs_diff(taped.d_erase.clone(), custom.d_erase.clone()),
        grad_tol,
    );
    check(
        "d_write",
        max_abs_diff(taped.d_write.clone(), custom.d_write.clone()),
        grad_tol,
    );
    check(
        "d_state",
        max_abs_diff(taped.d_state.clone(), custom.d_state.clone()),
        grad_tol,
    );
    if let (Some(a), Some(b)) = (&taped.d_g, &custom.d_g) {
        check("d_g", max_abs_diff(a.clone(), b.clone()), grad_tol);
    }
    assert!(
        failures.is_empty(),
        "Chunk vs ChunkRecalculated disagree ({label}):\n  {}",
        failures.join("\n  "),
    );
}

#[allow(clippy::too_many_arguments)]
fn check(
    batch: usize,
    sequence: usize,
    nheads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    chunk_len: usize,
    beta_max: f64,
    gates: Gates,
    grad_tol: f32,
) {
    let device: Device = Default::default();
    let raw = random_raw(
        batch, sequence, nheads, head_k_dim, head_v_dim, beta_max, gates, &device,
    );
    let y_head = Tensor::<4>::random(
        [batch, sequence, nheads, head_v_dim],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let s_head = Tensor::<4>::random(
        [batch, nheads, head_k_dim, head_v_dim],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    let taped = run_path(
        DeltaPath::Chunk {
            chunk_len: Some(chunk_len),
            solve: TriSolve::Blocked,
        },
        &raw,
        &y_head,
        &s_head,
    );
    let custom = run_path(
        DeltaPath::ChunkRecalculated {
            chunk_len: Some(chunk_len),
        },
        &raw,
        &y_head,
        &s_head,
    );
    assert_runs_match(
        &taped,
        &custom,
        &format!("{gates:?}, l={chunk_len}, s={sequence}"),
        1e-5,
        grad_tol,
    );
}

/// `α ≡ 1`: no gate tensors are built at all, and the node is told so.
#[test]
fn matches_the_taped_path_ungated() {
    check(2, 16, 3, 8, 8, 8, 0.95, Gates::Ungated, 1e-4);
}

/// One forget-gate value per head: the decay factors out of the key
/// contraction into a plain `[chunk_len, chunk_len]` mask.
#[test]
fn matches_the_taped_path_per_head_gate() {
    check(2, 16, 3, 8, 8, 8, 0.95, Gates::PerHead, 1e-4);
}

/// One value per channel on all three gates: the decay does *not* factor, so
/// the scores go through the block-reference split.
#[test]
fn matches_the_taped_path_per_channel_gates() {
    check(2, 16, 3, 8, 8, 8, 0.95, Gates::PerChannel, 1e-4);
    // Long enough to cross several reference blocks.
    check(2, 40, 2, 8, 8, 32, 0.95, Gates::PerChannel, 2e-4);
}

/// `β > 1` puts the Householder's second eigenvalue below zero — the
/// state-tracking regime, and the one most likely to expose a sign error.
#[test]
fn matches_the_taped_path_with_negative_eigenvalues() {
    check(2, 12, 2, 8, 8, 6, 1.95, Gates::PerHead, 2e-4);
    check(2, 12, 2, 8, 8, 6, 1.95, Gates::PerChannel, 2e-4);
}

/// The sequence need not be a multiple of the chunk length: the zero pad is an
/// identity in the forward, and its gradient must be sliced away, not folded in.
#[test]
fn matches_the_taped_path_with_a_partial_last_chunk() {
    check(2, 13, 2, 8, 8, 8, 0.95, Gates::PerHead, 1e-4);
    check(1, 5, 1, 4, 4, 8, 0.95, Gates::Ungated, 1e-4);
    check(2, 37, 2, 4, 6, 16, 0.95, Gates::PerChannel, 2e-4);
}

/// Asymmetric `head_k_dim` / `head_v_dim` (`expand_v ≠ 1`), the Gated DeltaNet
/// default: the state is rectangular and every transpose has to agree.
#[test]
fn matches_the_taped_path_with_expanded_values() {
    check(2, 16, 2, 8, 16, 8, 0.95, Gates::PerHead, 1e-4);
    check(2, 16, 2, 16, 8, 8, 0.95, Gates::PerChannel, 2e-4);
}

/// The chunk length is a performance knob, not a semantic one — including the
/// degenerate `chunk_len = 1` (an empty ladder, `T = I`) and lengths that do
/// not divide the reference-block size.
#[test]
fn every_chunk_length_agrees() {
    for chunk_len in [1, 2, 3, 6, 8, 12, 16, 24, 32] {
        check(2, 24, 2, 8, 8, chunk_len, 0.95, Gates::PerHead, 2e-4);
        check(2, 24, 2, 8, 8, chunk_len, 0.95, Gates::PerChannel, 2e-4);
    }
}

/// Without autodiff the node takes the trait's default body, which is the plain
/// forward — the same numbers the taped path produces.
#[test]
fn an_inference_pass_matches_the_taped_forward() {
    let device: Device = Default::default();
    let (batch, sequence, nheads, head_k_dim, head_v_dim) = (2, 20, 3, 8, 8);
    for gates in [Gates::Ungated, Gates::PerHead, Gates::PerChannel] {
        let raw = random_raw(
            batch, sequence, nheads, head_k_dim, head_v_dim, 0.95, gates, &device,
        );
        let input = || DeltaInput {
            q_bshk: raw.q.clone(),
            k_bshk: raw.k.clone(),
            v_bshv: raw.v.clone(),
            erase_bshK: raw.erase.clone(),
            write_bshV: raw.write.clone(),
            g_bshK: raw.g.clone(),
            state_bhkv: raw.state.clone(),
            scale: None,
        };
        let (y_taped, s_taped) = input().run(DeltaPath::Chunk {
            chunk_len: Some(8),
            solve: TriSolve::Blocked,
        });
        let (y_custom, s_custom) = input().run(DeltaPath::ChunkRecalculated {
            chunk_len: Some(8),
        });
        assert!(
            max_abs_diff(y_taped, y_custom) < 1e-5,
            "{gates:?}: y disagrees on the untracked path",
        );
        assert!(
            max_abs_diff(s_taped, s_custom) < 1e-5,
            "{gates:?}: final_state disagrees on the untracked path",
        );
    }
}
