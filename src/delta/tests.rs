//! The delta-rule core's contract: every [`DeltaPath`] evaluates the *same*
//! recurrence, so all of them must agree on outputs, final state and gradients.
//! [`DeltaPath::Recurrent`] is the definition and therefore the baseline.

use super::path::{DeltaInput, DeltaPath};
use super::tri::TriSolve;
use crate::common::norm::l2_normalize;
use burn::module::Param;
use burn::prelude::*;
use burn::tensor::Distribution;
use burn_stack::utils::test_helpers::max_abs_diff;

type Device = burn::prelude::Device;

/// Every chunked path at `chunk_len`: both triangular solves on the tape, plus
/// the custom-backward one. They compute the same function, so each is checked
/// against the recurrent baseline on values *and* gradients.
fn chunk_paths(chunk_len: usize) -> [DeltaPath; 3] {
    [
        DeltaPath::Chunk {
            chunk_len: Some(chunk_len),
            solve: TriSolve::Blocked,
        },
        DeltaPath::Chunk {
            chunk_len: Some(chunk_len),
            solve: TriSolve::Neumann,
        },
        DeltaPath::ChunkRecalculated {
            chunk_len: Some(chunk_len),
        },
    ]
}

struct Raw {
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    /// `[batch, sequence, nheads, 1 | head_k_dim]`
    erase: Tensor<4>,
    /// `[batch, sequence, nheads, 1 | head_v_dim]`
    write: Tensor<4>,
    /// `[batch, sequence, nheads, 1 | head_k_dim]`
    g: Option<Tensor<4>>,
    state: Tensor<4>,
}

/// Inputs shaped exactly as a block would hand them over: `q`/`k` already
/// L2-normalised, `β` already squashed into its range, `g ≤ 0`.
fn random_raw(
    batch: usize,
    sequence: usize,
    nheads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    beta_max: f64,
    gated: bool,
    random_state: bool,
    device: &Device,
) -> Raw {
    random_raw_gates(
        batch, sequence, nheads, head_k_dim, head_v_dim, beta_max, gated, random_state, false,
        device,
    )
}

/// As [`random_raw`], but `channel` widens every gate onto its channel axis —
/// GDN-2's shape rather than the scalar families'.
#[allow(clippy::too_many_arguments)]
fn random_raw_gates(
    batch: usize,
    sequence: usize,
    nheads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    beta_max: f64,
    gated: bool,
    random_state: bool,
    channel: bool,
    device: &Device,
) -> Raw {
    let normal = Distribution::Normal(0.0, 1.0);
    let q = l2_normalize(Tensor::<4>::random(
        [batch, sequence, nheads, head_k_dim],
        normal,
        device,
    ));
    let k = l2_normalize(Tensor::<4>::random(
        [batch, sequence, nheads, head_k_dim],
        normal,
        device,
    ));
    let v = Tensor::<4>::random([batch, sequence, nheads, head_v_dim], normal, device);
    // Width 1 = one number per head; the full channel count = GDN-2.
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
    let write = if channel {
        Tensor::<4>::random(
            [batch, sequence, nheads, v_gate],
            Distribution::Uniform(0.05, 1.0),
            device,
        )
    } else {
        erase.clone()
    };
    // `g = log α ≤ 0`: a per-step decay in roughly (0.6, 1).
    let g = gated.then(|| {
        Tensor::<4>::random(
            [batch, sequence, nheads, k_gate],
            Distribution::Uniform(-0.5, -0.01),
            device,
        )
    });
    let state = if random_state {
        Tensor::<4>::random(
            [batch, nheads, head_k_dim, head_v_dim],
            Distribution::Normal(0.0, 0.2),
            device,
        )
    } else {
        Tensor::<4>::zeros([batch, nheads, head_k_dim, head_v_dim], device)
    };
    Raw {
        q,
        k,
        v,
        erase,
        write,
        g,
        state,
    }
}

/// One autodiff graph over the same underlying values.
struct Leaves {
    q: Param<Tensor<4>>,
    k: Param<Tensor<4>>,
    v: Param<Tensor<4>>,
    erase: Param<Tensor<4>>,
    write: Param<Tensor<4>>,
    g: Option<Param<Tensor<4>>>,
    state: Param<Tensor<4>>,
}

impl Leaves {
    fn from_raw(raw: &Raw) -> Self {
        let lift4 = |t: &Tensor<4>| Param::from_tensor(Tensor::from_inner(t.clone()));
        Self {
            q: lift4(&raw.q),
            k: lift4(&raw.k),
            v: lift4(&raw.v),
            erase: lift4(&raw.erase),
            write: lift4(&raw.write),
            g: raw.g.as_ref().map(lift4),
            state: lift4(&raw.state),
        }
    }

    fn input(&self) -> DeltaInput {
        DeltaInput {
            q_bshk: self.q.val(),
            k_bshk: self.k.val(),
            v_bshv: self.v.val(),
            erase_bshK: self.erase.val(),
            write_bshV: self.write.val(),
            g_bshK: self.g.as_ref().map(|g| g.val()),
            state_bhkv: self.state.val(),
            scale: None,
        }
    }
}

struct PathRun {
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

/// Two distinct fixed heads so the `y` and `final_state` gradient paths are
/// exercised independently.
fn run_path(path: DeltaPath, raw: &Raw, y_head: &Tensor<4>, s_head: &Tensor<4>) -> PathRun {
    let leaves = Leaves::from_raw(raw);
    let (y, state) = leaves.input().run(path);
    let y_inner = y.clone().inner();
    let state_inner = state.clone().inner();

    let loss = (y * Tensor::from_inner(y_head.clone())).sum()
        + (state * Tensor::from_inner(s_head.clone())).sum();
    let grads = loss.backward();

    PathRun {
        y: y_inner,
        state: state_inner,
        d_q: leaves.q.val().grad(&grads).expect("grad q"),
        d_k: leaves.k.val().grad(&grads).expect("grad k"),
        d_v: leaves.v.val().grad(&grads).expect("grad v"),
        d_erase: leaves.erase.val().grad(&grads).expect("grad erase"),
        d_write: leaves.write.val().grad(&grads).expect("grad write"),
        d_g: leaves
            .g
            .as_ref()
            .map(|g| g.val().grad(&grads).expect("grad g")),
        d_state: leaves.state.val().grad(&grads).expect("grad state"),
    }
}

fn assert_runs_match(baseline: &PathRun, other: &PathRun, label: &str, tol: f32) {
    let mut failures = Vec::new();
    let mut check = |name: &str, diff: f32| {
        if !(diff < tol) {
            failures.push(format!("{name}: max abs diff {diff} (tol {tol})"));
        }
    };
    check("y", max_abs_diff(baseline.y.clone(), other.y.clone()));
    check(
        "final_state",
        max_abs_diff(baseline.state.clone(), other.state.clone()),
    );
    check("d_q", max_abs_diff(baseline.d_q.clone(), other.d_q.clone()));
    check("d_k", max_abs_diff(baseline.d_k.clone(), other.d_k.clone()));
    check("d_v", max_abs_diff(baseline.d_v.clone(), other.d_v.clone()));
    check(
        "d_erase",
        max_abs_diff(baseline.d_erase.clone(), other.d_erase.clone()),
    );
    check(
        "d_write",
        max_abs_diff(baseline.d_write.clone(), other.d_write.clone()),
    );
    check(
        "d_state",
        max_abs_diff(baseline.d_state.clone(), other.d_state.clone()),
    );
    if let (Some(a), Some(b)) = (&baseline.d_g, &other.d_g) {
        check("d_g", max_abs_diff(a.clone(), b.clone()));
    }
    assert!(
        failures.is_empty(),
        "Recurrent vs {label} disagree:\n  {}",
        failures.join("\n  "),
    );
}

#[allow(clippy::too_many_arguments)]
fn check_chunk_matches_recurrent(
    batch: usize,
    sequence: usize,
    nheads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    chunk_len: usize,
    beta_max: f64,
    gated: bool,
    random_state: bool,
    tol: f32,
) {
    let device: Device = Default::default();
    let raw = random_raw(
        batch,
        sequence,
        nheads,
        head_k_dim,
        head_v_dim,
        beta_max,
        gated,
        random_state,
        &device,
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

    let baseline = run_path(DeltaPath::Recurrent, &raw, &y_head, &s_head);
    for path in chunk_paths(chunk_len) {
        let run = run_path(path, &raw, &y_head, &s_head);
        assert_runs_match(&baseline, &run, &format!("{path:?}"), tol);
    }
}

#[test]
fn chunk_matches_recurrent_ungated() {
    check_chunk_matches_recurrent(2, 16, 3, 8, 8, 8, 0.95, false, false, 1e-3);
}

#[test]
fn chunk_matches_recurrent_gated() {
    check_chunk_matches_recurrent(2, 16, 3, 8, 8, 8, 0.95, true, false, 1e-3);
}

/// A non-zero incoming state is the streaming case: the chunk path reaches it
/// through `W S₀` and the decayed read, the recurrent path through the loop.
#[test]
fn chunk_matches_recurrent_from_a_carried_state() {
    check_chunk_matches_recurrent(2, 12, 2, 8, 16, 4, 0.95, true, true, 1e-3);
}

/// `β > 1` puts the Householder's second eigenvalue below zero — the
/// state-tracking regime, and the one most likely to expose a sign error.
#[test]
fn chunk_matches_recurrent_with_negative_eigenvalues() {
    check_chunk_matches_recurrent(2, 12, 2, 8, 8, 6, 1.95, true, true, 2e-3);
}

/// The sequence need not be a multiple of the chunk length: the pad must be an
/// exact identity.
#[test]
fn chunk_matches_recurrent_with_a_partial_last_chunk() {
    check_chunk_matches_recurrent(2, 13, 2, 8, 8, 8, 0.95, true, true, 1e-3);
    check_chunk_matches_recurrent(1, 5, 1, 4, 4, 8, 0.95, false, false, 1e-3);
}

/// Asymmetric `head_k_dim` / `head_v_dim` (`expand_v ≠ 1`), which is the Gated
/// DeltaNet default.
#[test]
fn chunk_matches_recurrent_with_expanded_values() {
    check_chunk_matches_recurrent(2, 16, 2, 8, 16, 8, 0.95, true, true, 1e-3);
}

/// The chunk length is a performance knob, not a semantic one.
#[test]
fn chunk_len_does_not_change_the_answer() {
    let device: Device = Default::default();
    let (batch, sequence, nheads, head_k_dim, head_v_dim) = (2, 24, 2, 8, 8);
    let raw = random_raw(
        batch, sequence, nheads, head_k_dim, head_v_dim, 0.95, true, true, &device,
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

    let baseline = run_path(DeltaPath::chunk_len(4), &raw, &y_head, &s_head);
    for chunk_len in [3, 6, 8, 12, 24, 32] {
        let run = run_path(DeltaPath::chunk_len(chunk_len), &raw, &y_head, &s_head);
        assert_runs_match(&baseline, &run, &format!("Chunk({chunk_len})"), 1e-3);
    }
}

/// Inputs that never change over the sequence — the worst case for the WY
/// transform, and what a constant image background is after the projections.
///
/// Every `k` in a chunk then points the same way, so `N[i, j] = −βᵢ(kᵢ·kⱼ)` is
/// `−β` on the whole strict lower triangle: `I − N` is the (scaled) prefix-sum
/// matrix, whose inverse is a *differencing* matrix and perfectly benign — but
/// whose Neumann series is not, its partial sums growing like the central
/// binomial coefficient before cancelling back down.
fn constant_raw(
    batch: usize,
    sequence: usize,
    nheads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    beta: f64,
    gated: bool,
    device: &Device,
) -> Raw {
    let normal = Distribution::Normal(0.0, 1.0);
    // One direction per (batch, head), held for the whole sequence.
    let hold = |t: Tensor<4>, last: usize| t.expand([batch, sequence, nheads, last]);
    let q = hold(
        l2_normalize(Tensor::<4>::random(
            [batch, 1, nheads, head_k_dim],
            normal,
            device,
        )),
        head_k_dim,
    );
    let k = hold(
        l2_normalize(Tensor::<4>::random(
            [batch, 1, nheads, head_k_dim],
            normal,
            device,
        )),
        head_k_dim,
    );
    let v = hold(
        Tensor::<4>::random([batch, 1, nheads, head_v_dim], normal, device),
        head_v_dim,
    );
    let erase = Tensor::<4>::full([batch, sequence, nheads, 1], beta, device);
    let g = gated.then(|| Tensor::<4>::full([batch, sequence, nheads, 1], -0.05, device));
    Raw {
        q,
        k,
        v,
        erase: erase.clone(),
        write: erase,
        g,
        state: Tensor::<4>::zeros([batch, nheads, head_k_dim, head_v_dim], device),
    }
}

/// The chunk path must survive a constant input at every chunk length — this is
/// the sequential-MNIST case (a mostly constant background over 784 tokens),
/// where a series-summed `(I − N)⁻¹` loses every float32 digit by `L ≈ 32` and
/// the block diverges to `NaN` within a couple of optimiser steps.
#[test]
fn chunk_matches_recurrent_on_a_constant_input() {
    let device: Device = Default::default();
    let (batch, sequence, nheads, head_k_dim, head_v_dim) = (2, 128, 2, 8, 8);
    for (beta, gated) in [(0.9, false), (0.9, true), (1.8, true)] {
        let raw = constant_raw(
            batch,
            sequence,
            nheads,
            head_k_dim,
            head_v_dim,
            beta,
            gated,
            &device,
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

        let baseline = run_path(DeltaPath::Recurrent, &raw, &y_head, &s_head);
        for chunk_len in [16, 32, 48, 64, 128] {
            let run = run_path(DeltaPath::chunk_len(chunk_len), &raw, &y_head, &s_head);
            assert_runs_match(
                &baseline,
                &run,
                &format!("Chunk({chunk_len}) on a constant input (beta {beta}, gated {gated})"),
                1e-3,
            );
        }
    }
}

/// Streaming: a sequence split across two calls, threading the state, equals
/// one call over the whole thing. This is the property every cache in the crate
/// relies on.
#[test]
fn split_calls_match_a_single_call() {
    let device: Device = Default::default();
    let (batch, sequence, nheads, head_k_dim, head_v_dim) = (2, 20, 2, 8, 8);
    let split = 7;
    let raw = random_raw(
        batch, sequence, nheads, head_k_dim, head_v_dim, 0.95, true, false, &device,
    );

    let make = |from: usize, len: usize, state: Tensor<4>| DeltaInput {
        q_bshk: raw.q.clone().narrow(1, from, len),
        k_bshk: raw.k.clone().narrow(1, from, len),
        v_bshv: raw.v.clone().narrow(1, from, len),
        erase_bshK: raw.erase.clone().narrow(1, from, len),
        write_bshV: raw.write.clone().narrow(1, from, len),
        g_bshK: raw.g.as_ref().map(|g| g.clone().narrow(1, from, len)),
        state_bhkv: state,
        scale: None,
    };

    let path = DeltaPath::chunk_len(4);
    let (y_whole, state_whole) = make(0, sequence, raw.state.clone()).run(path);

    let (y_head, state_mid) = make(0, split, raw.state.clone()).run(path);
    let (y_tail, state_end) = make(split, sequence - split, state_mid).run(path);
    let y_split = Tensor::cat(vec![y_head, y_tail], 1);

    assert!(max_abs_diff(y_whole, y_split) < 1e-4);
    assert!(max_abs_diff(state_whole, state_end) < 1e-4);
}

// ---------------------------------------------------------------------------
// Cross-check against the reference implementation
// ---------------------------------------------------------------------------

mod reference;

/// Rebuild a fixture tensor at the backend's working precision.
fn from_fixture<const D: usize>(data: &[f64], shape: [usize; D], device: &Device) -> Tensor<D> {
    assert_eq!(data.len(), shape.iter().product::<usize>());
    let values: Vec<f32> = data.iter().map(|v| *v as f32).collect();
    Tensor::from_data(burn::tensor::TensorData::new(values, shape), device)
}

fn fixture_input(gated: bool, device: &Device) -> DeltaInput {
    let [batch, sequence, nheads, head_k_dim, head_v_dim] = reference::DIMS;
    let beta = from_fixture(reference::IN_BETA, [batch, sequence, nheads, 1], device);
    DeltaInput {
        q_bshk: from_fixture(reference::IN_Q, [batch, sequence, nheads, head_k_dim], device),
        k_bshk: from_fixture(reference::IN_K, [batch, sequence, nheads, head_k_dim], device),
        v_bshv: from_fixture(reference::IN_V, [batch, sequence, nheads, head_v_dim], device),
        erase_bshK: beta.clone(),
        write_bshV: beta,
        g_bshK: gated.then(|| {
            from_fixture(reference::IN_G, [batch, sequence, nheads, 1], device)
        }),
        state_bhkv: from_fixture(
            reference::IN_STATE,
            [batch, nheads, head_k_dim, head_v_dim],
            device,
        ),
        scale: None,
    }
}

/// Every path must reproduce `flash-linear-attention`'s own naive delta rule on
/// a fixed input — the check that this is a port and not merely a
/// self-consistent invention.
fn check_matches_reference(gated: bool, expected_y: &[f64], expected_state: &[f64]) {
    let device: Device = Default::default();
    let [batch, sequence, nheads, head_k_dim, head_v_dim] = reference::DIMS;
    let want_y = from_fixture(expected_y, [batch, sequence, nheads, head_v_dim], &device);
    let want_state = from_fixture(
        expected_state,
        [batch, nheads, head_k_dim, head_v_dim],
        &device,
    );

    for path in [
        DeltaPath::Recurrent,
        DeltaPath::chunk_len(3),
        DeltaPath::chunk_len(4),
        DeltaPath::chunk_len(16),
    ] {
        let (y, state) = fixture_input(gated, &device).run(path);
        let y_diff = max_abs_diff(y, want_y.clone());
        let state_diff = max_abs_diff(state, want_state.clone());
        assert!(y_diff < 1e-4, "{path:?}: output differs from the reference by {y_diff}");
        assert!(
            state_diff < 1e-4,
            "{path:?}: final state differs from the reference by {state_diff}",
        );
    }
}

#[test]
fn matches_the_reference_delta_rule() {
    check_matches_reference(false, reference::UNGATED_Y, reference::UNGATED_STATE);
}

#[test]
fn matches_the_reference_gated_delta_rule() {
    check_matches_reference(true, reference::GATED_Y, reference::GATED_STATE);
}

/// The gated recurrence *is* the ungated one at `g = 0`. That is why one
/// implementation serves both families — and why `g: None` is only ever an
/// optimisation, never a different function.
#[test]
fn a_zero_gate_is_the_ungated_delta_rule() {
    let device: Device = Default::default();
    let (batch, sequence, nheads, head_k_dim, head_v_dim) = (2, 12, 2, 8, 8);
    let raw = random_raw(
        batch, sequence, nheads, head_k_dim, head_v_dim, 0.95, false, true, &device,
    );

    let input = |g: Option<Tensor<4>>| DeltaInput {
        q_bshk: raw.q.clone(),
        k_bshk: raw.k.clone(),
        v_bshv: raw.v.clone(),
        erase_bshK: raw.erase.clone(),
        write_bshV: raw.write.clone(),
        g_bshK: g,
        state_bhkv: raw.state.clone(),
        scale: None,
    };
    let zeros = Tensor::zeros([batch, sequence, nheads, 1], &device);

    for path in [DeltaPath::Recurrent, DeltaPath::chunk_len(5)] {
        let (y_none, state_none) = input(None).run(path);
        let (y_zero, state_zero) = input(Some(zeros.clone())).run(path);
        assert!(max_abs_diff(y_none, y_zero) < 1e-5, "{path:?}: outputs differ");
        assert!(
            max_abs_diff(state_none, state_zero) < 1e-5,
            "{path:?}: states differ",
        );
    }
}

// ---------------------------------------------------------------------------
// Channel-wise gates (GDN-2)
// ---------------------------------------------------------------------------

/// Same contract, GDN-2's gate widths: the chunked path must still equal the
/// recurrent definition on values *and* gradients. The chunk path takes a
/// genuinely different route here — the decay no longer factors out of the key
/// contraction, so the score matrices go through
/// [`decay::BlockDecay`](crate::delta::decay).
#[allow(clippy::too_many_arguments)]
fn check_channel_chunk_matches_recurrent(
    batch: usize,
    sequence: usize,
    nheads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    chunk_len: usize,
    beta_max: f64,
    random_state: bool,
    tol: f32,
) {
    let device: Device = Default::default();
    let raw = random_raw_gates(
        batch,
        sequence,
        nheads,
        head_k_dim,
        head_v_dim,
        beta_max,
        true,
        random_state,
        true,
        &device,
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

    let baseline = run_path(DeltaPath::Recurrent, &raw, &y_head, &s_head);
    for path in chunk_paths(chunk_len) {
        let run = run_path(path, &raw, &y_head, &s_head);
        assert_runs_match(
            &baseline,
            &run,
            &format!("{path:?} with channel gates"),
            tol,
        );
    }
}

#[test]
fn chunk_matches_recurrent_with_channel_gates() {
    check_channel_chunk_matches_recurrent(2, 16, 3, 8, 8, 8, 0.95, false, 1e-3);
}

#[test]
fn chunk_matches_recurrent_with_channel_gates_from_a_carried_state() {
    check_channel_chunk_matches_recurrent(2, 12, 2, 8, 16, 4, 0.95, true, 1e-3);
}

/// A chunk longer than one reference block, so the block-decomposed score
/// matrices are exercised with more than one block *and* a partial last chunk.
#[test]
fn chunk_matches_recurrent_with_channel_gates_across_reference_blocks() {
    check_channel_chunk_matches_recurrent(2, 40, 2, 8, 8, 32, 0.95, true, 2e-3);
    check_channel_chunk_matches_recurrent(1, 37, 1, 4, 6, 16, 1.95, true, 2e-3);
}

/// The chunk length remains a performance knob under channel gates — including
/// across the reference-block boundary, which is where a wrong reference point
/// would show up.
#[test]
fn channel_chunk_len_does_not_change_the_answer() {
    let device: Device = Default::default();
    let (batch, sequence, nheads, head_k_dim, head_v_dim) = (2, 24, 2, 8, 8);
    let raw = random_raw_gates(
        batch, sequence, nheads, head_k_dim, head_v_dim, 0.95, true, true, true, &device,
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

    let baseline = run_path(DeltaPath::Recurrent, &raw, &y_head, &s_head);
    for chunk_len in [3, 6, 8, 12, 16, 24, 32] {
        let run = run_path(DeltaPath::chunk_len(chunk_len), &raw, &y_head, &s_head);
        assert_runs_match(&baseline, &run, &format!("Chunk({chunk_len})"), 2e-3);
    }
}

/// Widening a per-head `β` and a per-head `g` onto their channel axes changes
/// nothing: the scalar rule *is* the channel rule with constant gates. This is
/// what lets one core serve both, and what makes GDN-2 a strict generalisation
/// of Gated DeltaNet rather than a separate recurrence.
#[test]
fn broadcast_channel_gates_are_the_scalar_delta_rule() {
    let device: Device = Default::default();
    let (batch, sequence, nheads, head_k_dim, head_v_dim) = (2, 20, 2, 8, 12);
    let raw = random_raw(
        batch, sequence, nheads, head_k_dim, head_v_dim, 0.95, true, true, &device,
    );

    // The same numbers, materialised across every channel.
    let widen = |t: &Tensor<4>, width: usize| -> Tensor<4> {
        t.clone() + Tensor::zeros([batch, sequence, nheads, width], &device)
    };
    let widened = Raw {
        q: raw.q.clone(),
        k: raw.k.clone(),
        v: raw.v.clone(),
        erase: widen(&raw.erase, head_k_dim),
        write: widen(&raw.write, head_v_dim),
        g: raw.g.as_ref().map(|g| widen(g, head_k_dim)),
        state: raw.state.clone(),
    };

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

    for path in [DeltaPath::Recurrent, DeltaPath::chunk_len(8)] {
        let scalar = run_path(path, &raw, &y_head, &s_head);
        let channel = run_path(path, &widened, &y_head, &s_head);
        assert!(
            max_abs_diff(scalar.y, channel.y) < 1e-4,
            "{path:?}: outputs differ",
        );
        assert!(
            max_abs_diff(scalar.state, channel.state) < 1e-4,
            "{path:?}: final states differ",
        );
    }
}

/// GDN-2 against `flash-linear-attention`'s own `naive_recurrent_gdn2` on the
/// same fixed input the scalar families are pinned to.
#[test]
fn matches_the_reference_gdn2() {
    let device: Device = Default::default();
    let [batch, sequence, nheads, head_k_dim, head_v_dim] = reference::DIMS;
    let want_y = from_fixture(
        reference::GDN2_Y,
        [batch, sequence, nheads, head_v_dim],
        &device,
    );
    let want_state = from_fixture(
        reference::GDN2_STATE,
        [batch, nheads, head_k_dim, head_v_dim],
        &device,
    );

    let input = || DeltaInput {
        q_bshk: from_fixture(reference::IN_Q, [batch, sequence, nheads, head_k_dim], &device),
        k_bshk: from_fixture(reference::IN_K, [batch, sequence, nheads, head_k_dim], &device),
        v_bshv: from_fixture(reference::IN_V, [batch, sequence, nheads, head_v_dim], &device),
        erase_bshK: from_fixture(reference::IN_B, [batch, sequence, nheads, head_k_dim], &device),
        write_bshV: from_fixture(reference::IN_W, [batch, sequence, nheads, head_v_dim], &device),
        g_bshK: Some(from_fixture(
            reference::IN_GK,
            [batch, sequence, nheads, head_k_dim],
            &device,
        )),
        state_bhkv: from_fixture(
            reference::IN_STATE,
            [batch, nheads, head_k_dim, head_v_dim],
            &device,
        ),
        scale: None,
    };

    for path in [
        DeltaPath::Recurrent,
        DeltaPath::chunk_len(3),
        DeltaPath::chunk_len(4),
        DeltaPath::chunk_len(16),
    ] {
        let (y, state) = input().run(path);
        let y_diff = max_abs_diff(y, want_y.clone());
        let state_diff = max_abs_diff(state, want_state.clone());
        assert!(
            y_diff < 1e-4,
            "{path:?}: output differs from the reference by {y_diff}",
        );
        assert!(
            state_diff < 1e-4,
            "{path:?}: final state differs from the reference by {state_diff}",
        );
    }
}
