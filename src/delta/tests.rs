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

struct Raw {
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    beta: Tensor<3>,
    g: Option<Tensor<3>>,
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
    let beta = Tensor::<3>::random(
        [batch, sequence, nheads],
        Distribution::Uniform(0.05, beta_max),
        device,
    );
    // `g = log α ≤ 0`: a per-step decay in roughly (0.6, 1).
    let g = gated.then(|| {
        Tensor::<3>::random(
            [batch, sequence, nheads],
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
        beta,
        g,
        state,
    }
}

/// One autodiff graph over the same underlying values.
struct Leaves {
    q: Param<Tensor<4>>,
    k: Param<Tensor<4>>,
    v: Param<Tensor<4>>,
    beta: Param<Tensor<3>>,
    g: Option<Param<Tensor<3>>>,
    state: Param<Tensor<4>>,
}

impl Leaves {
    fn from_raw(raw: &Raw) -> Self {
        let lift4 = |t: &Tensor<4>| Param::from_tensor(Tensor::from_inner(t.clone()));
        let lift3 = |t: &Tensor<3>| Param::from_tensor(Tensor::from_inner(t.clone()));
        Self {
            q: lift4(&raw.q),
            k: lift4(&raw.k),
            v: lift4(&raw.v),
            beta: lift3(&raw.beta),
            g: raw.g.as_ref().map(lift3),
            state: lift4(&raw.state),
        }
    }

    fn input(&self) -> DeltaInput {
        DeltaInput {
            q_bshk: self.q.val(),
            k_bshk: self.k.val(),
            v_bshv: self.v.val(),
            beta_bsh: self.beta.val(),
            g_bsh: self.g.as_ref().map(|g| g.val()),
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
    d_beta: Tensor<3>,
    d_g: Option<Tensor<3>>,
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
        d_beta: leaves.beta.val().grad(&grads).expect("grad beta"),
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
        "d_beta",
        max_abs_diff(baseline.d_beta.clone(), other.d_beta.clone()),
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
    for solve in [TriSolve::Doubling, TriSolve::Neumann] {
        let run = run_path(
            DeltaPath::Chunk {
                chunk_len: Some(chunk_len),
                solve,
            },
            &raw,
            &y_head,
            &s_head,
        );
        assert_runs_match(&baseline, &run, &format!("Chunk({chunk_len}, {solve:?})"), tol);
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
        beta_bsh: raw.beta.clone().narrow(1, from, len),
        g_bsh: raw.g.as_ref().map(|g| g.clone().narrow(1, from, len)),
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
