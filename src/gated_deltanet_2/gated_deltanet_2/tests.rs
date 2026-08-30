//! GDN-2's contract, which is every other family's: `forward` over a sequence
//! is `step` unrolled over the same tokens from the same cache — on outputs, on
//! the resulting cache, and on gradients. Plus the one thing only this family
//! can do, checked on the recurrence directly.

use super::*;
use burn::tensor::Distribution;
use burn_stack::utils::test_helpers::max_abs_diff;

type Device = burn::prelude::Device;

fn tiny_config(d_model: usize) -> GatedDeltaNet2Config {
    GatedDeltaNet2Config::new(d_model)
        .with_nheads(2)
        .with_head_k_dim(8)
        .with_expand_v(1.0)
}

fn random_input(batch: usize, sequence: usize, d_model: usize, device: &Device) -> Tensor<3> {
    Tensor::random(
        [batch, sequence, d_model],
        Distribution::Normal(0.0, 1.0),
        device,
    )
}

fn unroll_steps(
    block: &GatedDeltaNet2,
    input_bsd: Tensor<3>,
    cache: Option<GatedDeltaNet2Cache>,
) -> (Tensor<3>, GatedDeltaNet2Cache) {
    let [_batch, sequence, _d_model] = input_bsd.dims();
    let mut cache = cache;
    let mut outputs = Vec::with_capacity(sequence);
    for t in 0..sequence {
        let x_bd = input_bsd.clone().narrow(1, t, 1).squeeze_dim(1);
        let (y_bd, next) = block.step(x_bd, cache.take());
        cache = Some(next);
        outputs.push(y_bd.unsqueeze_dim(1));
    }
    (Tensor::cat(outputs, 1), cache.expect("at least one step"))
}

fn assert_caches_match(
    a: &GatedDeltaNet2Cache,
    b: &GatedDeltaNet2Cache,
    label: &str,
    tol: f32,
) {
    let state_diff = max_abs_diff(a.state_bhkv.clone(), b.state_bhkv.clone());
    assert!(state_diff < tol, "{label}: state differs by {state_diff}");
    match (&a.conv_bwc, &b.conv_bwc) {
        (Some(a), Some(b)) => {
            let diff = max_abs_diff(a.clone(), b.clone());
            assert!(diff < tol, "{label}: conv window differs by {diff}");
        }
        (None, None) => {}
        _ => panic!("{label}: one cache has a convolution window and the other does not"),
    }
}

fn check_forward_matches_step(
    config: GatedDeltaNet2Config,
    sequence: usize,
    path: DeltaPath,
    tol: f32,
) {
    let device: Device = Default::default();
    let (batch, d_model) = (2, config.d_model);
    let block = config.init(&device);
    let input = random_input(batch, sequence, d_model, &device);

    let (y_forward, cache_forward) = block.forward(input.clone(), None, path);
    let (y_stepped, cache_stepped) = unroll_steps(&block, input, None);

    let diff = max_abs_diff(y_forward, y_stepped);
    assert!(diff < tol, "outputs differ by {diff} (path {path:?})");
    assert_caches_match(&cache_forward, &cache_stepped, "forward vs step", tol);
}

#[test]
fn forward_matches_step_default() {
    for path in [
        DeltaPath::Recurrent,
        DeltaPath::chunk_len(4),
        DeltaPath::chunk(),
    ] {
        check_forward_matches_step(tiny_config(16), 10, path, 1e-4);
    }
}

#[test]
fn forward_matches_step_ungated_output() {
    check_forward_matches_step(
        tiny_config(16).with_use_gate(false),
        10,
        DeltaPath::chunk_len(4),
        1e-4,
    );
}

#[test]
fn forward_matches_step_with_expanded_values() {
    check_forward_matches_step(
        tiny_config(16).with_expand_v(2.0),
        9,
        DeltaPath::chunk_len(4),
        1e-4,
    );
}

/// Grouped values: several value heads per query/key head.
#[test]
fn forward_matches_step_with_grouped_values() {
    check_forward_matches_step(
        tiny_config(16).with_n_value_heads(4),
        9,
        DeltaPath::chunk_len(4),
        1e-4,
    );
}

#[test]
fn forward_matches_step_without_a_convolution() {
    check_forward_matches_step(
        tiny_config(16).with_use_short_conv(false),
        8,
        DeltaPath::chunk_len(4),
        1e-4,
    );
}

#[test]
fn forward_matches_step_with_negative_eigenvalues() {
    check_forward_matches_step(
        tiny_config(16).with_allow_neg_eigval(true),
        10,
        DeltaPath::chunk_len(4),
        1e-3,
    );
}

/// A strong decay (large `|A|`, large `Δ`) is the regime the cumulative-gate
/// arithmetic is least forgiving in.
#[test]
fn forward_matches_step_under_a_fast_decay() {
    check_forward_matches_step(
        tiny_config(16)
            .with_a_init_range((8.0, 16.0))
            .with_dt_min(0.05)
            .with_dt_max(0.5)
            .with_lower_bound(-5.0),
        12,
        DeltaPath::chunk_len(4),
        1e-3,
    );
}

/// A **constant** input over a long sequence — a z-scored image background,
/// which is most of sequential MNIST.
///
/// Every key in a chunk then points the same way, which is the worst case for
/// the WY transform's `(I − N)⁻¹` (see [`crate::delta::tri`]). The two paths
/// evaluate the same recurrence, so at the default chunk length the block must
/// still track `DeltaPath::Recurrent` and its state must stay bounded.
#[test]
fn a_constant_input_stays_bounded_at_the_default_chunk_length() {
    let device: Device = Default::default();
    let (batch, sequence, d_model) = (1, 256, 32);
    let block = tiny_config(d_model).with_nheads(4).init(&device);
    let input = Tensor::<3>::full([batch, sequence, d_model], -0.424, &device);

    let (y_recurrent, cache_recurrent) = block.forward(input.clone(), None, DeltaPath::Recurrent);
    for path in [
        DeltaPath::chunk_len(32),
        DeltaPath::chunk_len(48),
        DeltaPath::chunk(),
        DeltaPath::chunk_len(128),
    ] {
        let (y, cache) = block.forward(input.clone(), None, path);
        let diff = max_abs_diff(y_recurrent.clone(), y);
        assert!(diff < 1e-4, "outputs differ by {diff} (path {path:?})");
        assert_caches_match(&cache_recurrent, &cache, &format!("{path:?}"), 1e-4);
    }
}

#[test]
fn split_forward_matches_a_single_forward() {
    let device: Device = Default::default();
    let (batch, sequence, d_model, split) = (2, 12, 16, 5);
    let block = tiny_config(d_model).init(&device);
    let input = random_input(batch, sequence, d_model, &device);
    let path = DeltaPath::chunk_len(4);

    let (y_whole, cache_whole) = block.forward(input.clone(), None, path);
    let (y_head, cache_mid) = block.forward(input.clone().narrow(1, 0, split), None, path);
    let (y_tail, cache_end) = block.forward(
        input.narrow(1, split, sequence - split),
        Some(cache_mid),
        path,
    );

    let diff = max_abs_diff(y_whole, Tensor::cat(vec![y_head, y_tail], 1));
    assert!(diff < 1e-4, "split prefill differs by {diff}");
    assert_caches_match(&cache_whole, &cache_end, "whole vs split", 1e-4);
}

#[test]
fn forward_and_step_agree_on_parameter_gradients() {
    let device: Device = Device::default().autodiff();
    let (batch, sequence, d_model) = (2, 8, 16);
    let block = tiny_config(d_model).init(&device);
    let input = random_input(batch, sequence, d_model, &device);
    let head = random_input(batch, sequence, d_model, &device);

    let grads_of = |y: Tensor<3>| {
        let grads = (y * head.clone()).sum().backward();
        (
            block.qkv.in_proj.weight.val().grad(&grads).expect("in_proj"),
            block.out_proj.weight.val().grad(&grads).expect("out_proj"),
            block.gate.a_log_h.val().grad(&grads).expect("a_log"),
            block.gate.dt_bias_i.val().grad(&grads).expect("dt_bias"),
            block.gate.up.weight.val().grad(&grads).expect("gate.up"),
        )
    };

    let (f_in, f_out, f_a, f_dt, f_up) =
        grads_of(block.forward(input.clone(), None, DeltaPath::chunk_len(4)).0);
    let (s_in, s_out, s_a, s_dt, s_up) = grads_of(unroll_steps(&block, input, None).0);

    for (name, diff) in [
        ("in_proj", max_abs_diff(f_in, s_in)),
        ("out_proj", max_abs_diff(f_out, s_out)),
        ("a_log", max_abs_diff(f_a, s_a)),
        ("dt_bias", max_abs_diff(f_dt, s_dt)),
        ("gate.up", max_abs_diff(f_up, s_up)),
    ] {
        assert!(diff < 1e-3, "grad of {name} differs by {diff}");
    }
}

/// /// `g = lower_bound · σ(·)` lands in `(lower_bound, 0)` by construction, so the
/// decay `α = exp(g)` can neither exceed 1 — the state cannot grow through the
/// gate however the parameters move — nor run away below, which is what the
/// chunk path's factored decay needs.
#[test]
fn the_forget_gate_never_amplifies() {
    let device: Device = Default::default();
    let block = tiny_config(16).init(&device);
    let raw = Tensor::<3>::random(
        [4, 32, block.bottleneck()],
        Distribution::Normal(0.0, 10.0),
        &device,
    );
    let g: Tensor<4> = block.gate.log_decay::<3, 4>(raw);
    assert_eq!(
        [4, 32, block.n_qk_heads(), block.head_k_dim()],
        g.dims(),
        "the decay is per key channel, not per head",
    );
    assert!(
        g.clone().max().into_scalar::<f32>() <= 0.0,
        "log decay must be non-positive",
    );
    assert!(
        g.min().into_scalar::<f32>() >= block.gate.lower_bound as f32,
        "log decay must stay above the gate's lower bound",
    );
}

/// An explicit bottleneck rank, decoupled from `head_v_dim`.
#[test]
fn forward_matches_step_with_a_custom_bottleneck() {
    check_forward_matches_step(
        tiny_config(16).with_bottleneck(5),
        9,
        DeltaPath::chunk_len(4),
        1e-4,
    );
}

#[test]
fn output_shape_is_d_model() {
    let device: Device = Default::default();
    let (batch, sequence, d_model) = (3, 7, 16);
    let block = tiny_config(d_model)
        .with_expand_v(2.0)
        .with_n_value_heads(4)
        .init(&device);
    let (y, cache) = block.forward(
        random_input(batch, sequence, d_model, &device),
        None,
        DeltaPath::chunk(),
    );
    assert_eq!([batch, sequence, d_model], y.dims());
    assert_eq!([batch, 4, 8, 16], cache.state_bhkv.dims());
}

/// The capability the family exists for, checked on the recurrence itself.
///
/// Write `v` at a unit key, then ask to *add* `v` again. With decoupled gates
/// that is `b = 0, w = 1`: erase nothing, commit `v`, and the key now holds
/// `2v`. A shared `β` cannot express it at any value of `β` — the write it
/// commits and the erase it performs scale together, so with `v` already
/// stored the delta `β(v − Sᵀk)` is identically zero and the state does not
/// move. The compensation a scalar gate would need (`v + Sᵀk`) depends on what
/// the state currently holds, which no projection of the token can know.
#[test]
fn only_decoupled_gates_can_accumulate_onto_an_occupied_key() {
    use crate::delta::recurrent::delta_step;

    let device: Device = Default::default();
    let (batch, nheads, head_k_dim, head_v_dim) = (1, 1, 4, 3);
    let one = |shape: [usize; 3]| Tensor::<3>::ones(Shape::new(shape), &device);
    let zero = |shape: [usize; 3]| Tensor::<3>::zeros(Shape::new(shape), &device);

    // A unit key, so the readout at `q = k` returns exactly what was stored.
    let k = crate::common::norm::l2_normalize(Tensor::<3>::random(
        [batch, nheads, head_k_dim],
        Distribution::Normal(0.0, 1.0),
        &device,
    ));
    let v = Tensor::<3>::random(
        [batch, nheads, head_v_dim],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let state = Tensor::<4>::zeros(
        Shape::new([batch, nheads, head_k_dim, head_v_dim]),
        &device,
    );

    // Replace: b = w = 1 stores `v` at `k`.
    let (stored, state) = delta_step(
        k.clone(),
        k.clone(),
        v.clone(),
        one([batch, nheads, 1]),
        one([batch, nheads, 1]),
        None,
        state,
        1.0,
    );
    assert!(max_abs_diff(stored, v.clone()) < 1e-5, "the write did not land");

    // Accumulate: b = 0, w = 1.
    let (accumulated, _) = delta_step(
        k.clone(),
        k.clone(),
        v.clone(),
        zero([batch, nheads, 1]),
        one([batch, nheads, 1]),
        None,
        state.clone(),
        1.0,
    );
    let doubled = max_abs_diff(accumulated, v.clone() * 2.0);
    assert!(doubled < 1e-5, "b = 0, w = 1 must add: off by {doubled}");

    // A shared β leaves the state exactly where it is, whatever β is.
    for beta in [0.25, 0.5, 1.0, 2.0] {
        let (scalar, _) = delta_step(
            k.clone(),
            k.clone(),
            v.clone(),
            one([batch, nheads, 1]) * beta,
            one([batch, nheads, 1]) * beta,
            None,
            state.clone(),
            1.0,
        );
        let moved = max_abs_diff(scalar, v.clone());
        assert!(moved < 1e-5, "β = {beta} moved the stored value by {moved}");
    }
}
