//! Gated DeltaNet's contract, which is DeltaNet's: `forward` over a sequence is
//! `step` unrolled over the same tokens from the same cache — on outputs, on
//! the resulting cache, and on gradients.

use super::*;
use burn::tensor::Distribution;
use burn_stack::utils::test_helpers::max_abs_diff;

type Device = burn::prelude::Device;

fn tiny_config(d_model: usize) -> GatedDeltaNetConfig {
    GatedDeltaNetConfig::new(d_model)
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
    block: &GatedDeltaNet,
    input_bsd: Tensor<3>,
    cache: Option<GatedDeltaNetCache>,
) -> (Tensor<3>, GatedDeltaNetCache) {
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
    a: &GatedDeltaNetCache,
    b: &GatedDeltaNetCache,
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
    config: GatedDeltaNetConfig,
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
            .with_dt_max(0.5),
        12,
        DeltaPath::chunk_len(4),
        1e-3,
    );
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
            block.gate.dt_bias_h.val().grad(&grads).expect("dt_bias"),
        )
    };

    let (f_in, f_out, f_a, f_dt) =
        grads_of(block.forward(input.clone(), None, DeltaPath::chunk_len(4)).0);
    let (s_in, s_out, s_a, s_dt) = grads_of(unroll_steps(&block, input, None).0);

    for (name, diff) in [
        ("in_proj", max_abs_diff(f_in, s_in)),
        ("out_proj", max_abs_diff(f_out, s_out)),
        ("a_log", max_abs_diff(f_a, s_a)),
        ("dt_bias", max_abs_diff(f_dt, s_dt)),
    ] {
        assert!(diff < 1e-3, "grad of {name} differs by {diff}");
    }
}

/// `A = −exp(a_log)` is negative by construction, so `g = Δ·A ≤ 0` and the
/// decay `α = exp(g)` can never exceed 1 — the state cannot grow through the
/// gate however the parameters move.
#[test]
fn the_forget_gate_never_amplifies() {
    let device: Device = Default::default();
    let block = tiny_config(16).with_a_init_range((0.0, 16.0)).init(&device);
    let dt_raw = Tensor::<3>::random(
        [4, 32, block.nheads()],
        Distribution::Normal(0.0, 10.0),
        &device,
    );
    let g = block.gate.log_decay(dt_raw);
    assert!(
        g.max().into_scalar::<f32>() <= 0.0,
        "log decay must be non-positive",
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
