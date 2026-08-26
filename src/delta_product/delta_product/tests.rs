//! DeltaProduct's contract: the usual `forward` == `step`-unrolled parity, plus
//! the identity that gives the family its meaning — `n_householder = 1` is
//! Gated DeltaNet, exactly.

use super::*;
use burn::tensor::Distribution;
use burn_stack::utils::test_helpers::max_abs_diff;

type Device = burn::prelude::Device;

fn tiny_config(d_model: usize, n_householder: usize) -> DeltaProductConfig {
    DeltaProductConfig::new(d_model)
        .with_n_householder(n_householder)
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
    block: &DeltaProduct,
    input_bsd: Tensor<3>,
    cache: Option<DeltaProductCache>,
) -> (Tensor<3>, DeltaProductCache) {
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

fn assert_caches_match(a: &DeltaProductCache, b: &DeltaProductCache, label: &str, tol: f32) {
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

fn check_forward_matches_step(config: DeltaProductConfig, sequence: usize, path: DeltaPath, tol: f32) {
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

/// The micro-step unrolling makes the core's sequence `sequence · u`, so the
/// chunk length interacts with `u`: a chunk boundary can fall *inside* a
/// token's Householder product. It must not matter.
#[test]
fn forward_matches_step_across_householder_counts_and_chunk_lengths() {
    for n_householder in [1, 2, 3] {
        for chunk_len in [2, 3, 4, 8] {
            check_forward_matches_step(
                tiny_config(16, n_householder),
                7,
                DeltaPath::chunk_len(chunk_len),
                2e-3,
            );
        }
        check_forward_matches_step(tiny_config(16, n_householder), 7, DeltaPath::Recurrent, 2e-3);
    }
}

#[test]
fn forward_matches_step_without_a_forget_gate() {
    check_forward_matches_step(
        tiny_config(16, 2).with_use_forget_gate(false),
        8,
        DeltaPath::chunk_len(4),
        2e-3,
    );
}

#[test]
fn forward_matches_step_without_an_output_gate() {
    check_forward_matches_step(
        tiny_config(16, 2).with_use_gate(false),
        8,
        DeltaPath::chunk_len(4),
        2e-3,
    );
}

#[test]
fn forward_matches_step_with_grouped_values_and_expansion() {
    check_forward_matches_step(
        tiny_config(16, 2).with_n_value_heads(4).with_expand_v(2.0),
        6,
        DeltaPath::chunk_len(4),
        2e-3,
    );
}

#[test]
fn forward_matches_step_without_a_convolution() {
    check_forward_matches_step(
        tiny_config(16, 3).with_use_short_conv(false),
        6,
        DeltaPath::chunk_len(4),
        2e-3,
    );
}

#[test]
fn split_forward_matches_a_single_forward() {
    let device: Device = Default::default();
    let (batch, sequence, d_model, split) = (2, 10, 16, 4);
    let block = tiny_config(d_model, 3).init(&device);
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
    assert!(diff < 2e-3, "split prefill differs by {diff}");
    assert_caches_match(&cache_whole, &cache_end, "whole vs split", 2e-3);
}

#[test]
fn forward_and_step_agree_on_parameter_gradients() {
    let device: Device = Device::default().autodiff();
    let (batch, sequence, d_model) = (2, 6, 16);
    let block = tiny_config(d_model, 2).init(&device);
    let input = random_input(batch, sequence, d_model, &device);
    let head = random_input(batch, sequence, d_model, &device);

    let grads_of = |y: Tensor<3>| {
        let grads = (y * head.clone()).sum().backward();
        (
            block.qkv.in_proj.weight.val().grad(&grads).expect("in_proj"),
            block.out_proj.weight.val().grad(&grads).expect("out_proj"),
            block
                .gate
                .as_ref()
                .unwrap()
                .a_log_h
                .val()
                .grad(&grads)
                .expect("a_log"),
        )
    };

    let (f_in, f_out, f_a) =
        grads_of(block.forward(input.clone(), None, DeltaPath::chunk_len(4)).0);
    let (s_in, s_out, s_a) = grads_of(unroll_steps(&block, input, None).0);

    for (name, diff) in [
        ("in_proj", max_abs_diff(f_in, s_in)),
        ("out_proj", max_abs_diff(f_out, s_out)),
        ("a_log", max_abs_diff(f_a, s_a)),
    ] {
        assert!(diff < 2e-3, "grad of {name} differs by {diff}");
    }
}

/// One Householder factor per transition *is* Gated DeltaNet — not an
/// approximation of it. The two blocks have the same parameter shapes in the
/// same order at `u = 1`, so one set of weights can be run through both.
#[cfg(feature = "gated-deltanet")]
#[test]
fn one_householder_is_gated_deltanet() {
    use crate::gated_deltanet::prelude::{GatedDeltaNet, GatedDeltaNetConfig};

    let device: Device = Default::default();
    let (batch, sequence, d_model) = (2, 9, 16);

    let gated: GatedDeltaNet = GatedDeltaNetConfig::new(d_model)
        .with_nheads(2)
        .with_head_k_dim(8)
        .with_expand_v(1.0)
        .with_use_gate(true)
        .with_allow_neg_eigval(true)
        .init(&device);

    // Same weights, run as a one-factor product.
    let product = DeltaProduct {
        qkv: gated.qkv.clone(),
        gate: Some(gated.gate.clone()),
        norm: gated.norm.clone(),
        out_proj: gated.out_proj.clone(),
    };
    assert_eq!(1, product.n_householder());

    let input = random_input(batch, sequence, d_model, &device);
    let path = DeltaPath::chunk_len(4);
    let (y_gated, cache_gated) = gated.forward(input.clone(), None, path);
    let (y_product, cache_product) = product.forward(input, None, path);

    let y_diff = max_abs_diff(y_gated, y_product);
    let state_diff = max_abs_diff(cache_gated.state_bhkv, cache_product.state_bhkv);
    assert!(y_diff < 1e-6, "outputs differ by {y_diff}");
    assert!(state_diff < 1e-6, "states differ by {state_diff}");
}

#[test]
fn output_shape_is_d_model_and_the_state_does_not_grow_with_u() {
    let device: Device = Default::default();
    let (batch, sequence, d_model) = (3, 5, 16);
    for n_householder in [1, 2, 4] {
        let block = tiny_config(d_model, n_householder).init(&device);
        let (y, cache) = block.forward(
            random_input(batch, sequence, d_model, &device),
            None,
            DeltaPath::chunk(),
        );
        assert_eq!([batch, sequence, d_model], y.dims());
        // `u` buys transition expressiveness, not state.
        assert_eq!([batch, 2, 8, 8], cache.state_bhkv.dims());
    }
}
