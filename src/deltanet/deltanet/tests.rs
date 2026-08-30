//! DeltaNet's contract: `forward` over a sequence is `step` unrolled over the
//! same tokens from the same cache — on outputs, on the resulting cache, and on
//! gradients. Everything else in the crate (the layer stack, the caches, the
//! networks) assumes it.

use super::*;
use burn::tensor::Distribution;
use burn_stack::utils::test_helpers::max_abs_diff;

type Device = burn::prelude::Device;

fn tiny_config(d_model: usize) -> DeltaNetConfig {
    DeltaNetConfig::new(d_model).with_nheads(2)
}

fn random_input(batch: usize, sequence: usize, d_model: usize, device: &Device) -> Tensor<3> {
    Tensor::random(
        [batch, sequence, d_model],
        Distribution::Normal(0.0, 1.0),
        device,
    )
}

/// Unroll `step` over the sequence, returning the stacked outputs and the final
/// cache.
fn unroll_steps(
    block: &DeltaNet,
    input_bsd: Tensor<3>,
    cache: Option<DeltaCache>,
) -> (Tensor<3>, DeltaCache) {
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

fn assert_caches_match(a: &DeltaCache, b: &DeltaCache, label: &str, tol: f32) {
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

/// `forward` (chunkwise) == `step` unrolled, for a spread of configurations.
fn check_forward_matches_step(config: DeltaNetConfig, sequence: usize, path: DeltaPath, tol: f32) {
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
    for path in [DeltaPath::Recurrent, DeltaPath::chunk_len(4), DeltaPath::chunk()] {
        check_forward_matches_step(tiny_config(16), 10, path, 1e-4);
    }
}

#[test]
fn forward_matches_step_gated_output() {
    check_forward_matches_step(
        tiny_config(16).with_use_gate(true),
        10,
        DeltaPath::chunk_len(4),
        1e-4,
    );
}

#[test]
fn forward_matches_step_without_a_convolution() {
    check_forward_matches_step(
        tiny_config(16).with_use_short_conv(false),
        9,
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

#[test]
fn forward_matches_step_without_beta() {
    check_forward_matches_step(
        tiny_config(16).with_use_beta(false),
        8,
        DeltaPath::chunk_len(4),
        1e-4,
    );
}

#[test]
fn forward_matches_step_with_expanded_values() {
    check_forward_matches_step(
        tiny_config(16).with_expand_v(2.0).with_use_gate(true),
        8,
        DeltaPath::chunk_len(4),
        1e-4,
    );
}

#[test]
fn forward_matches_step_with_alternative_qk_activations() {
    for (activation, norm) in [
        (QkActivation::Identity, QkNorm::L2),
        (QkActivation::Relu, QkNorm::Sum),
        (QkActivation::EluPlusOne, QkNorm::Sum),
        (QkActivation::Relu, QkNorm::None),
    ] {
        check_forward_matches_step(
            tiny_config(16)
                .with_qk_activation(activation)
                .with_qk_norm(norm),
            8,
            DeltaPath::chunk_len(4),
            1e-3,
        );
    }
}

/// A conv kernel of 1 has no window to slide; a longer one than the sequence
/// exercises the window's zero-fill.
#[test]
fn forward_matches_step_across_conv_kernels() {
    for conv_kernel in [1, 2, 4, 8] {
        check_forward_matches_step(
            tiny_config(16).with_conv_kernel(conv_kernel),
            5,
            DeltaPath::chunk_len(4),
            1e-4,
        );
    }
}

/// Prefill in two calls, threading the cache, equals one call — the property a
/// chunked prompt ingest depends on.
#[test]
fn split_forward_matches_a_single_forward() {
    let device: Device = Default::default();
    let (batch, sequence, d_model, split) = (2, 12, 16, 5);
    let block = tiny_config(d_model).with_use_gate(true).init(&device);
    let input = random_input(batch, sequence, d_model, &device);
    let path = DeltaPath::chunk_len(4);

    let (y_whole, cache_whole) = block.forward(input.clone(), None, path);

    let (y_head, cache_mid) = block.forward(input.clone().narrow(1, 0, split), None, path);
    let (y_tail, cache_end) = block.forward(
        input.narrow(1, split, sequence - split),
        Some(cache_mid),
        path,
    );
    let y_split = Tensor::cat(vec![y_head, y_tail], 1);

    let diff = max_abs_diff(y_whole, y_split);
    assert!(diff < 1e-4, "split prefill differs by {diff}");
    assert_caches_match(&cache_whole, &cache_end, "whole vs split", 1e-4);
}

/// Gradients, not just values: the chunked path and the unrolled recurrence
/// must train the same weights the same way.
#[test]
fn forward_and_step_agree_on_parameter_gradients() {
    let device: Device = Device::default().autodiff();
    let (batch, sequence, d_model) = (2, 8, 16);
    let block = tiny_config(d_model).with_use_gate(true).init(&device);
    let input = random_input(batch, sequence, d_model, &device);
    let head = random_input(batch, sequence, d_model, &device);

    let grads_of = |y: Tensor<3>| {
        let loss = (y * head.clone()).sum();
        let grads = loss.backward();
        (
            block.qkv.in_proj.weight.val().grad(&grads).expect("in_proj"),
            block.out_proj.weight.val().grad(&grads).expect("out_proj"),
            block
                .qkv
                .conv
                .as_ref()
                .unwrap()
                .conv1d
                .weight
                .val()
                .grad(&grads)
                .expect("conv"),
        )
    };

    let (fwd_in, fwd_out, fwd_conv) = grads_of(
        block
            .forward(input.clone(), None, DeltaPath::chunk_len(4))
            .0,
    );
    let (step_in, step_out, step_conv) = grads_of(unroll_steps(&block, input, None).0);

    for (name, a, b) in [
        ("in_proj", fwd_in.clone(), step_in),
        ("out_proj", fwd_out.clone(), step_out),
    ] {
        let diff = max_abs_diff(a, b);
        assert!(diff < 1e-3, "grad of {name} differs by {diff}");
    }
    let conv_diff = max_abs_diff(fwd_conv, step_conv);
    assert!(conv_diff < 1e-3, "grad of conv differs by {conv_diff}");
}

/// The output shape is the model width, whatever the internal expansion.
#[test]
fn output_shape_is_d_model() {
    let device: Device = Default::default();
    let (batch, sequence, d_model) = (3, 7, 16);
    let block = tiny_config(d_model).with_expand_k(0.5).with_expand_v(2.0).init(&device);
    let (y, cache) = block.forward(random_input(batch, sequence, d_model, &device), None, DeltaPath::chunk());
    assert_eq!([batch, sequence, d_model], y.dims());
    assert_eq!([batch, 2, 4, 16], cache.state_bhkv.dims());
}
