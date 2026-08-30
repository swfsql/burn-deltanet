//! `Layers<M>` over a real block: forward/step parity through the residual
//! stack, virtual layers, and the truncated-BPTT gradient horizon.

use super::*;
use crate::delta::path::DeltaPath;
use crate::gated_deltanet_1::prelude::GatedDeltaNet1;
use burn_stack::modules::{Layers, LayersBuilder};
use burn_stack::utils::{GradHorizon, Schedule, test_helpers::max_abs_diff};

fn unroll_steps(
    layers: &Layers<GatedDeltaNet1>,
    input_bsd: Tensor<3>,
) -> Tensor<3> {
    let [_batch, sequence, _d_model] = input_bsd.dims();
    let mut caches = None;
    let mut outputs = Vec::with_capacity(sequence);
    for t in 0..sequence {
        let x_bd = input_bsd.clone().narrow(1, t, 1).squeeze_dim(1);
        let (y_bd, next) = layers.step(x_bd, caches.take(), None);
        caches = Some(next);
        outputs.push(y_bd.unsqueeze_dim(1));
    }
    Tensor::cat(outputs, 1)
}

fn check_stack(builder: LayersBuilder<crate::gated_deltanet_1::prelude::GatedDeltaNet1Config>) {
    let device: Device = Default::default();
    let (batch, sequence, d_model) = (2, 9, 16);
    let layers = builder.init(&device);
    let input = random_input(batch, sequence, d_model, &device);

    let (y_forward, _) = layers.forward(input.clone(), None, DeltaPath::chunk_len(4), None);
    let y_stepped = unroll_steps(&layers, input);

    let diff = max_abs_diff(y_forward, y_stepped);
    assert!(diff < 1e-3, "stack forward vs step differs by {diff}");
}

#[test]
fn a_plain_stack_keeps_forward_step_parity() {
    check_stack(LayersBuilder::new(3, tiny_block(16)));
}

/// Virtual layers: more logical passes than weight sets, each with its own
/// cache slot.
#[test]
fn virtual_layers_keep_forward_step_parity() {
    check_stack(
        LayersBuilder::new(2, tiny_block(16))
            .with_n_virtual_layers(Some((6, Schedule::Cyclic))),
    );
}

/// An interleaved SwiGLU sub-block sits between the mixer and the residual;
/// it is stateless, so parity must survive it untouched.
#[test]
fn an_interleaved_mlp_keeps_forward_step_parity() {
    check_stack(
        LayersBuilder::new(2, tiny_block(16))
            .with_mlp(Some(burn_stack::modules::GatedMlpConfig::new(16, 32))),
    );
}

#[test]
fn multi_gate_residuals_keep_forward_step_parity() {
    check_stack(
        LayersBuilder::new(4, tiny_block(16)).with_residuals(
            burn_stack::modules::ResidualsConfig::MultiGate {
                n_stream: 3,
                init_bias: 0.0,
                init_bias_step: 0.0,
                per_virtual_layer: false,
            },
        ),
    );
}

/// `grad_horizon` runs the stack's lower layers on the inner backend, which
/// means every cache has to make the backend hop too — the `CacheStack`
/// conversions this crate spells out by hand. Values must be unchanged; only
/// gradient *reachability* differs.
#[test]
fn a_gradient_horizon_leaves_the_forward_untouched() {
    let device: Device = Device::default().autodiff();
    let (batch, sequence, d_model) = (2, 6, 16);
    let path = DeltaPath::chunk_len(4);
    let input = random_input(batch, sequence, d_model, &device);

    let builder = || {
        LayersBuilder::new(2, tiny_block(d_model))
            .with_n_virtual_layers(Some((6, Schedule::Cyclic)))
    };
    let full = builder().init(&device);
    // Same weights, but only the top two virtual layers back-propagate.
    let truncated = Layers {
        grad_horizon: Some(GradHorizon::last(2, 6)),
        ..full.clone()
    };

    let (y_full, _) = full.forward(input.clone(), None, path, None);
    let (y_truncated, _) = truncated.forward(input.clone(), None, path, None);
    let diff = max_abs_diff(y_full.clone().inner(), y_truncated.clone().inner());
    assert!(diff < 1e-4, "grad_horizon changed the forward by {diff}");

    // And it still trains: the stack input is re-attached straight-through at
    // the boundary, so gradients reach the bottom of the graph.
    let grads = y_truncated.sum().backward();
    assert!(
        truncated.real_layers[0]
            .block
            .out_proj
            .weight
            .val()
            .grad(&grads)
            .is_some()
            || truncated.real_layers[1]
                .block
                .out_proj
                .weight
                .val()
                .grad(&grads)
                .is_some(),
        "no real layer received a gradient under grad_horizon",
    );
}

/// Stepping the stack in two runs, threading the caches, equals one run.
#[test]
fn a_split_forward_threads_the_stack_caches() {
    let device: Device = Default::default();
    let (batch, sequence, d_model, split) = (2, 10, 16, 4);
    let layers = LayersBuilder::new(2, tiny_block(d_model)).init(&device);
    let input = random_input(batch, sequence, d_model, &device);
    let path = DeltaPath::chunk_len(4);

    let (y_whole, _) = layers.forward(input.clone(), None, path, None);
    let (y_head, caches) = layers.forward(input.clone().narrow(1, 0, split), None, path, None);
    let (y_tail, _) = layers.forward(
        input.narrow(1, split, sequence - split),
        Some(caches),
        path,
        None,
    );

    let diff = max_abs_diff(y_whole, Tensor::cat(vec![y_head, y_tail], 1));
    assert!(diff < 1e-3, "split stack forward differs by {diff}");
}
