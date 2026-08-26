//! The networks — generic and runtime-selectable — over real blocks.

use super::*;
use crate::delta::path::DeltaPath;
use crate::unified::{
    DeltaCaches, DeltaLatentNetConfig, DeltaLatentShape, DeltaNetworkShape, DeltaVocabNetConfig,
    DeltaVocabShape,
};
use burn_stack::utils::test_helpers::max_abs_diff;

fn latent_config(d_model: usize, io: usize) -> DeltaLatentNetConfig {
    DeltaLatentNetConfig::GatedDeltaNet {
        shape: DeltaLatentShape::new(io, io, DeltaNetworkShape::new(2)).with_final_norm(true),
        block: tiny_block(d_model),
    }
}

/// The whole point of the stack: `forward` over a sequence is `step` unrolled,
/// still, with an `in_proj`, three layers, a final norm and an `out_proj`
/// wrapped around the block.
#[test]
fn a_latent_network_keeps_forward_step_parity() {
    let device: Device = Default::default();
    let (batch, sequence, d_model, io) = (2, 9, 16, 5);
    let net = latent_config(d_model, io).init(&device);
    let input = random_input(batch, sequence, io, &device);

    let (y_forward, cache_forward) =
        net.forward(input.clone(), None, DeltaPath::chunk_len(4), None);

    let mut caches: Option<DeltaCaches> = None;
    let mut outputs = Vec::new();
    for t in 0..sequence {
        let x_bd = input.clone().narrow(1, t, 1).squeeze_dim(1);
        let (y_bd, next) = net.step(x_bd, caches.take(), None);
        caches = Some(next);
        outputs.push(y_bd.unsqueeze_dim(1));
    }

    let diff = max_abs_diff(y_forward, Tensor::cat(outputs, 1));
    assert!(diff < 1e-3, "latent network forward vs step differs by {diff}");
    assert_eq!(2, cache_forward.slot_count());
}

/// Token in, logits out — and the same parity.
#[test]
fn a_vocab_network_keeps_forward_step_parity() {
    let device: Device = Default::default();
    let (batch, sequence, d_model, vocab) = (2, 7, 16, 11);
    let net = DeltaVocabNetConfig::GatedDeltaNet {
        shape: DeltaVocabShape::new(vocab, DeltaNetworkShape::new(2)),
        block: tiny_block(d_model),
    }
    .init(&device);

    let tokens = Tensor::<2, Int>::from_data(
        burn::tensor::TensorData::new(
            (0..batch * sequence).map(|i| (i % vocab) as i64).collect::<Vec<_>>(),
            [batch, sequence],
        ),
        &device,
    );

    let (logits_forward, _) = net.forward(tokens.clone(), None, DeltaPath::chunk_len(4), None);
    assert_eq!([batch, sequence, vocab], logits_forward.dims());

    let mut caches: Option<DeltaCaches> = None;
    let mut outputs = Vec::new();
    for t in 0..sequence {
        let token = tokens.clone().narrow(1, t, 1).squeeze_dim(1);
        let (logits, next) = net.step(token, caches.take(), None);
        caches = Some(next);
        outputs.push(logits.unsqueeze_dim(1));
    }

    let diff = max_abs_diff(logits_forward, Tensor::cat(outputs, 1));
    assert!(diff < 1e-3, "vocab network forward vs step differs by {diff}");
}

/// Every family reaches the same containers through the same enum.
#[test]
fn every_family_builds_and_runs_through_the_runtime_enum() {
    let device: Device = Default::default();
    let (batch, sequence, d_model, io) = (2, 6, 16, 5);
    let stack = || DeltaNetworkShape::new(2);
    let shape = || DeltaLatentShape::new(io, io, stack());

    let configs = vec![
        (
            "DeltaNet",
            DeltaLatentNetConfig::DeltaNet {
                shape: shape(),
                block: crate::deltanet::prelude::DeltaNetConfig::new(d_model).with_nheads(2),
            },
        ),
        (
            "Gated DeltaNet",
            DeltaLatentNetConfig::GatedDeltaNet {
                shape: shape(),
                block: tiny_block(d_model),
            },
        ),
        (
            "DeltaProduct",
            DeltaLatentNetConfig::DeltaProduct {
                shape: shape(),
                block: crate::delta_product::prelude::DeltaProductConfig::new(d_model)
                    .with_nheads(2)
                    .with_head_k_dim(4)
                    .with_expand_v(1.0),
            },
        ),
    ];

    for (name, config) in configs {
        let net = config.init(&device);
        let input = random_input(batch, sequence, io, &device);
        let (y, caches) = net.forward(input, None, DeltaPath::chunk_len(4), None);
        assert_eq!([batch, sequence, io], y.dims(), "{name}");
        assert_eq!(name, caches.family_name());
    }
}

/// Runtime selection means the compiler cannot check the pairing, so a
/// mismatch has to be a loud panic rather than a silent shape error.
#[test]
#[should_panic(expected = "cache family mismatch")]
fn caches_from_the_wrong_family_panic() {
    let device: Device = Default::default();
    let (d_model, io) = (16, 5);
    let gated = latent_config(d_model, io).init(&device);
    let plain = DeltaLatentNetConfig::DeltaNet {
        shape: DeltaLatentShape::new(io, io, DeltaNetworkShape::new(2)),
        block: crate::deltanet::prelude::DeltaNetConfig::new(d_model).with_nheads(2),
    }
    .init(&device);

    let (_, wrong_caches) = plain.forward(
        random_input(1, 3, io, &device),
        None,
        DeltaPath::chunk_len(4),
        None,
    );
    let _ = gated.forward(
        random_input(1, 3, io, &device),
        Some(wrong_caches),
        DeltaPath::chunk_len(4),
        None,
    );
}
