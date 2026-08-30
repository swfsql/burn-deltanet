//! The Muon plan: that each family's declared column seams actually line up
//! with the weight the forward built, and that the right channels are excluded.

use super::*;
use crate::unified::{
    DeltaBlockConfig, DeltaLatentNetConfig, DeltaLatentShape, DeltaNetworkShape,
};
use burn_stack::optim::ProjSpec;

/// The seams a family declares must sum to its real `in_proj` width. This is
/// the failure Muon would otherwise hit late and confusingly: a spec that adds
/// up to the wrong number silently orthogonalises the wrong columns.
fn check_seams_cover_the_weight(specs: Vec<ProjSpec>, in_proj_out: usize, d_model: usize) {
    let in_proj = specs
        .iter()
        .find(|s| s.path.contains("in_proj"))
        .expect("an in_proj spec");
    assert_eq!(
        in_proj_out,
        in_proj.width(),
        "the declared in_proj segments do not cover the projection",
    );
    let out_proj = specs
        .iter()
        .find(|s| s.path.contains("out_proj"))
        .expect("an out_proj spec");
    assert_eq!(d_model, out_proj.width());
}

#[test]
fn deltanet_seams_match_its_projection() {
    let device: Device = Default::default();
    let config = crate::deltanet::prelude::DeltaNetConfig::new(16)
        .with_nheads(2)
        .with_use_gate(true);
    let block = config.init(&device);
    let [_d_model, in_proj_out] = block.qkv.in_proj.weight.dims();
    check_seams_cover_the_weight(config.muon_projections(), in_proj_out, 16);
}

#[test]
fn gated_deltanet_1_seams_match_its_projection() {
    let device: Device = Default::default();
    for n_value_heads in [0, 4] {
        let config = tiny_block(16).with_n_value_heads(n_value_heads);
        let block = config.init(&device);
        let [_d_model, in_proj_out] = block.qkv.in_proj.weight.dims();
        check_seams_cover_the_weight(config.muon_projections(), in_proj_out, 16);
    }
}

#[test]
fn delta_product_seams_match_its_projection() {
    let device: Device = Default::default();
    for n_householder in [1, 2, 3] {
        for use_forget_gate in [true, false] {
            let config = crate::delta_product::prelude::DeltaProductConfig::new(16)
                .with_nheads(2)
                .with_head_k_dim(4)
                .with_expand_v(1.0)
                .with_n_householder(n_householder)
                .with_use_forget_gate(use_forget_gate);
            let block = config.init(&device);
            let [_d_model, in_proj_out] = block.qkv.in_proj.weight.dims();
            check_seams_cover_the_weight(config.muon_projections(), in_proj_out, 16);
        }
    }
}

#[test]
fn gdn2_seams_match_its_projection() {
    let device: Device = Default::default();
    for n_value_heads in [0, 4] {
        for use_gate in [true, false] {
            let config = crate::gated_deltanet_2::prelude::GatedDeltaNet2Config::new(16)
                .with_nheads(2)
                .with_head_k_dim(4)
                .with_n_value_heads(n_value_heads)
                .with_use_gate(use_gate);
            let block = config.init(&device);
            let [_d_model, in_proj_out] = block.qkv.in_proj.weight.dims();
            check_seams_cover_the_weight(config.muon_projections(), in_proj_out, 16);
        }
    }
}

/// GDN-2 is the family with *no* per-head scalar channel: its erase, write and
/// `Δ` maps all produce feature vectors, so every seam is Muon's. The two
/// bottlenecks' second factors are separate weights and are claimed whole.
#[test]
fn gdn2_puts_every_seam_on_muon() {
    let config = crate::gated_deltanet_2::prelude::GatedDeltaNet2Config::new(16)
        .with_nheads(2)
        .with_head_k_dim(4);
    let specs = config.muon_projections();
    let in_proj = specs.iter().find(|s| s.path.contains("in_proj")).unwrap();
    assert!(
        in_proj.segments.iter().all(|s| s.muon),
        "no GDN-2 in_proj segment is a per-head scalar",
    );
    assert_eq!(
        vec!["q", "k", "v", "erase", "write", "dt_in", "gate_in"],
        in_proj
            .segments
            .iter()
            .map(|s| s.name)
            .collect::<Vec<_>>(),
    );
    for path in ["gate.up.weight", "out_gate.weight"] {
        assert!(
            specs.iter().any(|s| s.path == path),
            "{path} is a matrix Muon should own",
        );
    }
}

/// Per-head *scalar* channels (`β`, the forget gate's `Δ`) are deliberately
/// left on AdamW: a `[d_model, nheads]` slice is a stack of independent linear
/// functionals, not a matrix whose singular values mean anything.
#[test]
fn per_head_scalar_channels_stay_on_adamw() {
    let config = tiny_block(16);
    let specs = config.muon_projections();
    let in_proj = specs.iter().find(|s| s.path.contains("in_proj")).unwrap();

    let adamw: Vec<&str> = in_proj
        .segments
        .iter()
        .filter(|s| !s.muon)
        .map(|s| s.name)
        .collect();
    assert_eq!(vec!["beta", "dt"], adamw);

    let muon: Vec<&str> = in_proj
        .segments
        .iter()
        .filter(|s| s.muon)
        .map(|s| s.name)
        .collect();
    assert_eq!(vec!["q", "k", "v", "gate"], muon);
}

/// DeltaProduct lists each of its `u` key and value maps separately —
/// orthogonalising them jointly would couple Householder factors that are meant
/// to be chosen independently.
#[test]
fn delta_product_lists_each_householder_map_separately() {
    let config = crate::delta_product::prelude::DeltaProductConfig::new(16)
        .with_nheads(2)
        .with_head_k_dim(4)
        .with_n_householder(3);
    let specs = config.muon_projections();
    let in_proj = specs.iter().find(|s| s.path.contains("in_proj")).unwrap();

    assert_eq!(3, in_proj.segments.iter().filter(|s| s.name == "k").count());
    assert_eq!(3, in_proj.segments.iter().filter(|s| s.name == "v").count());
}

/// The plan applies to a whole network, matching by path substring, and leaves
/// the network's boundary weights (in/out projections, embeddings) alone.
#[test]
fn the_plan_describes_a_built_network() {
    let device: Device = Default::default();
    let config = DeltaLatentNetConfig::new(
        DeltaLatentShape::new(5, 5, DeltaNetworkShape::new(2)),
        DeltaBlockConfig::GatedDeltaNet1(tiny_block(16)),
    );
    let net = config.init(&device);
    let report = config.muon_plan().describe(&net);

    assert!(report.contains("qkv.in_proj.weight"), "{report}");
    assert!(report.contains("muon"), "{report}");
    // The block is an enum, so its variant sits in the parameter path between
    // the container and the weight — which is why a `ProjSpec` matches the two
    // as separate substrings. A spec matching `"block.qkv.in_proj.weight"` as
    // one string would find nothing here, silently.
    assert!(
        report.contains("block.GatedDeltaNet1.qkv.in_proj.weight"),
        "{report}",
    );
    // The network's own boundary projections are not the block's.
    assert!(
        report
            .lines()
            .any(|line| line.contains("layers") && line.contains("out_proj") && line.contains("muon")),
        "the block's out_proj should be on muon:\n{report}",
    );
}
