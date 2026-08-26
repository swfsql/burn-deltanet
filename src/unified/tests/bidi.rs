//! The bidirectional stack over a real block.

use super::*;
use crate::delta::path::DeltaPath;
use crate::unified::{DeltaBidiLayersConfig, DeltaBidiShape};
use burn_stack::modules::OutputMergeConfig;
use burn_stack::utils::test_helpers::max_abs_diff;

fn config(d_model: usize, merge: OutputMergeConfig) -> DeltaBidiLayersConfig {
    DeltaBidiLayersConfig::GatedDeltaNet {
        shape: DeltaBidiShape::new(4, vec![merge.clone(), merge]),
        block: tiny_block(d_model),
    }
}

#[test]
fn a_bidirectional_stack_runs_and_keeps_its_shape() {
    let device: Device = Default::default();
    let (batch, sequence, d_model) = (2, 8, 16);
    for merge in [
        OutputMergeConfig::Mean,
        OutputMergeConfig::CatLinear,
    ] {
        let layers = config(d_model, merge.clone()).init(&device);
        let (y, caches) = layers.forward(
            random_input(batch, sequence, d_model, &device),
            None,
            DeltaPath::chunk_len(4),
            None,
        );
        assert_eq!([batch, sequence, d_model], y.dims(), "{merge:?}");
        // Two pairs, two directions: four cache slots.
        assert_eq!(4, caches.slot_count());
    }
}

/// The reversed pass must actually look at the future: reversing the input
/// permutes the output rather than leaving it alone. (A causal stack would
/// fail this only in the trivial direction — here both passes exist, so the
/// check is that the merge is order-sensitive at all.)
#[test]
fn the_reversed_pass_sees_later_tokens() {
    let device: Device = Default::default();
    let (batch, sequence, d_model) = (1, 6, 16);
    let layers = config(d_model, OutputMergeConfig::Mean).init(&device);
    let input = random_input(batch, sequence, d_model, &device);

    let (y, _) = layers.forward(input.clone(), None, DeltaPath::chunk_len(4), None);
    // Change only the *last* token; a causal stack would leave every earlier
    // output untouched.
    let mut edited = input.clone().narrow(1, 0, sequence - 1);
    edited = Tensor::cat(
        vec![edited, random_input(batch, 1, d_model, &device)],
        1,
    );
    let (y_edited, _) = layers.forward(edited, None, DeltaPath::chunk_len(4), None);

    let first_token_diff = max_abs_diff(
        y.narrow(1, 0, 1),
        y_edited.narrow(1, 0, 1),
    );
    assert!(
        first_token_diff > 1e-6,
        "the first output ignored a change to the last token: the reverse pass is not running",
    );
}
