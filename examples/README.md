# DeltaNet Examples

#### List of Examples:

- `register-*`: A two-rung ladder on the same three-register file (six state scalars, no conv, no residual), each rung the smallest task its block is *needed* for and that the rung below cannot solve: `register-majority` (the delta rule's *keyed, erasing* write, against an accumulating state) and `register-carousel` (DeltaProduct's second Householder factor, against a 3-cycle no single real reflection reaches). Each carries a hand-built exact solution and the sweeps that wall off the rung below; the single [`register/README.md`](register/README.md) covers both.
- `mnist-class`: A small GDN-2 model classifying MNIST digits read as 784-pixel sequences — nothing hand-built. Each layer is the reference block plus the SwiGLU MLP the reference architecture puts after it, under the reference's global init; 2 real layers are cycled to 16 virtual ones under a `GradHorizon::Depth(2)` truncated-BPTT cut.
- `tiny-stories`: A 34K-parameter Gated DeltaNet language model over the characters of the cleaned [TinyStories](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean) corpus, with a tied 48-character embedding at both ends and the reference SwiGLU MLP. Its README covers the alphabet, the datasets-server paging that avoids a 673MB parquet, and the prefill-`forward()` / decode-`step()` sampler. Sized to `burn-mamba`'s example of the same name so the two are comparable.

#### Examples Structure

Each example lives in its own directory (the `register-*` ladder one level deeper, under `register/`, sharing one README — those two are declared as explicit `[[example]]` targets in `Cargo.toml`, since cargo only autodiscovers `examples/<name>/main.rs`). An example defines a model in `model.rs`, a dataset (if applicable, and if it is not one of the shared `burn_stack::examples` ones) in `dataset.rs`, a training procedure in `training.rs`, an inference procedure in `inference.rs`, a launching procedure in `main.rs`, and the claims it makes (if any) in `tests.rs`.

The launching procedure first parses the basic command arguments, which set whether training and/or inference should run. Training validates every few epochs, and each example's README states what the training goal is. The `model.rs` also documents why the model is shaped the way it is.

There are shared definitions in `common/mod.rs`, imported as an outside module by each example. It is a thin shim: the CLI, the runtime device selection, the training config, and both datasets with their epoch loops (sequential-MNIST classification, character-level TinyStories language modelling) all live in **`burn_stack::examples`** (feature `examples-common`, dev-only), shared verbatim with `burn-mamba`, and the `config → module` seam is `burn_stack::modules::ModelConfigExt`, implemented by this crate's network configs. `common/mod.rs` re-exports those under the `common::*` paths and adds `ARTIFACT_PREFIX`, which has to be expanded in the example crate.

##### Model Definition

The network used throughout the examples is the lib-generic `DeltaLatentNet` (configured via `DeltaLatentNetConfig`), defined in `burn-deltanet`'s `src/unified/network.rs`; the token-based `tiny-stories` uses its sibling `DeltaVocabNet` (embedding → `Layers<M>` → tied LM head) instead. It is a continuous-I/O network: input and output projections (linear layers) around a generic `Layers<M>` stack, here at `M = DeltaBlock` — the enum that picks the family (`DeltaNet`/`GatedDeltaNet1`/`DeltaProduct`/`GatedDeltaNet2`) at run time, from the `block` field of the config. `ModelConfigExt` (config → `Module`, plus the Muon plan) is implemented on those configs in `src/unified/network.rs`; examples define no network types of their own.

##### Tests

Every `register-*` example's `tests.rs` is the example's argument, measured. Each one builds its model **by hand** — every weight in closed form from the recurrence, nothing fitted — to establish that the block can solve the task a priori, then re-runs that same construction with one knob changed to establish that a weaker block cannot. The sweeps are over real blocks rather than simulations, so what they bound is what this crate actually computes.

##### Optimizer

`burn_stack::examples::training` defines `OptimizerConfig { adamw, muon }`, held by `TrainingConfig`. `muon = None` (the default) is plain AdamW on every parameter. Setting `muon` moves the hidden weight matrices to [Muon](https://kellerjordan.github.io/posts/muon/), driven by the model config's `muon_plan()` (`ModelConfigExt::muon_plan`, backed by `burn_stack::optim`): Muon only ever gets rank-2 hidden matrices, and each fused projection is split into its independent sub-projections first, so the orthogonalisation is per linear map rather than per allocation. The `register-*` examples are far too small for it to matter and leave it off; `mnist-class` exposes it as a `-- --muon` flag, and `tiny-stories` has it **on** by default (`-- --no-muon` turns it off).

#### Backend Selection

A single backend must be enabled, and features are used to select it -- e.g. `backend-flex`. See `burn-deltanet/Cargo.toml` > `[features]` section for the backend list. Some extra "dev" features are also available for selection, them being the float precision selection (default f32 vs `dev-f16`) and whether fusion and/or autotune should be enabled.
If no backend is selected, you should get a compile error message.

#### Examples CLI

All examples use a CLI defined in `burn_stack::examples::cli`, re-exported as `common::cli`.

##### Usage Example

```bash
# training the first example on flex (fp32) and running inference:
cargo run --example register-majority --features "backend-flex" -- --training --inference

# assume /tmp/burn-deltanet-register-majority-abcd-0 got created:
ARTIFACTS="/tmp/burn-deltanet-register-majority-abcd-0"

# running only the inference from the trained model:
cargo run --example register-majority --features "backend-flex" -- --inference --artifacts-path "$ARTIFACTS"

# assume /some/path/ contains a different training config file, e.g. with a different seed:
TCONFIG="/some/path/training_config.json"

# continue training from another training config
# warning: "$ARTIFACTS/training_config.json" gets overwritten by "$TCONFIG"
cargo run --example register-majority --features "backend-flex" -- --training --artifacts-path "$ARTIFACTS" --training-config "$TCONFIG"
```

Downstream flags go after a trailing `--`; `register-carousel` takes `--turn rotate|swap` and `--factors N`, `mnist-class` takes `--muon`, and `tiny-stories` takes the corpus knobs listed in its README.

##### CLI Help Message

Run any example with `-h` for the full flag list: it covers `--training` / `--inference`, the artifacts directory, the training/model config overrides, and `--remove-artifacts`.
