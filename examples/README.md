# DeltaNet Examples

#### List of Examples:

- `register-majority`: One DeltaNet block (a three-row register file, six state scalars, no conv, no residual) on the majority of what those registers currently hold — the smallest task the block's *keyed, erasing* write is required for. Its README carries a hand-built exact solution and a sweep showing no accumulating state (the erase gate pinned at zero, i.e. gated linear attention) reaches it.
- `register-carousel`: The DeltaProduct rung above it — the same register file plus a `turn` instruction that permutes it, so the transition has to be a *permutation matrix*. `-- --turn swap` is a transposition, which one Householder reaches once `β` may reach 2; the default `rotate` is a 3-cycle, whose complex eigenvalues no single real Householder has. Hand-built exact solutions for both, plus sweeps over every single Householder.

#### Examples Structure

Each example defines a model in `model.rs`, a dataset in `dataset.rs`, a training procedure in `training.rs`, an inference procedure in `inference.rs`, a launching procedure in `main.rs`, and the claims it makes in `tests.rs`.

The launching procedure first parses the basic command arguments, which set whether training and/or inference should run. Training validates every few epochs, and each example's README states what the training goal is. The `model.rs` also documents why the model is shaped the way it is.

There are shared definitions in `common/mod.rs`, imported as an outside module by each example. A common model-factory seam and the backend selection are shared among all examples.

##### Model Definition

The network used throughout the examples is the lib-generic `DeltaLatentNet` (configured via `DeltaLatentNetConfig`), defined in `burn-deltanet`'s `src/unified/network.rs`. It is a continuous-I/O network: input and output projections (linear layers) around a generic `Layers<M>` stack, where `M` is the chosen delta-rule block (`DeltaNet`/`GatedDeltaNet`/`DeltaProduct`/`GatedDeltaNet2`). `common/model.rs` only supplies the `ModelConfigExt` glue (config → `Module`); examples define no network types of their own.

##### Tests

Every example's `tests.rs` is the example's argument, measured. Each one builds its model **by hand** — every weight in closed form from the recurrence, nothing fitted — to establish that the block can solve the task a priori, then re-runs that same construction with one knob changed to establish that a weaker block cannot. The sweeps are over real blocks rather than simulations, so what they bound is what this crate actually computes.

##### Optimizer

`common/training.rs` defines `OptimizerConfig { adamw, muon }`, held by `TrainingConfig`. `muon = None` (the default) is plain AdamW on every parameter. Setting `muon` moves the hidden weight matrices to [Muon](https://kellerjordan.github.io/posts/muon/), driven by the model config's `muon_plan()` (`ModelConfigExt::muon_plan`, backed by `burn_stack::optim`): Muon only ever gets rank-2 hidden matrices, and each fused projection is split into its independent sub-projections first, so the orthogonalisation is per linear map rather than per allocation. These two examples are far too small for it to matter, and leave it off.

#### Backend Selection

A single backend must be enabled, and features are used to select it -- e.g. `backend-flex`. See `burn-deltanet/Cargo.toml` > `[features]` section for the backend list. Some extra "dev" features are also available for selection, them being the float precision selection (default f32 vs `dev-f16`) and whether fusion and/or autotune should be enabled.
If no backend is selected, you should get a compile error message.

#### Examples CLI

All examples use a CLI defined in `common/cli.rs`.

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

Downstream flags go after a trailing `--`; `register-carousel` takes `--turn rotate|swap` and `--factors N`.

##### CLI Help Message

Run any example with `-h` for the full flag list: it covers `--training` / `--inference`, the artifacts directory, the training/model config overrides, and `--remove-artifacts`.
