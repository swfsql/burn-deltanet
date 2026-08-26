# burn-deltanet

> A minimal, readable reference implementation of **DeltaNet, Gated DeltaNet,
> and DeltaProduct** for the [Burn](https://github.com/tracel-ai/burn) deep
> learning framework.

`burn-deltanet` ports the delta-rule family of linear-attention architectures —
[DeltaNet](https://arxiv.org/abs/2406.06484),
[Gated DeltaNet](https://arxiv.org/abs/2412.06464), and
[DeltaProduct](https://arxiv.org/abs/2502.10297) — down to **standard, portable
Burn tensor operations**. There are no custom CUDA/Triton kernels, so the exact
same code runs on every Burn backend (CPU, WGPU, CUDA, Metal, LibTorch, …). The
goal is clarity: a faithful, well-documented translation of the
[`flash-linear-attention`](https://github.com/fla-org/flash-linear-attention)
kernels that is easy to read, verify, and learn from.

It is the delta-rule counterpart of
[`burn-mamba`](https://github.com/swfsql/burn-mamba), and shares its composition
layer, [`burn-stack`](https://github.com/swfsql/burn-stack) — so a delta-rule
block drops into the same layers, virtual-layer stacks, bidirectional pairs,
networks and Muon plan that the Mamba families use.

---

## What is the delta rule?

Every model here carries one fixed-size associative memory `S ∈ ℝ^{k×v}` per
head — the same recurrent state a linear-attention model has. The difference is
how it is *written*:

```text
  linear attention:   Sₜ = Sₜ₋₁ + kₜ vₜᵀ
  delta rule:         Sₜ = αₜ (I − βₜ kₜ kₜᵀ) Sₜ₋₁ + βₜ kₜ vₜᵀ
  readout:            yₜ = Sₜᵀ qₜ
```

Linear attention accumulates unconditionally, so every new pair adds
interference to every stored one. The delta rule first *retrieves* what the
memory currently associates with `kₜ` and moves it a fraction `βₜ` of the way
towards `vₜ` — an error-correcting write, which is exactly the classical
[delta rule](https://arxiv.org/abs/2102.11174) of fast-weight programmers.

Two consequences carry the whole family:

- **The transition is a matrix, not a scalar.** With `‖kₜ‖ = 1` (`q`/`k` are
  L2-normalised) `I − βₜ kₜ kₜᵀ` is a generalised Householder with spectrum
  `{1, 1 − βₜ}` — the identity at `β = 0`, a projection at `β = 1`, an exact
  **reflection** at `β = 2`. Nothing outside `span(kₜ)` is disturbed, which is
  what makes targeted overwriting possible.
- **Reflections buy state tracking.** A scalar decay can only ever forget.
  Letting `β` reach 2 (`allow_neg_eigval`) puts `−1` in the spectrum, which is
  what a recurrence needs to *track* group structure rather than wash it out.

Each family is a different answer to "how much transition do you want?":

- **DeltaNet** — `α ≡ 1`. The state changes only when a token deliberately
  writes to it.
- **Gated DeltaNet** — adds the Mamba-2 scalar forget gate `αₜ = exp(Δₜ A)`.
  The paper's framing is a pairing: `α` erases indiscriminately (useful when the
  context has genuinely moved on), `β kkᵀ` erases one association (useful when a
  specific fact is being updated), and neither subsumes the other.
- **DeltaProduct** — `u = n_householder` micro-steps per token, so one
  transition is a *product* of Householder reflections. By Cartan–Dieudonné a
  product of `u` reflections spans the orthogonal transformations of a
  `u`-dimensional subspace, making `u` a direct dial on trackable group
  structure — at `u`× the recurrence work and **no extra state**. `u = 1` is
  Gated DeltaNet exactly, which the test suite asserts by running both from one
  set of weights.

## Highlights

- **All three families** — as a block, and (through `burn-stack`) a Pre-LN
  residual layer, a virtual-layer stack, bidirectional pairs, and full
  latent/vocabulary networks.
- **Backend-agnostic** — pure Burn tensor ops; no custom kernels.
- **Two execution modes** — a chunkwise `forward()` for training and prefill,
  and a recurrent `step()` for decoding at `O(state)` per token with no growing
  KV cache. They are the same function, asserted on outputs, caches **and**
  gradients.
- **Pinned to the reference** — the recurrence is checked numerically against
  `flash-linear-attention`'s own naive implementations at float64, not merely
  against itself.
- **One core, three families** — the recurrence and its chunkwise WY
  reformulation are written once; a family is a different way of *producing*
  `(q, k, v, β, α)`.

## Installation

```toml
# Note: check Cargo.toml for the actual burn rev being used.
[dependencies]
burn-deltanet = { git = "https://github.com/swfsql/burn-deltanet.git", default-features = false, features = ["backend-flex"] }
```

A backend must be selected by feature: `backend-{flex,cpu,wgpu,webgpu,metal,`
`vulkan,cuda,rocm,tch-cpu,tch-gpu,remote,ndarray}`. Several may be compiled in
at once; `Device::default()` resolves which to use at runtime (honouring
`BURN_DEVICE`). The families are features too — `deltanet`, `gated-deltanet`,
`delta-product` — all on by default.

## Quick start

```rust
use burn::prelude::*;
use burn_deltanet::prelude::*;

let device = Device::default();

// A block on its own.
let block = GatedDeltaNetConfig::new(256)
    .with_nheads(4)
    .with_head_k_dim(64)
    .init(&device);

let x = Tensor::<3>::zeros([2, 128, 256], &device);
let (y, cache) = block.forward(x, None, DeltaPath::chunk());

// Decoding continues from exactly that cache.
let token = Tensor::<2>::zeros([2, 256], &device);
let (y_next, _cache) = block.step(token, Some(cache));
```

For a stack, name the generic container directly when the family is known at
compile time:

```rust
use burn_stack::modules::{LayersBuilder, VocabNetworkBuilder};

let net = VocabNetworkBuilder {
    vocab_size: 32_000,
    pad_vocab_size_multiple: 8,
    layers: LayersBuilder::new(12, GatedDeltaNetConfig::new(768)),
    missing_lm_head: true,
}
.init(&device);
```

…or use the runtime-selectable `DeltaVocabNet` / `DeltaLatentNet` /
`DeltaBidiLayers` when the family comes out of a config file.

## Two execution modes

`forward()` runs the chunkwise WY algorithm — serial over chunks, batched GEMMs
within — and is what training and prompt prefill use. `step()` runs the
recurrence directly, one token at a time, in state-sized memory.

They agree: a `forward()` over a sequence equals `step()` unrolled from the same
cache, on outputs, on the resulting cache, and on gradients. Every block, layer
and network asserts it, and the `associative-recall` example checks it end to
end on a trained model.

## Choosing an algorithm

`DeltaPath` selects how the recurrence is evaluated:

| Variant | Algorithm | Use |
|---------|-----------|-----|
| `Recurrent` | token-by-token; `O(sequence)` serial steps | decoding, short prefills, the correctness reference |
| `Chunk { chunk_len, solve }` | chunkwise WY, batched GEMMs | training, prefill (the default) |

The chunk length trades intra-chunk GEMM work against the number of serial
inter-chunk steps; every reference kernel settles on 64, which is the default.
`solve` picks how the WY transform's `(I − N)⁻¹` is evaluated — `Doubling`
(`⌈log₂ L⌉` steps of two matmuls, the default) or `Neumann` (the literal
term-by-term series, kept as the obviously-correct reference).

Unlike Mamba-2's chunk scan, the inter-chunk recurrence here is matrix-valued
with a rank-`chunk_len` update, so there is no scalar-decay shortcut to
parallelise it. The win is entirely intra-chunk.

## Examples

See [`examples/README.md`](examples/README.md).

```bash
# ~98% held-out associative-recall accuracy in about 80 seconds on a CPU
cargo run --release --example associative-recall --features backend-flex
```

## Benchmarks

```bash
cargo bench                      # all cases, default features (flex)
cargo bench -- forward/gated     # one case
cargo bench -- --save-baseline flex
cargo bench -- --baseline flex
```

`benches/layer.rs` measures a single block — no layer or network wrapper — in
all three modes (`forward`, `train`, `step`) across the families, both
`TriSolve` variants, `Recurrent` against `Chunk`, and DeltaProduct's
`n_householder`. Sizes come from the environment (`BENCH_SEQ`, `BENCH_D_MODEL`,
…); see the file header.

## Documentation

```bash
cargo doc --no-deps --open
```

The module headers carry the mathematics: [`src/delta/`](src/delta/mod.rs) for
the recurrence and its chunkwise form, and each family's own module for what it
adds. [`files.md`](files.md) is a per-file signature reference.

## Scope

This crate is the **blocks**. Everything around them — layers, stacks,
bidirectional pairs, networks, class tokens, schedules, the Muon plan — lives in
[`burn-stack`](https://github.com/swfsql/burn-stack) and is deliberately
family-agnostic.

Not implemented: variable-length (`cu_seqlens`) packing, and the wider
`flash-linear-attention` zoo (GLA, RWKV, Comba, KDA, …).

## References

#### Papers

- [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174) — the delta rule as a fast-weight update.
- [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484) — DeltaNet, and the WY chunkwise algorithm.
- [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) — Gated DeltaNet.
- [Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](https://arxiv.org/abs/2411.12537) — why `β` is allowed to reach 2.
- [DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products](https://arxiv.org/abs/2502.10297) — DeltaProduct.

#### Implementation references

- [`flash-linear-attention`](https://github.com/fla-org/flash-linear-attention) — the authoritative Triton kernels this crate ports.
- [`burn-mamba`](https://github.com/swfsql/burn-mamba) — the sibling crate, same shape, different recurrence.
- [`burn-stack`](https://github.com/swfsql/burn-stack) — the block-generic composition layer both build on.
