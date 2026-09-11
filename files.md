# files.md

Per-file signature reference for `burn-deltanet`: what each file defines, and
the decisions that are not obvious from the signatures. The mathematics lives in
the module headers (`src/delta/mod.rs` above all); this is the index into them.

The composition layer (`Layer`/`Layers`/networks/bidi/multi-gate/class
tokens/schedules/norms/losses/Muon) is `../burn-stack/` and has its own.

---

## `src/lib.rs`

Crate root. `#![warn(missing_docs)]`. Module declarations, the `prelude`
(families + `burn_stack::prelude::*`), a `pub use burn_stack`, and the
quick-start doctest in the crate header.

Module declarations carry **no** outer `///` docs — see the rustdoc gotcha in
`CLAUDE.md`.

---

## `src/common/` — the block-level pieces every family shares

### `cache.rs`
- `DeltaCache { conv_bwc: Option<Tensor<3>>, state_bhkv: Tensor<4> }` +
  `from_parts`, `sanity()`; `DeltaCacheConfig` (batch, nheads, head_k_dim,
  head_v_dim, conv_dim, conv_kernel) → zero-initialised tensors.
- `DeltaCaches { caches: Vec<DeltaCache> }` + `caches_len`, `from_vec`,
  `into_options`/`from_options`; `DeltaCachesConfig { n_caches, cache }`.

**One cache type for all four families.** They differ in what they *project* —
a scalar `β`, a per-head gate, per-channel erase/write gates, `u` Householder
factors — and every one of those is computed from the current token and consumed
within the step. What crosses a step boundary is the same window and the same
associative memory. This is what lets `DeltaBlock` dispatch a family at runtime
with no cache tag, and `CacheStack` be implemented once.

### `conv.rs`
- `ConvActivation { Silu, Identity }` — a `#[module(skip)]` constant.
- `ShortConv { conv1d, activation }`, `ShortConvConfig`.
  - `forward(x_bsw, window_bwc) -> (y_bsw, next_window)` — left-pads with the
    window's newest `conv_kernel − 1` columns; `Valid` padding, so causality is
    manual.
  - `step(x_bw, window_bwc) -> (y_bw, next_window)` — slides the window and
    evaluates the filter against it as a dot product along the kernel axis.
  - `zero_window`, `conv_dim`, `conv_kernel`.

The window carries `conv_kernel` columns — one more than a `forward` needs —
because `step` evaluates against the whole of it.

### `gate.rs`
- `ForgetGate { dt_bias_h, a_log_h, dt_limit }`, `ForgetGateConfig`.
- `log_decay<D>(dt_raw) -> g ≤ 0` — `softplus(dt_raw + dt_bias) · (−exp(a_log))`,
  clamped to `dt_limit`; per-head parameters broadcast over every leading axis.

- `ChannelForgetGate { up, dt_bias_i, a_log_h, nheads, head_k_dim, lower_bound }`,
  `ChannelForgetGateConfig`; `log_decay(raw) -> g` over `[.., nheads,
  head_k_dim]` — GDN-2's per-channel gate, `sigmoid(Δ·exp(a_log)) · lower_bound`
  from the low-rank bottleneck's second factor.

`A = −exp(a_log)` makes the gate non-amplifying unconditionally; `dt_bias` is
the inverse softplus of a log-uniform `Δ` spread, so heads start with different
timescales. `ChannelForgetGate` bounds `g` by construction instead
(`lower_bound ∈ [-5, 0)`), which is what keeps the factored chunk decay in range.

### `norm.rs`
- `QkActivation { Identity, Silu, Relu, EluPlusOne }`, `QkNorm { L2, Sum, None }`
  — both `#[module(skip)]` constants; `.apply(x)` over the last dim.
- `l2_normalize` (`x/√(Σx²+1e-6)`, the reference's own epsilon), `sum_normalize`.
- `OutNorm { Gated(RmsNormGated), Plain(RmsNorm) }` + `init(head_v_dim, has_gate,
  device)`; `forward(x, gate: Option<_>)` panics on a mismatch. Normalises over
  **`head_v_dim`, per head** — not over the concatenated `value_dim`.

The module header argues why L2 is load-bearing: it forces `‖k‖ = 1`, which puts
the Householder's spectrum at `{1, 1−β}` and so moves the contract/reflect
decision entirely onto `β`.

### `qkv.rs`
- `WriteGate { Fixed, Scalar, Channel }` — how finely a family may separate the
  erase from the write; a `#[module(skip)]` constant.
- `QkvProjection { in_proj, conv, qk_activation, qk_norm, nheads, n_value_heads,
  head_k_dim, head_v_dim, n_householder, write_gate, has_gate, allow_neg_eigval,
  extra_channels }`, `QkvProjectionConfig`.
- `forward(x_bsd, window) -> (Qkv, next_window)`, `step(x_bd, window) -> (QkvStep,
  next_window)`.
- `Qkv { q_bshk, k_bShk, v_bShv, erase_bShK, write_bShV, gate_bshv, extra_bsx }`
  — the gate axes `K`/`V` are `1` or the full channel width; `QkvStep` is the
  same with an explicit micro-step axis instead of a sequence.
- `segments() -> Vec<(&'static str, usize)>` — the column widths, and the single
  source of truth for both the forward's split and the Muon seams.
- `key_dim`, `value_dim`, `conv_dim`, `write_gate_width`, `d_in_proj`,
  `state_heads`, `heads_per_group`, `conv_kernel`, `zero_conv_window`,
  `expand_to_value_heads` (public: a family may have its own query/key-side
  tensor to replicate).

Four decisions live here:
- **The micro-step fold.** `k`/`v` and the write gate are projected
  `n_householder`-wide and returned at length `sequence · u`, in micro-step
  order. `u = 1` is the plain sequence, so nothing downstream has a special
  case.
- **The gate squashing.** `squash_write` turns a raw segment into the
  `(erase, write)` pair at its broadcast width: `σ` on both, and
  `allow_neg_eigval` doubles the **erase** only — under `Fixed`/`Scalar` the one
  number is both halves, so it is doubled there too, matching the scalar
  reference's `beta = beta * 2`.
- **Grouped values.** `q`/`k` are replicated up to `n_value_heads` immediately
  after the head split, so the delta rule, caches and norms see one head count.
- **Activation placement.** `activation_fused_into_conv()` — the convolution's
  SiLU covers `q`/`k`/`v` at once when `qk_activation` is SiLU; otherwise the
  convolution stays linear and the activations are applied per segment.
  Identical values either way.

---

## `src/delta/` — the delta-rule core

### `mod.rs`
Module header only: the recurrence, the WY derivation, why chunks are serial,
and the notation table. No code.

### `path.rs`
- `DeltaPath { Recurrent, Chunk { chunk_len: Option<usize>, solve: TriSolve },
  ChunkRecalculated { chunk_len: Option<usize> } }` — `Default` is
  `ChunkRecalculated { None }`; `DEFAULT_CHUNK_LEN = 64`; `chunk()`,
  `chunk_len(n)`, `chunk_len_on_tape(n)`, `resolved_chunk_len()`.
  `Chunk`/`ChunkRecalculated` are the same forward and differ only in how it is
  differentiated (autodiff vs. the hand-written node).
- `DeltaInput { q_bshk, k_bshk, v_bshv, erase_bshK, write_bshV, g_bshK:
  Option<_>, state_bhkv, scale: Option<f64> }` + `dims()`, `resolved_scale()`
  (`1/√head_k_dim`), `has_channel_decay()`, `sanity()`,
  `run(path) -> (y_bshv, final_state_bhkv)`.
- `beta_gates(beta)` — the `(erase, write)` pair a per-head `β` stands for.

The three gates ride their broadcast axes: last axis `1` (one number per head)
or the full channel count (GDN-2). The same expressions serve both; only the
chunk path looks at the width.

`q`/`k` arrive activated and normalised but **not** scaled — `scale` is applied
inside, so the same bundle reads identically on either path.

### `recurrent.rs`
- `delta_step(q_bhk, k_bhk, v_bhv, erase_bhK, write_bhV, g_bhK: Option<_>,
  state_bhkv, scale) -> (y_bhv, next_state)` — the definition, and the primitive every family's
  `step()` decodes with. Retrieval and readout are `[1,k]@[k,v]` matmuls rather
  than broadcast-and-sum, so the backend sees GEMMs.
- `DeltaInput::delta_recurrent()` — unrolls it.

### `chunk.rs`
- `DeltaInput::delta_chunk(chunk_len, solve)` — the chunkwise WY algorithm,
  differentiated by autodiff. The status quo; `chunk_recalculated/` is the same
  forward with a hand-written backward.

Layout is `[batch, nchunks, nheads, chunk_len, ·]`: heads ahead of the chunk axis
so every matmul batches over `(b, n, h)` and acts on the `[chunk_len, ·]` planes.
The gate is threaded as an `Option` throughout — `None` builds no decay tensors
at all rather than multiplying by ones.

### `decay.rs`
- `DECAY_BLOCK_LEN = 16`; `BlockDecay::new(k_bnhlk, gc_bnhlk)`,
  `.scores(rows_bnhlk)` — the intra-chunk score matrix under a **per-channel**
  gate (`pub(super)`).

A per-head decay factors out of the key contraction into a `[chunk_len,
chunk_len]` mask; a per-channel one shares the channel index with the
contraction, and splitting it as `e^{Gᵢ}·e^{−Gⱼ}` overflows f32 within a few
dozen tokens. Following `fla/ops/gdn2/chunk_intra.py`, each block of
`DECAY_BLOCK_LEN` rows gets its own reference `G` at the block's middle row, so
neither factor spans more than half a block. Out-of-range columns get `−∞`
(exactly `0`, never `inf`) and are masked away downstream anyway.

### `tri.rs`
- `TriSolve { Blocked (default), Neumann }`,
  `unit_lower_inverse<D>(n_strict, solve)`. Both are **forward** algorithms,
  differentiated by autodiff; which backward runs is chosen on `DeltaPath`.

`N` is nilpotent, so the series is exact — and unusable: with a chunk's keys
correlated its partial sums peak near `C(L−2, L/2)` before cancelling down to a
`T` of order 1, so `Neumann` is a reference only. `Blocked` inverts by blocked
forward substitution instead — `P ← P + P X P` over doubling block sizes,
`⌈log₂ L⌉` steps of two matmuls, every intermediate an exact inverse of a
principal submatrix. The caller owes the strict-lower property; nothing here
re-masks it.

### `tri/prim.rs`
- `unit_lower_inverse::<B>(n_strict_fss)` — `tri.rs`'s `Blocked` ladder on
  backend primitives (`burn_stack::utils::fprim::F`), for
  `[flat, size, size]`. Same values; it exists because a custom autodiff node
  runs under a generic `B`, where the `Dispatch`-pinned `Tensor` is
  unavailable. `Neumann` is not ported — it is a reference for the *forward*,
  and this path is about the backward.
- `block_strict_lower::<B>` — the primitive port of `tri.rs`'s helper. `F`
  carries `triu` but no `eye`, hence `eye = triu(0) − triu(1)`.

### `tests/reference.rs`
Values captured from `flash-linear-attention`'s `delta_rule_recurrence` and
`naive_recurrent_gated_delta_rule` at float64, for a fixed input. Regenerate
with `scripts/gen_fixture.py`. This is what makes the suite a *port* check
rather than only a self-consistency check.

---

## `src/delta/chunk_recalculated/` — the same forward, backward by hand

`ChunkRecalculated`: one custom autodiff node over the *whole* chunk body that
retains only its seven leaf inputs, replays the forward in its backward, and
takes the gradient analytically. This is the reference kernel's own split —
`chunk_delta_rule_fwd`/`_bwd` in `fla/ops/delta_rule/chunk.py` save
`(q, k, v, β, A, h₀)` and recompute `w`, `u` and the per-chunk state stream.

### `chunk_recalculated.rs`
- `DeltaInput::delta_chunk_recalculated(chunk_len)` — the entry point. An
  absent gate is passed as an untracked `[1,1,1,1]` placeholder plus
  `has_gate: false`, because the node's parent list is fixed at 7 while
  `DeltaInput::g_bshK` is an `Option`.
- `DeltaChunkBackendExt: Backend` (`#[backend_extension(...)]`) with
  `delta_chunk_recalculated(q, k, v, erase, write, g, state, has_gate,
  chunk_len, scale) -> (y_bshv, final_state_bhkv)`; the default body is the
  plain forward on `B`'s primitives, which is what every non-autodiff backend
  runs. `DeltaChunkAutodiffBackendExt` from `decl_autodiff_backend_ext!`. The
  backend list is one `Cube` arm covering every cubecl `backend-*` feature
  (mirroring burn's `cube_backend` cfg), plus `Flex`/`NdArray`/`LibTorch`/
  `Remote`/`Autodiff`.

### `forward.rs`
The forward in three replayable stages, so the backward can re-run them rather
than store their results: `Chunked::prepare` (pad, chunk, scale `q`, cumulate
the gate), `Chunked::wy` (the two score matrices, `T`, `U`/`W`/`attn`), and
`Chunked::scan` (the serial inter-chunk recurrence). `ScanMode::States` returns
the state entering each chunk and skips `y`, which is all the backward wants.
Also `Dims`, the layout helpers (`pad_sequence`/`chunked`/`unchunked`/`pick`/
`reduce_to_width`) and `BlockDecayPrim`, the primitive port of `decay.rs`.

### `backward.rs` (`cfg(autodiff)`)
`impl DeltaChunkBackendExt for Autodiff<B, C>`: one `Backward<B, 7>` node whose
`State` is the seven leaf primitives, the shapes, and `(has_gate, chunk_len,
scale)`. The two outputs are flattened into one tracked 1-D tensor
(`burn_stack::utils::combined_grad`) so a single node covers both.

### `combined_backward.rs`
- `combined_backward::<B>(d_y, d_final_state, …leaves…, chunk_len, scale) ->
  DeltaChunkGrads` — recompute, then the gradients.

Three steps are worth naming. **The inverse**: `dT = T dN T`, so
`Ḡ_N = tril(Tᵀ Ḡ Tᵀ, −1)` — two matmuls reading only `T`, never the ladder
(`prepare_wy_repr_bwd_kernel` in `fla/ops/delta_rule/wy_fast.py` is
`dA ← −tril(A · tril(dA) · A, −1)`, the opposite sign convention for `N`).
**The scan**: walked in reverse over the recomputed state stream, because
running the recurrence backwards would divide by `e^G`. **The gate**: `G` is a
cumulative sum, so every use of it accumulates into `Ḡ` first and the reverse
cumsum happens once at the end.

### `tests.rs`
`ChunkRecalculated` against `Chunk { Blocked }` on values *and* gradients, at a
tight tolerance because the two run identical arithmetic: the three gate widths,
`β > 1`, a partial last chunk, `expand_v ≠ 1`, every chunk length from 1, and
the untracked (inference) path. The broader "all paths are the same recurrence"
contract is `delta/tests.rs`, against `Recurrent`.

---

## The four families

Each is one file, `<family>.rs` (block + config); the state they carry between
calls is the shared `DeltaCache` (see `src/common/cache.rs`).

Every block exposes `forward(x_bsd, cache, path)`, `step(x_bd, cache)`,
`zero_caches(batch, n_virtual, device)`, and the dimension accessors
(`nheads`/`head_k_dim`/`head_v_dim`/`value_dim`/`d_model`).

### `src/deltanet/deltanet.rs`
- `DeltaNet { qkv, norm, out_proj }`, `DeltaNetConfig`.
- Parameterised by `expand_k`/`expand_v` ratios of `d_model` (the paper's knobs);
  `scaled_dim` asserts the ratio lands on a whole number of channels.
- `use_beta = false` fixes `β ≡ 1`; `qk_activation`/`qk_norm` configurable.
- `g_bshK: None` at the `DeltaInput` — that is the whole difference from Gated
  DeltaNet.

### `src/gated_deltanet_1/gated_deltanet_1.rs`
- `GatedDeltaNet1 { qkv, gate: ForgetGate, norm, out_proj }`, `GatedDeltaNet1Config`.
- Parameterised by `head_k_dim` + `expand_v` + `n_value_heads` (grouped values).
- The gate's raw `Δ` rides in the fused projection as its `extra_channels`
  segment.
- Always SiLU + L2 on `q`/`k`; output gate on by default.

### `src/gated_deltanet_2/gated_deltanet_2.rs`
- `GatedDeltaNet2 { qkv, gate: ChannelForgetGate, out_gate: Option<Linear>,
  norm, out_proj }`, `GatedDeltaNet2Config`; `bottleneck()`, `n_householder()`.
- Same knobs as v1 (`head_k_dim`, `expand_v`, `n_value_heads`) plus the
  bottleneck rank (`0` ⇒ `head_v_dim`, the reference's choice).
- `n_householder` (default 1) takes DeltaProduct's micro-step fold onto these
  gates: `forward`/`step` place `q` on the last micro-step and the per-channel
  `g` on the first exactly as `delta_product.rs` does, and `step` keeps its
  direct `delta_step` call at `u = 1`. `muon_projections` then lists each
  factor's `k`/`v`/`erase`/`write` map separately.
- `allow_neg_eigval: Option<bool>`, resolved by `allow_neg_eigval_resolved()`
  to `n_householder > 1` when unset — the third sentinel knob here, beside
  `n_value_heads` and `bottleneck`.
- Two low-rank bottlenecks, `Δ` and the output gate: their **first** factors are
  ordinary `extra` segments of the fused projection, their second factors are
  `ChannelForgetGate::up` and `out_gate`.
- Erase (`head_k_dim`) and write (`head_v_dim`) gates in place of the scalar
  `β`, so deleting one fact and accumulating into another are single-token
  operations — the property the family exists for, asserted directly on
  `delta_step`.

### `src/delta_product/delta_product.rs`
- `DeltaProduct { qkv, gate: Option<ForgetGate>, norm, out_proj }`,
  `DeltaProductConfig`; `n_householder()`.
- `forward`/`step` build the unrolled inputs: `q` on the **last** micro-step
  (zero elsewhere), `g` on the **first** (zero elsewhere), both by `cat` with a
  zero block; the output keeps only each token's last micro-step.
- `step` runs the token's `u` micro-steps as a length-`u` `DeltaPath::Recurrent`
  pass.
- `allow_neg_eigval` defaults **on**; `n_householder` defaults to 2.

---

## `src/unified/` — the runtime-selectable API

### `mod.rs`
Module header: what the runtime enums are for, and what Muon does and does not
see. Declares the submodules and re-exports.

### `cache.rs`
- `impl CacheStack for DeltaCaches` — once, for every family.
  `cache_to_inner`/`cache_from_inner` are spelled out field by field:
  `Module::map` is a no-op on the bare `Tensor`s a cache holds.
- `impl_block_for_family!` — one macro emitting `Block` + `BlockConfig` for all
  four families; the bodies are the same text four times over. No family unties
  a parameter (nor `DeltaBlock`): `untied_params` is empty, `init_block` ignores
  the application count.

### `block.rs`
- `DeltaBlock` (a `Module` enum over the four family blocks) + `impl Block`;
  `family_name`, `d_model`/`nheads`/`head_k_dim`/`head_v_dim`, `forward`,
  `step`, `zero_caches`.
- `DeltaBlockConfig` (a `Config` enum over the four family configs) +
  `impl BlockConfig`; `init`, `d_model`, `muon_projections`.

The families agree on their cache, their options (`DeltaPath`) and their
interface, so a runtime choice fits *inside* the block and every container above
it is used as-is. The enum's variant name enters the parameter paths, which is
why a `ProjSpec` matches container and weight separately.

### `network.rs`
- `DeltaNetworkShape` / `DeltaLatentShape` / `DeltaVocabShape` — **aliases** of
  `burn_stack::modules::{NetworkShape, LatentShape, VocabShape}`. Nothing about
  a layer stack is delta-specific, so the knobs (real/virtual layers,
  `grad_horizon`, class markers, residuals, mlp, init, the I/O boundary) are
  declared and documented once, there; `build<C: BlockConfig>` on a shape is
  what a statically named family uses.
- `DeltaLatentNet` / `DeltaVocabNet` — aliases of `burn-stack`'s
  `LatentNetwork`/`VocabNetwork` at `DeltaBlock`. What this file *defines* is
  `DeltaLatentNetConfig` / `DeltaVocabNetConfig` (`{ shape, block }` — the
  family is chosen in `block`), whose `init` + `muon_plan` forward to the
  shape's.
- The shape's `init` applies its `InitPolicy` after building, filling in the
  residual depth the policy cannot know: layers × branches per layer (2 with an
  MLP, 1 without).
- A trailing `mod model_config_ext` implements `burn_stack::modules::
  ModelConfigExt` (`init` + `muon_plan`) for both configs, forwarding to the
  inherent methods. It has to live in the lib: the trait is `burn-stack`'s and
  the types are this crate's, so an example crate cannot own it (orphan rule).

`grad_horizon` is a `burn_stack::utils::GradHorizon` — per weight set, not a
suffix length.

### `bidi.rs`
- `DeltaBidiShape` = `burn_stack::modules::BidiShape` (its own shape, not a
  `NetworkShape`: per-pair merges, a `BidiSchedule`, no mlp/init), `DeltaBidiLayers`
  = `BidiLayers<DeltaBlock>`, and the `DeltaBidiLayersConfig` defined here
  (`init`, `muon_plan`). No `step` — the reversed pass is non-causal.

### `tests/`
`layers.rs` (stack parity, virtual layers, MLP, multi-gate, `grad_horizon`,
split forward), `network.rs` (latent/vocab parity, every family through the
block enum), `bidi.rs` (shapes, and that the first output moves
when the *last* token changes), `optim.rs` (declared seams cover the real
projection; per-head scalars excluded; DeltaProduct's `u` maps listed
separately).

---

## `benches/layer.rs`

Single-block criterion benches in the three modes (`forward` / `train` /
`step`), across the families, the `TriSolve` variants, `Recurrent` against
`Chunk`, and DeltaProduct's `n_householder`. Sizes and criterion's sampling
come from the environment. Each case builds its block *inside* the closure
criterion only calls for cases passing its filter, and drains the device once
per measured batch rather than per iteration (`timed`), so an async backend is
measured at steady state.

## `scripts/gen_fixture.py`

Runs `flash-linear-attention`'s naive delta-rule implementations at float64 on a
seeded input and writes `src/delta/tests/reference.rs`. Needs torch and the FLA
checkout at `../flash-linear-attention`.
