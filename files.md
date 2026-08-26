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

`A = −exp(a_log)` makes the gate non-amplifying unconditionally; `dt_bias` is
the inverse softplus of a log-uniform `Δ` spread, so heads start with different
timescales.

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
- `QkvProjection { in_proj, conv, qk_activation, qk_norm, nheads, n_value_heads,
  head_k_dim, head_v_dim, n_householder, has_beta, has_gate, allow_neg_eigval,
  extra_channels }`, `QkvProjectionConfig`.
- `forward(x_bsd, window) -> (Qkv, next_window)`, `step(x_bd, window) -> (QkvStep,
  next_window)`.
- `Qkv { q_bshk, k_bShk, v_bShv, beta_bSh, gate_bshv, extra_bsx }`;
  `QkvStep` is the same with an explicit micro-step axis instead of a sequence.
- `segments() -> Vec<(&'static str, usize)>` — the column widths, and the single
  source of truth for both the forward's split and the Muon seams.
- `key_dim`, `value_dim`, `conv_dim`, `d_in_proj`, `state_heads`,
  `heads_per_group`, `zero_conv_window`.

Three decisions live here:
- **The micro-step fold.** `k`/`v`/`β` are projected `n_householder`-wide and
  returned at length `sequence · u`, in micro-step order. `u = 1` is the plain
  sequence, so nothing downstream has a special case.
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
- `DeltaPath { Recurrent, Chunk { chunk_len: Option<usize>, solve: TriSolve } }`
  — `Default` is `Chunk { None, Doubling }`; `DEFAULT_CHUNK_LEN = 64`;
  `chunk()`, `chunk_len(n)`, `resolved_chunk_len()`.
- `DeltaInput { q_bshk, k_bshk, v_bshv, beta_bsh, g_bsh: Option<_>, state_bhkv,
  scale: Option<f64> }` + `dims()`, `resolved_scale()` (`1/√head_k_dim`),
  `sanity()`, `run(path) -> (y_bshv, final_state_bhkv)`.

`q`/`k` arrive activated and normalised but **not** scaled — `scale` is applied
inside, so the same bundle reads identically on either path.

### `recurrent.rs`
- `delta_step(q_bhk, k_bhk, v_bhv, beta_bh, g_bh: Option<_>, state_bhkv, scale)
  -> (y_bhv, next_state)` — the definition, and the primitive every family's
  `step()` decodes with. Retrieval and readout are `[1,k]@[k,v]` matmuls rather
  than broadcast-and-sum, so the backend sees GEMMs.
- `DeltaInput::delta_recurrent()` — unrolls it.

### `chunk.rs`
- `DeltaInput::delta_chunk(chunk_len, solve)` — the chunkwise WY algorithm.

Layout is `[batch, nchunks, nheads, chunk_len, ·]`: heads ahead of the chunk axis
so every matmul batches over `(b, n, h)` and acts on the `[chunk_len, ·]` planes.
The gate is threaded as an `Option` throughout — `None` builds no decay tensors
at all rather than multiplying by ones.

### `tri.rs`
- `TriSolve { Doubling, Neumann }`, `unit_lower_inverse<D>(n_strict, solve)`.

`N` is nilpotent, so the series is exact. `Doubling` factors it as
`∏(I + N^{2^j})` — `⌈log₂ L⌉` steps of two matmuls, versus the reference
kernel's `L` serial row updates. The caller owes the strict-lower property;
nothing here re-masks it.

### `tests/reference.rs`
Values captured from `flash-linear-attention`'s `delta_rule_recurrence` and
`naive_recurrent_gated_delta_rule` at float64, for a fixed input. Regenerate
with `scripts/gen_fixture.py`. This is what makes the suite a *port* check
rather than only a self-consistency check.

---

## The three families

Each is `<family>.rs` (block + config) plus `cache.rs`, and each cache is the
same two fields: `conv_bwc: Option<Tensor<3>>` and
`state_bhkv: Tensor<4>` (`[batch, nheads, head_k_dim, head_v_dim]`), with
`from_parts` for the by-hand `CacheStack` conversion, `sanity()`, and the
`*Caches`/`*CachesConfig` collection.

Every block exposes `forward(x_bsd, cache, path)`, `step(x_bd, cache)`,
`zero_caches(batch, n_virtual, device)`, and the dimension accessors
(`nheads`/`head_k_dim`/`head_v_dim`/`value_dim`/`d_model`).

### `src/deltanet/deltanet.rs`
- `DeltaNet { qkv, norm, out_proj }`, `DeltaNetConfig`.
- Parameterised by `expand_k`/`expand_v` ratios of `d_model` (the paper's knobs);
  `scaled_dim` asserts the ratio lands on a whole number of channels.
- `use_beta = false` fixes `β ≡ 1`; `qk_activation`/`qk_norm` configurable.
- `g_bsh: None` at the `DeltaInput` — that is the whole difference from Gated
  DeltaNet.

### `src/gated_deltanet/gated_deltanet.rs`
- `GatedDeltaNet { qkv, gate: ForgetGate, norm, out_proj }`, `GatedDeltaNetConfig`.
- Parameterised by `head_k_dim` + `expand_v` + `n_value_heads` (grouped values).
- The gate's raw `Δ` rides in the fused projection as its `extra_channels`
  segment.
- Always SiLU + L2 on `q`/`k`; output gate on by default.

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
- `DeltaCaches { DeltaNet, GatedDeltaNet, DeltaProduct }` (plain runtime state,
  not a `Module`) + `family_name()`, `slot_count()`.
- `impl_block_for_family!` — one macro emitting `CacheStack`, `Block` and
  `BlockConfig` for all three families. They differ in what they *project*, not
  in what they *carry*, so the wiring is the same text three times over.
- `cache_to_inner`/`cache_from_inner` are spelled out field by field:
  `Module::map` is a no-op on the bare `Tensor`s a cache holds.

### `family.rs`
- `DeltaFamily: Block<Options = DeltaPath>` — `NAME`, `wrap_caches`,
  `unwrap_caches` (panics on a family mismatch, which runtime selection cannot
  check at compile time).

This is what keeps `network.rs`/`bidi.rs` to one line per family: the real
bodies are generic functions over `DeltaFamily`.

### `network.rs`
- `DeltaNetworkShape` — the family-independent stack knobs (real/virtual layers,
  `grad_horizon`, class latents, residuals, mlp), shared by both networks.
- `DeltaLatentShape` / `DeltaVocabShape` — each network's own knobs on top.
- `DeltaLatentNet` / `DeltaVocabNet` (+ `*Config` enums): `forward`, `step`,
  `prime`, `init`, `muon_plan`.
- Private generics `latent_forward`/`latent_step`/`latent_prime`/`vocab_*` that
  every arm delegates to.

### `bidi.rs`
- `DeltaBidiShape`, `DeltaBidiLayers` (+ `Config`): `forward`, `init`,
  `muon_plan`. No `step` — the reversed pass is non-causal.

### `tests/`
`layers.rs` (stack parity, virtual layers, MLP, multi-gate, `grad_horizon`,
split forward), `network.rs` (latent/vocab parity, every family through the
enum, the mismatch panic), `bidi.rs` (shapes, and that the first output moves
when the *last* token changes), `optim.rs` (declared seams cover the real
projection; per-head scalars excluded; DeltaProduct's `u` maps listed
separately).

---

## `benches/layer.rs`

Single-block criterion benches in the three modes (`forward` / `train` /
`step`), across the families, both `TriSolve` variants, `Recurrent` against
`Chunk`, and DeltaProduct's `n_householder`. Sizes and criterion's sampling
come from the environment. Each case builds its block *inside* the closure
criterion only calls for cases passing its filter, and drains the device once
per measured batch rather than per iteration (`timed`), so an async backend is
measured at steady state.

## `scripts/gen_fixture.py`

Runs `flash-linear-attention`'s naive delta-rule implementations at float64 on a
seeded input and writes `src/delta/tests/reference.rs`. Needs torch and the FLA
checkout at `../flash-linear-attention`.
