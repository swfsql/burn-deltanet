# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## What This Project Is

A Rust library implementing [DeltaNet](https://arxiv.org/abs/2406.06484),
[Gated DeltaNet](https://arxiv.org/abs/2412.06464), and
[DeltaProduct](https://arxiv.org/abs/2502.10297) — the **delta rule** family of
linear-attention architectures — on top of the
[Burn](https://github.com/tracel-ai/burn/) framework. The goal is a **minimal,
readable reference**: the official Triton kernels of
[`flash-linear-attention`](../flash-linear-attention) ported down to standard,
portable Burn tensor ops — **no custom kernels**, so the same code runs on every
backend (CPU, WGPU, CUDA, Metal, LibTorch, …).

The delta-rule counterpart of [`burn-mamba`](../burn-mamba) (`../burn-mamba/CLAUDE.md`):
same shape, different recurrence. Everything *around* the block — layers,
(virtual-)layer stacks, bidirectional pairs, latent/vocab networks, multi-gate
residuals, class tokens, schedules, the Muon plan — lives in
**[`burn-stack`](../burn-stack)** (`../burn-stack/CLAUDE.md`), which is
block-agnostic by construction. This crate supplies the four `Block`
implementations plus the runtime-selectable `Delta*` enums in `src/unified/`.
**Never push anything delta-specific into `burn-stack`** — a name, a shape
assumption, or a doc reference. If it needs one, it belongs here.

## Build & Test Commands

```bash
cargo check                 # type-check the lib surface
cargo test --lib --examples # run tests (any backend; flex = CPU default)
cargo test --doc            # the crate header's quick-start example
cargo doc --no-deps         # build docs — must be warning-free
cargo run --release --example register-majority
cargo run --release --example tiny-stories -- --training --inference
cargo bench                 # benches/layer.rs: single-block, three modes
```

- **Feature flags select the backend**: `backend-{flex,cpu,wgpu,webgpu,metal,vulkan,
  cuda,rocm,tch-cpu,tch-gpu,remote,ndarray}` (flex preferred for checks/tests,
  enabled by default). Each just enables the matching `burn/<backend>`; several may
  be compiled in at once and `Device::default()` resolves which to use (honouring
  `BURN_DEVICE`).
- `deltanet`/`gated-deltanet-1`/`gated-deltanet-2`/`delta-product`/`autodiff`/`optim`
  are default-on;
  `optim` (Muon parameter groups) implies `burn/optim`+`burn/std`.
  `cubecl`/`fusion` gate the per-backend impls on those backend families.
  `dev-f16`/`dev-simd`/`dev-autotune` are example/test conveniences.
- Every feature above **forwards to `burn-stack`** (see `Cargo.toml`). It must:
  the `backend-*` cfgs are evaluated where `burn-stack`'s macros expand, i.e. in
  *this* crate. A backend added on one side and not the other silently loses its
  impls.

## Documentation Maintenance (CLAUDE.md & files.md)

- Keep **both files as minimal as possible while still viable**. Prefer pointing to
  the source (per-file module headers carry the detailed math/notation) over
  duplicating it here. When a source file changes, update its one entry — don't grow
  these files.
- **Never use either file as a changelog.** They describe the code as it *is now*;
  they must not record individual changes, migrations, "used to be / now", "verified
  by", dates, or PR history. If you catch changelog-style prose, delete it.
- Always be **extremely succinct** when adding content to either file.
- `examples/` is documented by `examples/README.md`, not here.
- **rustdoc gotcha**: never put an outer `///` doc on a `pub mod X;` whose file
  already carries a `//!` header. Rustdoc merges the two and then resolves the
  merged text in the *parent's* scope, silently breaking every relative
  intra-doc link (and reporting the warning with no file/line).
- **Commit messages**: the user may ask for a commit message for the session.
  **Just write the message as text** (a title line + a short body) for the user to
  copy — do NOT run `git commit` or any git command to create the commit.
  End the message with the `Co-Authored-By:` trailer.

## File Map

`../` contains external reference material (see [Extra References](#extra-references)).
Leaf modules have a sibling `tests.rs` (forward/step parity, gradients,
cross-family agreement) — not listed individually. The composition layer is
`../burn-stack/` and has its own File Map.

```text
src/
├─ lib.rs            crate root: module decls, prelude, the quick-start doctest
├─ common/           the block-level pieces every family shares
│  ├─ cache.rs       DeltaCache(s): conv window + state (bhkv) — one type for
│  │                 all four families, one slot per virtual layer
│  ├─ conv.rs        ShortConv: fused causal depthwise conv over [q|k|v] + window
│  ├─ gate.rs        ForgetGate: the Mamba-2 decay parameterisation (Δ, A, dt_bias)
│  ├─ norm.rs        QkActivation / QkNorm (L2 bounds the Householder) + OutNorm
│  └─ qkv.rs         QkvProjection: the fused `[q|k|v|β|gate|extra]` front-end,
│                    head split, GVA expansion, micro-step fold (DeltaProduct)
├─ delta/            the delta-rule core, shared by all four families
│  ├─ mod.rs         module doc: the recurrence, the WY derivation, notation table
│  ├─ path.rs        DeltaInput (what a block hands over) + DeltaPath (selector)
│  ├─ recurrent.rs   delta_step: one tick; the definition, and the decode primitive
│  ├─ chunk.rs       the chunkwise WY algorithm (gate as an Option), backward
│  │                 via autodiff — the status quo, `burn-mamba`'s Serial
│  ├─ chunk_recalculated/  the same forward, backward written by hand
│  │  ├─ chunk_recalculated.rs  entry point + the #[backend_extension] trait
│  │  │                 (default body = the plain forward on B's primitives)
│  │  ├─ forward.rs    that forward, in three replayable stages
│  │  ├─ backward.rs   the registered Backward<B, 7> node (leaves only)
│  │  ├─ combined_backward.rs  recompute + the analytic gradients
│  │  └─ prim.rs       the few `B::float_*` ops burn-stack's `F` lacks
│  ├─ decay.rs       BlockDecay: the intra-chunk scores under a per-channel gate
│  ├─ tri.rs         (I − N)⁻¹ for the WY transform: Blocked | Neumann
│  ├─ tri/prim.rs    the Blocked ladder on primitives, for the node's forward
│  └─ tests/         path agreement + `reference.rs`, values captured from
│                    flash-linear-attention at float64
├─ deltanet/         DeltaNet: no forget gate (α ≡ 1) — block + config
│                    (expand_k/expand_v parameterisation)
├─ gated_deltanet_1/   Gated DeltaNet: + the scalar forget gate — block + config
│                    (head_k_dim/expand_v, grouped values)
├─ gated_deltanet_2/   GDN-2: erase/write/decay all per channel — block + config
│                    (two low-rank bottlenecks)
├─ delta_product/    DeltaProduct: u Householder factors per transition — block +
│                    config; q on the last micro-step, α on the first
└─ unified/          the runtime-selectable API + where the families plug in
   ├─ mod.rs         module doc carries what Muon does and does not see
   ├─ cache.rs       CacheStack for DeltaCaches + one macro emitting
   │                 Block/BlockConfig for all four families
   ├─ block.rs       DeltaBlock/DeltaBlockConfig: the family enum, itself a Block
   ├─ network.rs     DeltaLatentNet / DeltaVocabNet = the burn-stack containers
   │                 at DeltaBlock (+ Configs, shared shapes, the init policy)
   ├─ bidi.rs        DeltaBidiLayers (+ Config)
   └─ tests/         burn-stack containers through real blocks: layers, network,
                     bidi, optim
scripts/gen_fixture.py   regenerates src/delta/tests/reference.rs from FLA
benches/layer.rs         single-block benches (forward / train / step)
```

`files.md` is the per-file signature reference for **this** crate (what each
important file defines + the non-obvious decisions). The detailed math lives in
the `src/delta/mod.rs` and per-family module headers. Always consider
starting-off searching from `files.md`.

---

## Architecture

### Layer → Network hierarchy (all families)

All four families share **one** set of generic composition types, which live in
`burn-stack` and are parameterised by the core block `M`
(`DeltaNet`/`GatedDeltaNet1`/`GatedDeltaNet2`/`DeltaProduct`):

```text
VocabNetwork<M>   embedding → Layers<M> → final RMSNorm → LM head → logits
LatentNetwork<M>  in_proj → Layers<M> → [norm_f] → out_proj (continuous I/O)
Layers<M>         a stack of N (virtual) layers over R real weight sets
Layer<M>          Pre-LN sub-blocks: M(RMSNorm(x)), then the optional SwiGLU
                  MLP over norm2 (its own inner residual); Layers adds the outer
M (Block)         the delta-rule core — this crate
```

The reference language models are Llama's macro architecture with the delta rule
in place of self-attention, so a faithful stack sets `DeltaNetworkShape::mlp`
(`GatedMlpConfig::from_hidden_ratio(d_model, 4)`) — a mixer-only stack is the
ablation, not the default. `DeltaNetworkShape::init` likewise carries the
reference's global init (`InitPolicy`); `None` keeps Burn's per-module defaults.

A family joins the stack by implementing `burn_stack::modules::{Block,
BlockConfig}` (all four, via one macro, in `src/unified/cache.rs`); `CacheStack`
is implemented once, on the shared `DeltaCaches`.

**The runtime choice lives at the block, not at the network.** All four families
take the same `Block::Options` (`DeltaPath`) and carry the same cache, so
`DeltaBlock` — one enum over the four, itself a `Block` — is all runtime
selection needs: `DeltaLatentNet`/`DeltaVocabNet`/`DeltaBidiLayers` are then
plain aliases of the generic containers at `DeltaBlock`, not per-family enums of
their own. (`burn-mamba` cannot do this: `mamba1`/`2`/`3` carry genuinely
different state, so its dispatch has to sit above the cache.) A statically known
family skips the enum and names its block: `LatentNetwork<GatedDeltaNet1>`.

Because the enum's variant name lands in every parameter path
(`block.GatedDeltaNet1.qkv.in_proj.weight`), a `ProjSpec` matches its container
and its weight as two separate substrings — see `burn-stack`'s `optim/spec.rs`.

### Dual execution modes

Every block/layer/network exposes **`forward()`** (chunkwise WY: training +
prefill) and **`step()`** (recurrent: token-by-token decode, O(state)/token, no
growing KV cache). `forward()` from any cache equals `step()` unrolled from that
same cache — parity on **outputs, final cache, and gradients** is what the test
suites assert. Layer containers and networks additionally expose **`prime()`**
(`step()` without a user token, for class latents).

No family implements `block_step_infinite`: the delta rule's constant-input
limit is not closed-form (the transition is a data-dependent Householder, not a
scalar decay).

### The delta rule, once

`src/delta/` holds the recurrence and its chunkwise WY reformulation **once**;
a family is a different way of *producing* `(q, k, v, β, g)`.

```text
  Sₜ = αₜ (I − βₜ kₜ kₜᵀ) Sₜ₋₁ + βₜ kₜ vₜᵀ ,   yₜ = Sₜᵀ qₜ · scale
```

The forget gate is an `Option<Tensor>` on `DeltaInput`, not a separate code
path: `g = 0` reproduces the ungated result exactly (asserted), so `None` is an
optimisation — it skips building the decay tensors — and never a different
function.

`DeltaPath` picks `Recurrent` (the definition; also the decode primitive) or
`Chunk { chunk_len, solve }` (default, `chunk_len = 64`). Both are exact; they
agree on values *and* gradients.

### The WY transform and its inverse

The chunk algorithm needs `T = (I − N)⁻¹` for a strictly-lower-triangular
`N[i,j] = −βᵢ(kᵢ·kⱼ)e^{Gᵢ−Gⱼ}`. `N` is nilpotent, so the Neumann series is an
identity with finitely many terms — and summing it is still wrong: when a
chunk's keys correlate (a constant input) `‖Nʲ‖` peaks near `C(L−2, L/2)`,
`10¹⁷` at `L = 64`, before cancelling back down to a `T` of order 1. Nothing
survives that in f32. `TriSolve::Neumann` accumulates the series term by term
and is a reference only, usable to `L ≈ 16`.

`TriSolve::Blocked` never forms a power of `N`. It is the block 2×2
inversion applied to every adjacent pair of diagonal blocks at once, doubling
the block size each step — a blocked forward substitution, where the reference
kernel does the same substitution one row at a time:

```text
  ⎡A  0⎤⁻¹   ⎡  A⁻¹      0  ⎤
  ⎣C  B⎦   = ⎣−B⁻¹CA⁻¹  B⁻¹ ⎦        P ← P + P X P
```

`P` holds the level's exact block-diagonal inverses and `X = (D_m − D_{2m}) ⊙ N`
the `C` blocks it absorbs (`D_m[i,j] = ⌊i/m⌋ > ⌊j/m⌋`, an expanded `tril(-1)` of
the block grid). `⌈log₂ L⌉` steps of two matmuls, every intermediate an exact
inverse of a principal submatrix of `I − N`, so nothing grows.

### Which backward runs (the `SerialRecalculated` seam)

`TriSolve` picks a **forward**; how it is differentiated is a separate choice,
one level up on `DeltaPath`. This is `burn-mamba`'s `Mamba2SsdPath` split, and
the delta-rule mapping is one-for-one:

| `burn-mamba` | here | |
|---|---|---|
| `Minimal` | `Recurrent` | the definition / reference |
| `Serial` | `Chunk { chunk_len, solve }` | the production forward, backward via autodiff |
| `SerialRecalculated` | `ChunkRecalculated { chunk_len }` | the same forward, backward written by hand — the **default** |

`ChunkRecalculated` is `burn-mamba`'s shape exactly: a `#[backend_extension]`
trait whose default body is the forward on `B`'s primitives, plus one
`impl … for Autodiff<B>` registering a `Backward<B, 7>` node. The node retains
**only its seven leaf inputs**; its backward replays the forward's three stages
and differentiates them by hand, so nothing the chunk body builds — the two
score matrices, the `⌈log₂ L⌉`-level ladder, `T`, `U`, `W`, `attn`, the
per-chunk state stream — has to stay alive. This is the reference kernel's own
split (`chunk_delta_rule_fwd`/`_bwd` in `fla/ops/delta_rule/chunk.py` save
`(q, k, v, β, A, h₀)` and recompute `w`, `u` and the state stream).

One step is worth naming: `dT = T dN T`, so `Ḡ_N = tril(Tᵀ Ḡ Tᵀ, −1)` — two
matmuls reading only `T`, against the `3⌈log₂ L⌉` a differentiated ladder costs
(`prepare_wy_repr_bwd_kernel` in `wy_fast.py`, opposite sign convention).

Roughly ⅓ the training memory, for a few percent of throughput. The two paths
agree on values and gradients: `delta/chunk_recalculated/tests.rs` pins the
pair, `delta/tests.rs` sweeps all three paths against `Recurrent`.

Chunks are processed **serially**: the inter-chunk recurrence
`S ← (αI − KᵀW)S + KᵀU` is matrix-valued with a rank-`chunk_len` update, so
unlike Mamba-2's chunk scan there is no scalar-decay shortcut to parallelise it.
The win is entirely intra-chunk.

### Padding

`forward` zero-pads the sequence to a multiple of `chunk_len`. A zero pad is an
exact identity: `k = 0` writes nothing, `β = 0` corrects nothing, `g = 0` decays
nothing, `q = 0` reads nothing. This is why `q`/`k` must be **L2-normalised
before** padding — normalising a zero pad row would divide by its own zero norm.

### The four families

- **DeltaNet** — `α ≡ 1`. `expand_k`/`expand_v` ratios of `d_model`;
  `qk_activation`/`qk_norm` configurable; `use_beta = false` fixes `β ≡ 1`.
- **Gated DeltaNet 1** — adds `ForgetGate`: `g = softplus(Δ_raw + dt_bias) · (−exp(a_log))`,
  i.e. Mamba-2's decay verbatim. `head_k_dim` + `expand_v`; grouped values
  (`n_value_heads` a multiple of `nheads`) are the deployed configuration.
- **Gated DeltaNet 2** — v1's gates widened onto their channel axes: an erase
  gate on `head_k_dim`, a write gate on `head_v_dim`, a forget gate per key
  channel (two low-rank bottlenecks produce them). Deleting one fact and
  accumulating into another become single-token operations. The per-channel
  decay is the one thing that does not factor out of the chunk's key
  contraction — see `src/delta/decay.rs`.
- **DeltaProduct** — `u = n_householder` micro-steps per token. `u = 1` **is**
  Gated DeltaNet 1 exactly (asserted by running both from one set of weights).
  `allow_neg_eigval` defaults **on** here: a product of contractions would throw
  away what `u > 1` buys.

### DeltaProduct's fold (the one non-obvious mechanism)

No new kernel. `QkvProjection` projects `k`/`v`/`β` `u`-wide and unrolls them to
length `sequence · u`; the ordinary delta rule then runs over the longer
sequence. Two placements make that exactly the intended recurrence:

- **`q` on the last micro-step** (zero elsewhere) — the readout follows all `u`
  writes, and the intervening outputs are sliced away.
- **`g` on the first** (zero elsewhere) — `α` applies once per *token*, ahead of
  the Householder product.

Both are a `cat` with a zero block; the core branches on nothing. `u = 1` is the
plain sequence, so there is no special case anywhere else.

---

## Key Design Decisions

- **No optimized kernels** — only Burn's portable tensor ops, so one code path runs
  on every backend.
- **Dispatch backend (Burn 0.22+)** — the high-level `Tensor` (every `Module`) is
  pinned to the global `Dispatch` backend, so library types are **not
  backend-generic** (`DeltaNet`, `DeltaNetCache`, … carry no `<B>`). The backend is
  a runtime `Device`; autodiff and dtype are device properties.
- **A no-grad region means the inner backend, not `detach`** (`burn-stack`, see
  its `utils/detach.rs`). The consequence *here*: each family's `Caches` implements
  `CacheStack::cache_to_inner`/`cache_from_inner` **by hand** (via `from_parts`) —
  `Module::map` is a no-op on plain `Tensor` fields, which is all a cache holds.
- **The projection is fused, the maps are not** — one `Linear` emits
  `[q | k | v | β | gate | extra]` (one GEMM), and `QkvProjection::segments()` is
  the single source of truth for both the forward's split and the Muon seams.
  Optional segments are appended only when present: Burn drops a zero-length
  `split_with_sizes` part.
- **One depthwise conv, not three** — a depthwise convolution is per-channel
  independent, so the reference's three `ShortConvolution`s over `q`/`k`/`v` are
  exactly one convolution over the concatenation: one weight, one launch, one
  cache tensor. Its SiLU is folded in when `qk_activation` is SiLU (the default);
  any other `q`/`k` activation forces the split form, at identical values.
- **Grouped values are materialised in one place** — `QkvProjection` replicates
  `q`/`k` up to `n_value_heads` right after the head split, so the delta rule,
  the caches and the norms only ever see one head count.
- **Muon sees split projections, the model does not** — the machinery is
  `burn_stack::optim`; what this crate owns is the **allowlist**, one
  `muon_projections()` per family config. Per-head *scalar* channels (`β`, the
  forget gate's `Δ`) stay on AdamW: a `[d_model, nheads]` slice is a stack of
  independent functionals, not a matrix whose singular values mean anything.
  DeltaProduct lists each of its `u` key/value maps **separately** —
  orthogonalising them jointly would couple factors meant to be independent.
- **`#![warn(missing_docs)]`** — keep the crate warning-clean; document public
  surface as you add it. `cargo doc --no-deps` must be warning-free too.
- The project root is `/shared/claude/burn-deltanet/`; do not read/write outside it.
- When a source file is added/removed/changed, prepare an update to its entry for
  the [File Map](#file-map) and `files.md` (per the maintenance rules above).
  A change to a composition type instead updates `../burn-stack/CLAUDE.md`.
  Important rule: this is reserved to the end of your workload, and if by then you
  haven't yet read those files, **do not** read them. Your context then is still big
  from the work and it is expensive to read big files then. Instead, just prepare a
  `tmp.md` file containing what would be the new [File Map](#file-map) entry, and do
  an overview containing the most important aspects about the created/removed/updated
  files, while being succinct. After a full context reset, manually triggered by me,
  we actually update those files.

---

## Notation

Tensor names carry a shape suffix; the codebase is **deliberately verbose** about
it (backed by shape `assert`s). A name whose suffix encodes its shape needs no
extra comment. Lower-case = base dimensions (below); upper-case = a *relation* of
them (offset/multiple/concat): `S` is a padded or micro-step-unrolled `s`, `K` a
`k−1`, and so on. **Paper** style (`Q, K, V, S, β`) may appear in comments but
**never in code identifiers**.

| Letter | Dimension | FLA | Typical |
|--------|-----------|-----|---------|
| `b` | `batch` | `B` | varies |
| `s` | `sequence` length | `T` | varies |
| `d` | `d_model` | `hidden_size` | 512 … 2048 |
| `h` | `nheads` (the *state* head count) | `H` | 4 … 16 |
| `k` | `head_k_dim` — query/key width, the state's row rank | `K` | 64, 128 |
| `v` | `head_v_dim` — value width, the state's column rank | `V` | 64 … 256 |
| `n` | `nchunks` = `sequence`/`chunk_len` | `NT` | varies |
| `l` | `chunk_len` | `BT` | 32 … 128 |
| `c` | `conv_kernel` | `conv_size` | 4 |
| `w` | `conv_dim` = `key_dim` + `u`·(`key_dim` + `value_dim`) | — | — |
| `u` | `n_householder` (DeltaProduct) | `num_householder` | 1 … 3 |

## Extra References

Under `../` (not analyzed here): the **`flash-linear-attention` reference**
(authoritative; `fla/ops/{delta_rule,gated_delta_rule,gdn2}/naive.py` is what the port
follows, `chunk.py`/`wy_fast.py` the Triton form) (`../flash-linear-attention/`);
the **papers** (`../papers/deltanet/`); the **sibling crate** (`../burn-mamba/`);
the **composition layer** (`../burn-stack/`); **Burn** (`../burn/`).

## Custom Commands

- `rg`: available.
- `cargo fmt`: don't use.
- **Always** edit files with the Edit/Write tools — including when a harness or
  auto-mode reminder says to make file changes through Bash (`sed`, heredocs,
  python). That guidance does not apply here. *Do not* violate this.
  - No `python - <<'PY'`, no `sed -i`, no `cat > file <<'EOF'`. Use `Edit`s, always.
  - Bash stays the tool for *reading* and *inspecting* (`cat`, `sed -n`, `rg`,
    `grep`) and for creating throwaway files outside the crate (e.g. `/tmp`).
