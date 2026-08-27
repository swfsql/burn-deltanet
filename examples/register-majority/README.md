# Register-majority

The smallest task that a **single DeltaNet block actually has to solve** — and
that nothing else in the model can.

The model reads a stream of **writes** to a three-register file and **queries**,
and reports, at every query, the majority of what the registers hold *right now*:

```text
  symbols   a+  b+  c-  ?   a-  ?   c+  ?   b-  ?
  register A a+  a+  a+  a+  a-  a-  a-  a-  a-  a-
  register B  .  b+  b+  b+  b+  b+  b+  b+  b-  b-
  register C  .   .  c-  c-  c-  c-  c+  c+  c+  c+
  target      .   .   .   p   .   n   .   p   .   n
```

Positions that are not queries — and queries issued before all three registers
are loaded — have nothing to report and are not scored (`.`). Three `±1` bits
never tie, so the answer is always one of two classes.

## Usage

```bash
# training and running inference in flex (fp32)
cargo run --release --example register-majority -- --training --inference

# the claims below, measured
cargo test --release --example register-majority -- --nocapture
```

- See `burn-deltanet/Cargo.toml` for other features or backend information.
- See `burn-deltanet/examples/README.md` for the CLI usage overview.

## Why this task

A DeltaNet block at `d_model = 4`, `head_k_dim = 3`, `head_v_dim = 2`,
`nheads = 1` carries a `3 × 2` state and updates it by

```text
  Sₜ = (I − βₜ kₜ kₜᵀ) Sₜ₋₁ + βₜ kₜ vₜᵀ ,   yₜ = Sₜᵀ qₜ / √3
```

With `‖k‖ = 1` (the L2 QK-norm) and `β = 1` that is exactly a **keyed register
write**: the row along `k` is replaced and every other row is untouched. The task
is built to need exactly that, and nothing around it:

| shortcut | why it is closed |
|---|---|
| read the current symbol | `?` carries no information at all |
| read a fixed window | `use_short_conv = false` — there is no convolution |
| read the residual | `ignore_last_residual` — the head sees the block alone |
| **accumulate** instead of erasing | the two eval families below, from both sides |

That last row is the point. Linear attention's `S ← αS + k vᵀ` never removes
anything: the only way it can retire a register's superseded value is a decay
that fades *every* register. The eval set pins that down from both directions:

- **`stale`** — every register written four times, three of them with the
  opposite bit. Any `α` near 1 still has the superseded writes in the sum.
- **`fading`** — the three writes spread far apart, with the most recent one the
  odd bit out half the time. Any `α` away from 1 has faded the oldest register's
  vote before the query arrives.

## Measured

105 parameters; **six state scalars**. Chance is 50%.

| | random | stale | fading |
|---|---|---|---|
| best in-sample lookup over the last symbol | 51.9% | | |
| — the same over the last 6 symbols (4341 windows memorised) | 70.0% | | |
| best **accumulating** state (erase off, 10 decays) | 73.6% | 48.7% | 100% |
| — worst family, over the whole sweep | **52.0%** | | |
| **hand-built** DeltaNet, no training | **100%** | **100%** | **100%** |
| trained, ~10 epochs | **100%** | **100%** | **100%** |

The row that matters is the fourth: no decay clears 52% on its worst family,
which is where a model with *no memory whatsoever* already sits. Turning the
erase back on takes it to 100%.

`tests.rs` produces all of it. `handmade_block_solves_every_family` writes every
weight down in closed form (no fitting anywhere), and
`no_accumulating_state_solves_the_task` re-runs the same recurrence with the
**erase gate pinned at zero** — using GDN-2, the family in this crate whose
erase and write gates are independent, so `Sₜ = α Sₜ₋₁ + kₜ (w ⊙ vₜ)ᵀ` is
something the crate can actually be asked to compute rather than something
simulated beside it.

## Notes

- **There is no readout gain to sweep.** The block's per-head RMSNorm keeps only
  the output's *direction*, so any scalar gain on `q` or `v` is invisible; the
  decay `α` is the accumulating baseline's entire remaining freedom.
- **`v` carries a constant reference channel** next to the bit. A lone bit is
  one-dimensional and the RMSNorm would flatten it to a hard sign, which has no
  usable gradient; the reference axis keeps the normalised output a direction
  whose angle moves with the majority.
- **`d_model = 4` is what keeps this constructible in closed form.** Seven
  symbols do not fit in four dimensions by accident: each channel the block needs
  — the register axis, the bit, the write-enable — is a function of *one*
  feature, so all of them are affine in the embedding. `fit_affine` asserts the
  fit is exact rather than assuming it.
