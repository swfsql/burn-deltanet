# The `register-*` ladder

Two examples on the **same three-register file** — six state scalars, no
convolution, no residual — each the smallest task its block is *needed* for and
that the rung below cannot solve. Read together they isolate, one at a time,
what each piece of the delta-rule recurrence actually buys:

| rung | what only that block can do | what it needs |
|---|---|---|
| [`register-majority`](#register-majority) | retire a superseded value | the **erase** half of the delta rule (`DeltaNet`) |
| [`register-carousel`](#register-carousel) | permute the file with period 3 | **two** Householder factors per token (`DeltaProduct`) |

## The shared shape

Both rungs run one block at `d_model = 4`, `nheads = 1`, `head_k_dim = 3`
(one row per register), `head_v_dim = 2` — a `3 × 2` state, and the model's
entire memory. `q` reads every row at once, `v = (±bit·V, R)` carries the bit
next to a constant reference axis, and `β` is the write-enable.

The three shortcuts both rungs close:

| shortcut | why it is closed |
|---|---|
| read the current symbol | the label is not a function of it |
| read a fixed window | `use_short_conv = false` — there is no convolution |
| read the residual | `ignore_last_residual` — the head sees the block alone |

Each rung then closes one more, and *that* is what the rung is about: the last
row of each "Why this task" table below.

## Usage

```bash
# training and running inference in flex (fp32) — <rung> ∈ majority|carousel
cargo run --release --example register-<rung> -- --training --inference

# the claims below, measured: the hand-built exact solution and the ablations
cargo test --release --example register-<rung> -- --nocapture
```

`register-carousel` also takes downstream flags, forwarded after a second `--`:
`--turn rotate|swap` picks the permutation and `--factors N` the Householder
count, so the rung can be run as its own ablation.

- See `burn-deltanet/Cargo.toml` for other features or backend information.
- See `burn-deltanet/examples/README.md` for the CLI usage overview.

## Reading the tables

Every "Measured" number below comes out of that rung's `tests.rs`, and every
sweep is over **real blocks** — each candidate is a complete model whose one
varied piece is substituted in and whose every other weight is the exact
hand-built construction, not a matrix-norm proxy. `handmade_*` writes every
weight down in closed form, with nothing fitted anywhere.

---

## register-majority

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

### Why this task

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

### Measured

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

### Notes

- **There is no readout gain to sweep.** The block's per-head RMSNorm keeps only
  the output's *direction*, so any scalar gain on `q` or `v` is invisible; the
  decay `α` is the accumulating baseline's entire remaining freedom.
- **`d_model = 4` is what keeps this constructible in closed form.** Seven
  symbols do not fit in four dimensions by accident: each channel the block needs
  — the register axis, the bit, the write-enable — is a function of *one*
  feature, so all of them are affine in the embedding. `fit_affine` asserts the
  fit is exact rather than assuming it.

---

## register-carousel

The rung above: the same three-register file, the same six state scalars, one
new instruction.

`R` **turns the carousel**, permuting the registers under a single input port.
`+` / `-` overwrite the register at the port, `?` reads it back:

```text
  turn = rotate     +  R  +  R  -  R  ?  R  ?  R  ?
  register A (port) +  .  +  -  +  .  -  +  .  -  +
  register B        .  +  .  +  .  -  .  .  -  .  .
  register C        .  .  .  +  .  +  .  -  .  .  -
  target            .  .  .  .  .  .  n  .  p  .  n
```

Only `?` positions are scored, and only once the port holds something (`.`).

### Why this task

A delta-rule transition is a generalised Householder. With the L2 QK-norm
forcing `‖k‖ = 1`, its eigenvalues are `{1, 1, 1−β}` — **all real**, whatever the
axis and whatever `β`. That single fact is the whole ladder:

| what `R` is | order | reachable by |
|---|---|---|
| a contraction, `β ≤ 1` | — | spectrum in `[0, 1]`: nothing oscillates at all |
| a transposition `(A B)` | 2 | **one** factor: `I − 2wwᵀ`, `w = (e_A − e_B)/√2` — needs `β = 2`, i.e. `allow_neg_eigval` |
| a 3-cycle `A ← C ← B ← A` | 3 | eigenvalues `1, e^{±2πi/3}`: **two** factors, `swap(A,C) ∘ swap(A,B)` |

So `--turn swap` is what
[negative eigenvalues](https://arxiv.org/abs/2411.12537) buy, and `--turn
rotate` — the default — is what
[DeltaProduct](https://arxiv.org/abs/2502.10297) buys. Everything else in the
model is held fixed between them: same block, same state, same readout, same
four symbols.

The **`cycle`** eval family is where it bites. It loads the register file and
then reads the port once per turn, so the answer marches around the carousel with
the permutation's own period. A model whose state cannot hold period 3 is wrong
on a third of those queries no matter what else it does.

### Measured

97 parameters at `n_householder = 2`; **six state scalars**. Chance is 50%.

| | random | cycle |
|---|---|---|
| best in-sample lookup, last symbol | 50.9% | |
| — the same over the last 8 symbols (~3.1k windows memorised) | 76.2% | |
| **rotate**, best single Householder (50 axes × 8 `β`) | 84.1% | **69.2%** |
| **rotate**, hand-built two factors, no training | **100%** | **100%** |
| **swap**, best single Householder with `β ≤ 1` | 85.0% | **53.3%** |
| **swap**, hand-built one reflection (`β = 2`), no training | **100%** | **100%** |
| trained, `--turn rotate --factors 1` (the ablation) | 89.4% | **67.9%** |
| trained, either turn, at the factor count it needs | **100%** | **100%** |

The two sweep rows are the claim. `69.2%` is the ⅔ a period-≤2 output scores
against a period-3 target, and `53.3%` is chance: a contraction has no negative
eigenvalue to alternate with, so it cannot even swap. The trained one-factor row
is the same bound found the other way round: gradient descent on the 3-cycle
lands at 67.9%, inside the 69.2% the sweep says is the most any single
Householder can do.

`tests.rs` produces all of it: `handmade_block_solves_both_turns` writes every
weight down in closed form (a push is one factor at `β = 1`, a turn one factor
per transposition at `β = 2`), and the two sweeps re-run that same block with
only the turn's factor replaced.

### Notes

- **Watch the plateau.** A `--turn rotate` run parks at ≈68% on `cycle` — the
  single-Householder ceiling, exactly — for twenty-odd epochs before it finds the
  second reflection and jumps to 100%. That is the `--factors 1` ablation
  happening inside the training curve, and it is why the schedule is 150 epochs:
  at 60 the anneal reaches `min_lr` while the run is still on the plateau, and it
  ends at ≈70% having never left it.
- **The second reflection is a basin you have to find.** The default seed reaches
  it; a run that ends near 70% on `cycle`, or near 90% on `random`, has stalled
  in the one-factor solution — restart it with a different `seed` in the training
  config.
- **Reflection axes are expensive for silu to reach.** Every family but DeltaNet
  applies silu to `q`/`k` before the L2 norm (this is what the reference does),
  and silu's floor is `−0.2785`: a key with a negative component of equal
  magnitude has to sit in a narrow, nearly flat part of that curve. It is exactly
  reachable — the hand-built construction does it — but it is why the ablation
  plateau is as sticky as it is.
- **No forget gate.** `use_forget_gate = false`: a turn is a permutation, and
  `α ≡ 1` also keeps the negative claim clean, since the transition is then a
  pure Householder product with no decay to hide behind.

---

## Notes shared by the rungs

- **`v` carries a constant reference channel** next to the bit. A lone bit is
  one-dimensional and the block's per-head RMSNorm would flatten it to a hard
  sign, which has no usable gradient; the reference axis keeps the normalised
  output a *direction* whose angle moves with the answer.
- **The state is the only memory.** With no short convolution and no residual
  reaching the head, a window lookup is not merely discouraged — it is not
  representable. The "best in-sample lookup" rows measure what a model that
  memorised every window *of the training set* would score, which is the bar a
  shortcut would have to clear.
