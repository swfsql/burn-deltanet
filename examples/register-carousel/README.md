# Register-carousel

The rung above [`register-majority`](../register-majority/README.md): the same
three-register file, the same six state scalars, one new instruction.

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

## Usage

```bash
# the 3-cycle: needs two Householder factors per token
cargo run --release --example register-carousel -- --training --inference

# the rung below: R is a transposition, which one factor reaches
cargo run --release --example register-carousel -- --training --inference -- --turn swap

# the ablation: the 3-cycle with one factor, which cannot reach it
cargo run --release --example register-carousel -- --training --inference -- --factors 1

# the claims below, measured
cargo test --release --example register-carousel -- --nocapture
```

- See `burn-deltanet/Cargo.toml` for other features or backend information.
- See `burn-deltanet/examples/README.md` for the CLI usage overview.

## Why this task

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

## Measured

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
eigenvalue to alternate with, so it cannot even swap. Both sweeps are over
**real blocks** — each candidate is a complete model whose turn factor is
`(axis, β)` and whose every other weight is the exact construction — not a
matrix-norm proxy. The trained one-factor row is the same bound found the other
way round: gradient descent on the 3-cycle lands at 67.9%, inside the 69.2% the
sweep says is the most any single Householder can do.

`tests.rs` produces all of it: `handmade_block_solves_both_turns` writes every
weight down in closed form (a push is one factor at `β = 1`, a turn one factor
per transposition at `β = 2`), and the two sweeps re-run that same block with
only the turn's factor replaced.

## Notes

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
