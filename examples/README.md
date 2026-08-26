# burn-deltanet Examples

#### List of Examples

- `associative-recall`: A tiny Gated DeltaNet language model on MQAR-style
  associative recall — read a stream of key/value pairs, then answer a query
  key. This is the capability the delta rule exists for: linear attention
  accumulates `Σ kᵢ vᵢᵀ` and lets writes interfere, while the delta rule removes
  the association currently held at `kₜ` before writing the new one. Two layers,
  `d_model = 64`; the defaults reach **~98% held-out accuracy in about 80
  seconds** on a CPU. `--family deltanet | gated | product` switches families
  through the runtime-selectable [`DeltaVocabNet`], and the run ends by decoding
  the held-out problems token by token with `step()` and comparing against the
  chunked `forward()`.

#### Structure

Each example is a **single self-contained file** — dataset, model, training loop
and evaluation. There is no shared `common/` harness: the point of an example
here is to be read top to bottom, and a task small enough to demonstrate a
recurrence is small enough not to need one.

#### Model definition

Examples build the runtime-selectable networks from `burn_deltanet::unified`
(`DeltaVocabNet` for token models, `DeltaLatentNet` for continuous I/O) rather
than defining their own — the same types a caller reading a family out of a
config file would use. A model that knows its family at compile time should name
the generic container directly instead: `VocabNetwork<GatedDeltaNet>`.

#### Backend selection

A backend is selected by feature — `backend-flex` (CPU, the default),
`backend-cuda`, `backend-wgpu`, … See `Cargo.toml`'s `[features]` section for
the full list. `Device::default()` resolves which enabled backend to use, and
honours the `BURN_DEVICE` environment variable.

```bash
cargo run --release --example associative-recall --features backend-flex
```

Extra development features are available alongside: `dev-f16` (half precision),
`fusion`, `dev-autotune`.
