# TinyStories (character-level LM)

An auto-regressive **Gated DeltaNet** language model over single **characters**
of
[karpathy/tinystories-gpt4-clean](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean),
a cleaned 2.7M-story subset of [TinyStories](https://arxiv.org/abs/2305.07759)
(GPT-4-generated children's stories, plain ASCII).

The model is deliberately tiny: two Gated DeltaNet blocks (`d_model = 32`,
`nheads = 4`, `head_k_dim = 8`, `expand_v = 2`), each followed by the SwiGLU MLP
the reference architecture puts there, cycled to an 8-deep virtual stack over
Multi-Gate residuals, between a tied character embedding and its transpose.
**38,232 parameters**, of which the embedding is 1,536.

That budget is `burn-mamba`'s `tiny-stories` model (39,632 parameters) to within
4%, on purpose: same corpus, same tokenizer, same window, same optimizer
schedule, so the block is the variable. Everything except the block and its
`model.rs` — the corpus, the windowing, the epoch loops, the sampler — is one
shared copy in `burn_stack::examples::tiny_stories`.

## The model

The family is **Gated DeltaNet 1**: the one the deployed language models use
(Qwen3-Next), and the smallest family with everything text needs — plain
DeltaNet's keyed, *erasing* write (one fact per key, rather than an accumulation
of them) plus the scalar forget gate that lets a head drop what it holds at a
document boundary. Swapping the family is one line in `model.rs`; nothing else in
the example knows which one it got.

Two choices differ from `mnist-class`, and both are because this is a *language*
model:

- **The SwiGLU MLP is on.** The reference delta-rule LMs are Llama's macro
  architecture with the delta rule in place of self-attention, so a mixer-only
  stack is the ablation, not the default. `GatedMlpConfig::from_hidden_ratio(32,
  4)` states the reference sizing; its 256-alignment is not meaningful at
  `d_model = 32` (it would make the MLP eight times the width the rule asks for),
  so the rounding drops to 16 and the realised inner width is 96.
- **No `grad_horizon`.** An LM is scored at *every* position, so leaving most
  applications of a shared weight undifferentiated biases every one of those
  readouts — unlike `mnist-class`, which reads out once at the end.
  `burn-mamba`'s `tiny-stories` README measures that on this same corpus.

The stack also carries the reference's global init (`InitPolicy`: every 2-D
weight from `N(0, 0.02²)`, biases zeroed). That is what keeps the **tied** head
sane at this width: Burn initialises an `Embedding` from `N(0, 1)`, which at
`d_model = 32` would start the logits at variance ~32 — tens of bits per
character — and spend the opening of the LR schedule shrinking them. With the
policy on, the first batch starts at the uniform baseline, `log2(48) = 5.58`
bits/char.

## Vocabulary

The dataset's cleaning pipeline guarantees exactly 74 distinct ASCII characters:
the 52 cased letters plus ``\n !"$',-.0123456789:;?``. Case-folding the letters
leaves **48** tokens, and every one of them actually occurs — so the alphabet is
the corpus's own inventory, not a slice of ASCII:

```text
\n !"$',-.0123456789:;?abcdefghijklmnopqrstuvwxyz
```

There is no `<unk>`, no `<bos>` and no padding class (`pad_vocab_size_multiple =
1`), so every logit the model emits is a character the decoder understands. The
embedding is **tied** (`missing_lm_head = true`): one table answers both "which
character is this" and "which character comes next".

Stories are joined with `"\n\n"` — a blank line, which never occurs *inside* a
story (single `\n` separates its paragraphs), so it is an unambiguous document
boundary, and it is also the prompt used for unconditional sampling.

## Data

The dataset ships as a single 673MB parquet file, which is absurd for an example
this size, so instead of the `HuggingfaceDatasetLoader` path (python + the
`datasets` library + a full sqlite import) the corpus is paged out of the public
[datasets-server](https://huggingface.co/docs/datasets-server) `/rows` endpoint,
100 stories per request (its hard maximum). The normalized text is cached in
`~/.cache/burn-dataset/tinystories-gpt4-clean/<split>-<n>.txt`, so the download
happens once per `(split, story count)` — and is shared with `burn-mamba`'s copy
of the example.

The endpoint is rate limited — the measured budget is ~28 requests per two
minutes, after which CloudFront answers `429` with an HTML body for ~15s at a
time — so the pager paces itself at one page per 4s and retries a failed one with
exponential backoff (30s, doubling, 6 attempts). The default corpus is 43
requests, roughly three minutes; `--train-stories 32768` is 329 requests, closer
to half an hour. All of it is one-time.

Splits follow the dataset card's suggested row ranges (the rows are pre-shuffled,
so a contiguous range is already a random sample): rows `0..10k` are test,
`10k..20k` validation, `20k..` training. The defaults pull 4,096 train and 256
validation stories (~3.4MB of text); `--train-stories` scales that up.

The character stream is cut into non-overlapping windows of `seq_len + 1`; each
window is one training item, scored at **every** position against its next
character (so one window contributes `seq_len` classification examples, and the
reported accuracy is per character).

## Usage

```bash
# debug check in flex (fp32)
cargo check --example tiny-stories

# train and then sample (downloads ~3.4MB of stories on the first run)
cargo run --release --example tiny-stories --features "backend-cuda" -- --training --inference

# a bigger corpus and a longer window
cargo run --release --example tiny-stories --features "backend-cuda" -- --training \
    -- --train-stories 32768 --seq-len 512
```

Downstream flags, all forwarded after the trailing `--` and persisted into the
artifacts' `training_config.json`:

| Flag | Default | Meaning |
|------|---------|---------|
| `--seq-len <n>` | 256 | characters per window (the BPTT length) |
| `--train-stories <n>` | 4096 | stories pulled from the train split |
| `--valid-stories <n>` | 256 | stories pulled from the validation split |
| `--epochs <n>` | 16 | passes over the corpus |
| `--batch-size <n>` | 8 | windows per optimizer step |
| `--no-muon` | off | keep the hidden weight matrices on AdamW instead of [Muon](https://kellerjordan.github.io/posts/muon/) (see `mnist-class`'s README) |

- See `burn-deltanet/Cargo.toml` for other features or backend information.
- See `burn-deltanet/examples/README.md` for the CLI usage overview.

The schedule (batch 8, 16 epochs, cosine `12e-3 → 12e-4` after a 5%-of-an-epoch
warmup, Muon on the hidden matrices) is carried over from `burn-mamba`'s
`tiny-stories`, where it was tuned against this same corpus, window and parameter
budget. It has **not** been re-swept for the delta rule, so it is a starting
point rather than a tuned optimum: at 39K parameters that model was
optimization-limited rather than capacity-limited, and the same is likely here.

## Sampling

`inference.rs` shows the library's two execution modes back to back: the prompt
is consumed by one chunkwise `forward()` (prefill, `DeltaPath::Chunk`), and every
generated character then costs one `step()` against that same cache — O(state)
per token, with no growing KV cache. Sampling is temperature-scaled multinomial
over the full 48-way softmax (`temperature <= 0` is greedy), seeded by
`ChaCha8Rng` so a run is reproducible.

`DeltaVocabNet` is the block-generic `VocabNetwork` at `DeltaBlock`, so the
sampler is `burn_stack`'s own generic one; this example only supplies the path.

`--inference` writes one story per temperature (0.5 / 0.8 / 1.0) plus one
continuation of a fixed prompt into `<artifacts>/inference/`. Training samples a
short story at every small validation check into
`<artifacts>/sample-epoch-{e}-batch-{b}.txt`, so the text can be watched turning
from noise into words into sentences.

## Notes

- Loss is reported both in nats (Burn's cross-entropy) and as **bits per
  character**; the uniform baseline is `log2(48) = 5.58` bits.
- The whole 256-character window is processed in parallel by the chunkwise WY
  path (`chunk_len = 64`), and the 8-deep virtual stack keeps every layer's
  activations, so vram scales with `batch_size · seq_len · n_virtual_layers`.
