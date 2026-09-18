# TinyStories (character-level LM)

An auto-regressive **Gated DeltaNet** language model over single **characters**
of
[karpathy/tinystories-gpt4-clean](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean),
a cleaned 2.7M-story subset of [TinyStories](https://arxiv.org/abs/2305.07759)
(GPT-4-generated children's stories, plain ASCII).

The model is deliberately tiny: two Gated DeltaNet blocks (`d_model = 32`,
`nheads = 3`, `head_k_dim = 8`, `expand_v = 2`), each followed by the SwiGLU MLP
the reference architecture puts there, joined by Multi-Gate residuals, between a
tied character embedding and its transpose. **33,872 parameters**, of which the
embedding is 1,536 and the four story-opening class latents 128.

That is `burn-mamba`'s `tiny-stories` model (39,760 parameters) to within 15%,
on purpose: same corpus, same tokenizer, same window, so the block is the
variable. Everything except the block and its `model.rs` — the corpus, the
windowing, the epoch loops, the sampler — is one shared copy in
`burn_stack::examples::tiny_stories`. The budget is not matched exactly because
the block's own sizing is fixed by the reference rather than by the target: with
the output gate on, `fla/layers/gated_deltanet.py` asks for
`nheads · head_k_dim = 0.75 · d_model` and `expand_v = 2`, which puts a layer at
`6·d_model²` — a Transformer layer's budget — and leaves `key_dim = 24`,
`value_dim = 48` at this width.

The remaining headroom under a 40K budget is left **unspent on purpose**. A
wider MLP, `expand_v = 3`, a fourth head, an 8-wide conv and a third layer were
each measured, and none of them paid. This model is limited by its optimizer,
not by its capacity.

Every screening quoted in this file — the ones above and the optimizer notes
below — was measured on the **previous** corpus layout: one continuous
`"\n\n"`-joined character stream cut into stateless windows, with no class
latents. Story-per-item scoring changes what the average is over, so read them as
a ranking of levers rather than as current numbers.

## The model

The family is **Gated DeltaNet 1**: the one the deployed language models use
(Qwen3-Next), and the smallest family with everything text needs — plain
DeltaNet's keyed, *erasing* write (one fact per key, rather than an accumulation
of them) plus the scalar forget gate that lets a head drop what it holds at a
document boundary. Swapping the family is one line in `model.rs`; nothing else in
the example knows which one it got.

Both examples build the same reference layer — Pre-LN mixer, Pre-LN SwiGLU MLP
of `GatedMlpConfig::from_hidden_ratio(d_model, 4)` width, a final norm before
the head, the reference's global init — so what differs here is only what the
*task* changes:

- **No virtual stack, and so no `grad_horizon`.** `mnist-class` cycles its 2
  real layers over 16 virtual ones; here 2, 4 and 8 applied layers score alike
  while the deeper ones cost proportionally more time per step, so the stack is
  left at its 2 real layers and there is nothing to truncate. Virtual depth is
  free in parameters and is `burn-mamba`'s largest structural win on this same
  corpus — it simply is not one for the delta rule.
- **Multi-Gate residuals** instead of the plain additive skip `mnist-class` (and
  the reference) uses. This is the one structural departure from the reference
  architecture kept here: two pooled streams between the layers cost three
  vectors each and measurably beat the single skip.

  `n_stream` has to be read against the layer count. MGR *accumulates* one new
  stream per layer until it holds `n_stream` of them and only then starts
  mixing, so any `n_stream` above the layer count never mixes at all — it
  degenerates into a pooled concatenation of the layer outputs, and measures
  well below the plain skip. `n_stream = 2` is what actually mixes on a 2-layer
  stack, and it is the setting that wins.

The MLP's `ratio = 4` is the reference figure, but its 256-alignment is not
meaningful at `d_model = 32` (it would make the MLP eight times the width the
rule asks for), so the rounding drops to 16 and the realised inner width is 96.

The global init (`InitPolicy`: every 2-D weight from `N(0, 0.02²)`, biases
zeroed) is also what keeps the **tied** head sane at this width: Burn
initialises an `Embedding` from `N(0, 1)`, which at `d_model = 32` would start
the logits at variance ~32 — tens of bits per character — and spend the opening
of the LR schedule shrinking them. With the policy on, the first batch starts at
the uniform baseline, `log2(48) = 5.58` bits/char.

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

A story's start is marked out of band, by four learnable **class latents**, not
by a character — see [Story boundaries](#story-boundaries).

## Data

The dataset is a single 673MB parquet file: one column (`text`), one row per
story, 2,669 ZSTD row groups of 1,024 rows. It is downloaded **whole**, once, the
same way `mnist-class` downloads its IDX files, into
`~/.cache/burn-dataset/tinystories-gpt4-clean/`; only the row groups a request
touches are decompressed. The stories that come out are normalized and cached
again as text, one file per `(split, story count)` — so every later run reads a
few MB of text and never opens the parquet at all. Both caches are shared with
`burn-mamba`'s copy of the example.

Splits follow the dataset card's suggested row ranges (the rows are pre-shuffled,
so a contiguous range is already a random sample): rows `0..10k` are test,
`10k..20k` validation, `20k..` training. The defaults pull 4,096 train and 256
validation stories (~3.4MB of text); `--train-stories` scales that up at no extra
download.

## Story boundaries

**One item is one story** (303–4,149 characters, median 724), stripped of its
surrounding whitespace, and nothing is spliced between two of them: a story is a
self-contained example. What marks its start is four `ClassLatent::Start`
registers — learnable `d_model`-wide rows the stack prepends to the sequence — and
the **last of them is scored against the story's first character**. So the model
is trained to answer "what does a story open with?" from the latents alone.

That is what unconditional sampling then does: `prime()` replays the latents
against a zero cache, with no input token, and hands back the first character's
distribution; generation continues from there with plain `step()`s. The
alternative — the `"\n\n"` that used to join the stories, fed in as a seed — is out
of distribution, because that sequence only ever occurred *between* two stories,
i.e. always on a state still carrying the previous one. The gated block's scalar
forget gate could learn to clear the state at such a boundary instead; the latents
make the boundary something the model is *given* rather than something it has to
infer, for 128 parameters.

One `generate()` call is therefore one story. A second story wants a second call
against a **reset** cache, which is the one place these examples genuinely reset
one.

Every position is scored against its next character (so the reported accuracy is
per character), and a story is walked in **windows** of `seq_len`: the loop takes
one optimizer step per window and carries the (detached) state into the next one
for as long as the **frontier gate** admits it, discarding the rest of the story
when a window's loss says the state is not worth passing on. The run's length is
the story's; `--run-len` only caps it, and `--run-len 1` trains each story's
opening window and no more. A closed gate is not a *wrong* regime — window 0 is
the story's own beginning — it just costs reach.

Stories differ in length, so a batch is padded to a whole number of windows of its
longest one; the batch carries how many positions of each slot are real, and the
padding is gathered away before the loss, never reaching it or the accuracy. The
mechanism is `burn_stack::examples::tiny_stories::lm` — see
`burn-mamba/examples/tiny-stories/README.md`'s "Runs and the frontier" for the
full account, including that peak memory does not grow with the run length.

## Usage

```bash
# debug check in flex (fp32)
cargo check --example tiny-stories

# train and then sample (downloads the 673MB parquet once, if it is not cached yet)
cargo run --release --example tiny-stories --features "backend-cuda" -- --training --inference

# a bigger corpus and a longer window
cargo run --release --example tiny-stories --features "backend-cuda" -- --training \
    -- --train-stories 32768 --seq-len 512
```

Downstream flags, all forwarded after the trailing `--` and persisted into the
artifacts' `training_config.json` (the number of epochs is the shared CLI's
`--epochs`, before the `--`):

| Flag | Default | Meaning |
|------|---------|---------|
| `--seq-len <n>` | 256 | characters per window (the BPTT length) |
| `--run-len <n>` | `usize::MAX` | cap on the windows one story may spend (`1` ⇒ openings only) |
| `--frontier-bits <f>` | 1.6 | the frontier gate's threshold, in bits per character |
| `--no-frontier` | off | carry the state through the whole story, ungated |
| `--train-stories <n>` | 4096 | stories pulled from the train split |
| `--valid-stories <n>` | 256 | stories pulled from the validation split |
| `--batch-size <n>` | 16 | windows per optimizer step |
| `--no-muon` | off | keep the hidden weight matrices on AdamW instead of [Muon](https://kellerjordan.github.io/posts/muon/) (see `mnist-class`'s README) |

- See `burn-deltanet/Cargo.toml` for other features or backend information.
- See `burn-deltanet/examples/README.md` for the CLI usage overview.

The schedule (batch 16, 16 epochs, cosine `24e-3 → 24e-4` after a 5%-of-an-epoch
warmup, Muon on the hidden matrices) was swept against this corpus, window and
parameter budget. Batch 16 is the largest that is *free*: at this model size the
GPU is launch-bound, so batches 8 and 16 run at the same steps per second and 16
simply sees twice the corpus for the same wall clock — 32 halves the rate. The
learning rate was tuned over the opening few hundred steps, where a cosine sized
for the whole run has barely moved, so read it as a peak rate rather than as a
tuned anneal.

Muon is the single most load-bearing line in that config: `--no-muon` costs more
than every other knob here put together.

## Sampling

`inference.rs` shows the library's three execution modes back to back: the class
latents are replayed by one `prime()` (no input token, and it already answers with
the first character's distribution), a prompt — when there is one — is consumed by
one chunkwise `forward()` (prefill, `DeltaPath::Chunk`), and every generated
character then costs one `step()` against that same cache — O(state) per token,
with no growing KV cache. Sampling is temperature-scaled multinomial over the full
48-way softmax (`temperature <= 0` is greedy), seeded by `ChaCha8Rng` so a run is
reproducible.

`DeltaVocabNet` is the block-generic `VocabNetwork` at `DeltaBlock`, so the
sampler is `burn_stack`'s own generic one; this example only supplies the path.

`--inference` writes one story per temperature (0.5 / 0.8 / 1.0), each primed and
unprompted, plus one continuation of a fixed prompt into
`<artifacts>/inference/`. Training samples a
short story at every small validation check into
`<artifacts>/sample-epoch-{e}-batch-{b}.txt`, so the text can be watched turning
from noise into words into sentences.

## Notes

- Loss is reported both in nats (Burn's cross-entropy) and as **bits per
  character**; the uniform baseline is `log2(48) = 5.58` bits.
- The whole 256-character window is processed in parallel by the chunkwise WY
  path (`chunk_len = 64`), and every layer's activations are kept, so vram
  scales with `batch_size · seq_len · n_layers`.
