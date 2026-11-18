# llmcodec

Encode a sentence as **a single real number in `[0, 1]`** — and decode it back —
using a small language model as the engine of an arithmetic coder.

The model is a compact decoder-only transformer (default: **6 layers, 512 wide**)
trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).
Its next-token probability distribution drives a classic arithmetic-coding
recursion: each character of the input selects a sub-interval of the current
probability interval, and the final interval *is* the code.

```text
"the cat sat"  ──encode──▶  0.700752097497  ──decode──▶  "the cat sat"
```

> **Why a language model?** A good model assigns high probability to plausible
> continuations, so likely sentences consume *less* of the `[0, 1]` range and end
> up with wider intervals (fewer bits of information). The coder itself is
> **lossless** and works with *any* model — even a randomly initialised one.

---

## Table of contents

- [Background: arithmetic coding with a language model](#background-arithmetic-coding-with-a-language-model)
- [The scheme](#the-scheme)
- [A worked example](#a-worked-example)
- [Algorithm](#algorithm)
- [Why exact rational arithmetic](#why-exact-rational-arithmetic)
- [Candidate selection is pluggable](#candidate-selection-is-pluggable)
- [Vocabulary](#vocabulary)
- [Model architecture](#model-architecture)
- [Device and mixed precision](#device-and-mixed-precision)
- [Install](#install)
- [Train](#train)
- [Encode / decode](#encode--decode)
- [Python API](#python-api)
- [CLI reference](#cli-reference)
- [Configuration reference](#configuration-reference)
- [Testing](#testing)
- [Project layout](#project-layout)
- [Limitations and FAQ](#limitations-and-faq)
- [License](#license)

---

## Background: arithmetic coding with a language model

Arithmetic coding is a lossless compression technique. Given a *source model*
that predicts the probability of the next symbol, it maps a whole message onto a
single number: you maintain an interval `[lo, hi]` that starts as `[0, 1]`, and
for every symbol you replace the interval by the sub-interval the model assigns
to that symbol. The more probable a symbol, the wider its sub-interval, and the
less the interval shrinks. A more confident, better-calibrated model therefore
produces a wider final interval — i.e. a "shorter" message in information terms.

`llmcodec` uses a **transformer language model as the source model**. The twist
is that the "message" is an ordinary sentence and the "symbols" are its
characters (plus `BOS`/`EOS`), and the output we keep is not a bitstring but the
**midpoint of the final interval**, a real number in `[0, 1]`. Decoding is the
exact inverse recursion.

## The scheme

**Vocabulary (43 tokens).** `<pad>`, `<bos>`, `<eos>`, `<unk>` plus the 39
message characters `a–z`, `0–9`, space, `.`, `,`. Text is lower-cased; every
other character maps to `<unk>`.

**Encoding.** Start with the interval `[0, 1]` and the prompt `<bos>`:

1. Feed the current prefix to the model; take the next-token distribution
   (softmax over the logits).
2. Ask the **candidate selector** for an ordered candidate alphabet derived
   *only* from that distribution.
3. Lay the candidates out in **token-id (alphabetical) order**, splitting the
   current interval into slices whose widths are proportional to the
   (renormalized) candidate probabilities.
4. The true next character picks a slice → that slice becomes the new interval.
5. Append the true character to the prefix and repeat through the whole input,
   then finally consume `<eos>`.

The emitted code is the **midpoint** of the final interval. Every value inside
that interval decodes to the same sentence.

**Decoding.** Given a real `r` in `[0, 1]`, replay the exact same candidate sets
from the same prefixes (no side information is needed), find the slice of the
current interval containing `r`, emit that token, narrow the interval to the
slice, and repeat until `<eos>`.

## A worked example

Suppose at the first step the model's top candidates are `a = 0.1`, `m = 0.3`,
`t = 0.4` (all other probability mass is ignored, e.g. in `topn` mode). We order
them alphabetically — `a, m, t` — and renormalize so they sum to one:

```text
sum        = 0.1 + 0.3 + 0.4 = 0.8
a -> 0.1/0.8 = 0.125
m -> 0.3/0.8 = 0.375
t -> 0.4/0.8 = 0.500
```

They slice the current interval (here `[0, 1]`) in that order:

```text
[0.000, 0.125)  -> 'a'
[0.125, 0.500)  -> 'm'
[0.500, 1.000)  -> 't'
```

If the first character of the input is `m`, the new interval is
`[0.125, 0.500)`. We then feed `BOS + 'm'` to the model, renormalize the new top
candidates, and slice `[0.125, 0.500)` the same way. The second character picks
one of those sub-slices, and so on. After the last character we consume `<eos>`
one final time; the midpoint of the resulting interval is the code.

Decoding reverses this: given a code, find which slice of `[0, 1]` contains it →
first character; narrow to that slice; feed `BOS + first_char`; repeat.

## Algorithm

```text
encode(text):
    ids      = tokenize(lower(text))          # chars + final EOS
    prefix   = [BOS]
    interval = [0, 1]
    for tok in ids + [EOS]:
        probs     = softmax(model(prefix))[last]
        cand      = selector(probs)           # ordered ids + integer weights
        i         = cand.index_of(tok)        # or NotEncodableError
        interval  = interval.subdivide(cand.weights, i)
        prefix.append(tok)
    return midpoint(interval)

decode(code):
    prefix   = [BOS]
    interval = [0, 1]
    out      = []
    loop:
        probs    = softmax(model(prefix))[last]
        cand     = selector(probs)
        i        = interval.find_slice(cand.weights, code)
        tok      = cand.ids[i]
        if tok == EOS: return text(out)
        interval = interval.subdivide(cand.weights, i)
        out.append(tok); prefix.append(tok)
```

The encoder and decoder build the **same** `cand` at every step because `cand`
depends only on `probs`, which depends only on `prefix`. That is the entire
correctness argument for the coder.

## Why exact rational arithmetic

If the interval were tracked with floats, rounding errors would accumulate over
the dozens of steps of a sentence and the interval recovered during decoding
could drift off the one produced during encoding, breaking the round-trip.

Instead the codec stores each interval as an exact rational
`[num_lo / den, num_hi / den]` (`llmcodec.interval.Interval`) and subdivides with
integer arithmetic, reducing by the gcd of the three integers at every step to
keep the numbers small. The cost is that the denominator grows with the sum of
the candidate weights each step, which is why the weights are quantized to a
fixed scale (`WEIGHT_SCALE = 2**20`) rather than using the raw float
probabilities.

## Candidate selection is pluggable

The candidate set must be a deterministic function of the model distribution so
that encoder and decoder agree. Three strategies ship in `llmcodec.candidates`:

| mode       | behaviour                                                                    | lossless? |
|------------|------------------------------------------------------------------------------|-----------|
| `full`     | every token with non-zero probability (**default**)                          | yes       |
| `topn`     | the `N` most probable tokens, laid out alphabetically                        | no\*      |
| `adaptive` | keep adding tokens (up to `max_n`) until `coverage` of the mass is reached    | no\*      |

\* With `topn`/`adaptive`, encoding raises `NotEncodableError` if the true next
token is not in the candidate set. `full` never fails, so it is the default and
the one used by the tests to prove exact invertibility.

## Vocabulary

| id range | tokens                                                     |
|----------|------------------------------------------------------------|
| 0–3      | `<pad>`, `<bos>`, `<eos>`, `<unk>`                         |
| 4–42     | `a`–`z`, `0`–`9`, ` `, `.`, `,`                            |

`<pad>` pads batched training sequences; `<bos>`/`<eos>` delimit sentences;
`<unk>` absorbs any character outside the message alphabet. The id assignment is
fixed so checkpoints are portable.

## Model architecture

| field          | default |
|----------------|---------|
| `vocab_size`   | 43      |
| `d_model`      | 512     |
| `n_layers`     | 6       |
| `n_heads`      | 8       |
| `d_ff`         | 2048    |
| `max_seq_len`  | 256     |
| `dropout`      | 0.0     |

A pre-norm causal transformer (`TinyLM`). `next_token_probs(prefix)` returns the
softmax distribution over the vocabulary for the next token — the codec's only
model dependency.

## Device and mixed precision

Both training and inference **use a GPU automatically when one is present** and
fall back to the CPU otherwise. The `device` argument defaults to `"auto"`:

| value     | meaning                                              |
|-----------|------------------------------------------------------|
| `"auto"`  | CUDA if available, else CPU (**default**)            |
| `"cpu"`   | force CPU                                            |
| `"cuda"`  | force CUDA (errors if unavailable), e.g. `"cuda:1"`  |

**Training** additionally supports mixed precision via `amp` / `amp_dtype`:

- `--amp` (default on) wraps the forward pass in `torch.autocast` and uses a
  `GradScaler` (only actually engaged for float16 on CUDA).
- `--amp-dtype auto` picks **float16 on CUDA** and **bfloat16 on CPU**; you can
  force `float16`, `bfloat16` or `float32`.

The loss and all reductions are always computed in float32, and the codec's
`next_token_probs` computes the softmax in float32 even under autocast, so the
probabilities the coder consumes stay numerically faithful.

```bash
llmcodec train --device auto --amp --amp-dtype auto ...   # GPU/AMP when available
llmcodec train --device cpu --no-amp ...                  # force CPU, no autocast
llmcodec encode "the cat" --model runs/default/model.pt --device auto
```

## Install

```bash
# CPU PyTorch (recommended in CI / sandboxes)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# the package (+ data & dev extras)
pip install -e ".[all]"
```

## Train

```bash
# full run: 6 layers, 512 wide, TinyStories
llmcodec train --source tinystories --max-stories 200000 --steps 50000 \
    --batch-size 64 --block-size 256 --out-dir runs/default

# quick offline smoke run (no network, tiny model)
llmcodec train --source sample --steps 200 --out-dir runs/sample
```

`runs/default/train_config.json` records the exact configuration (including the
vocabulary) for reproducibility.

> **CPU budget.** A 6×512 transformer trains at roughly 6–8k tokens/s on a
> typical CPU, so the full 50k-step run above is a multi-hour job. Use
> `--max-stories` / `--steps` (and optionally `--num-threads`) to fit your
> hardware. Even a lightly trained model round-trips exactly with the `full`
> selector.

## Encode / decode

```bash
llmcodec encode "once upon a time" --model runs/default/model.pt
# text      : 'once upon a time'
# value     : 0.510150913490
# fraction  : 711416.../1394521...
# interval  : Interval([0.510151, 0.510151] = ...)
# steps     : 17
# selector  : {'strategy': 'full'}

llmcodec decode 0.510150913490 --model runs/default/model.pt
# text      : 'once upon a time'

llmcodec roundtrip "the cat sat" --model runs/default/model.pt
# match     : True
```

Use `--mode topn --n 5` or `--mode adaptive --coverage 0.999` to try the lossy
selectors. If a character is not in the candidate set, `encode` prints a hint and
exits non-zero rather than raising.

## Python API

```python
from llmcodec import Codec, TinyLM, build_selector
from llmcodec.vocab import DEFAULT_VOCAB

model = TinyLM.load("runs/default/model.pt")
codec = Codec(model, DEFAULT_VOCAB, build_selector("full"))

enc = codec.encode("the cat sat on the mat.")
print(enc.as_float)             # a real in [0, 1]
print(enc.value)                # the exact Fraction
print(float(enc.interval.width))  # how "confident" the code is
print(codec.decode(enc.value))  # 'the cat sat on the mat.'
```

See [`examples/encode_decode.py`](examples/encode_decode.py) for a runnable demo
(which trains a tiny model on the offline corpus if you don't pass a checkpoint).

## CLI reference

| command | arguments |
|---------|-----------|
| `train` | `--source`, `--max-stories`, `--steps`, `--batch-size`, `--block-size`, `--lr`, `--warmup-steps`, `--log-every`, `--eval-every`, `--out-dir`, `--device`, `--num-threads`, `--seed` |
| `encode` | `TEXT`, `--model`, `--mode`, `--n`, `--max-n`, `--coverage` |
| `decode` | `VALUE`, `--exact`, `--model`, `--mode`, `--n`, `--max-n`, `--coverage` |
| `roundtrip` | `TEXT`, `--model`, `--mode`, `--n`, `--max-n`, `--coverage` |

Run `llmcodec <command> --help` for the full help text.

## Configuration reference

### `TrainConfig`

| field | default | meaning |
|-------|---------|---------|
| `source` | `"tinystories"` | `"tinystories"` or `"sample"` |
| `split` | `"train"` | dataset split |
| `max_stories` | `200000` | story cap (`None` = all) |
| `block_size` | `256` | training context length |
| `batch_size` | `64` | blocks per step |
| `steps` | `50000` | optimizer steps |
| `lr` / `min_lr` | `3e-4` / `3e-5` | peak / final LR |
| `warmup_steps` | `500` | linear warmup |
| `weight_decay` | `0.1` | AdamW weight decay |
| `grad_clip` | `1.0` | max gradient norm |
| `seed` | `1337` | RNG seed |
| `device` | `"auto"` | torch device (`auto` = CUDA if available) |
| `amp` | `True` | enable mixed-precision autocast |
| `amp_dtype` | `"auto"` | autocast dtype (`auto` = float16 on CUDA, bfloat16 on CPU) |
| `eval_every` | `500` | validation cadence (0 disables) |
| `out_dir` | `"runs/default"` | checkpoint directory |
| `num_threads` | `0` | torch CPU threads (0 = default) |

### `ModelConfig`

See the [architecture table](#model-architecture).

## Testing

```bash
pytest -m "not slow"   # fast unit + property tests (no training/network)
pytest -m slow         # includes a short end-to-end training run
```

The codec tests assert **exact** round-trips with an *untrained* model, which is
the strongest possible correctness check for the coder: losslessness must not
depend on model quality. Other tests cover the interval arithmetic, the selector
strategies, the vocabulary, the transformer (shapes, causality, save/load) and
the data pipeline.

## Project layout

```text
src/llmcodec/
  vocab.py        # the 43-token vocabulary
  interval.py     # exact rational interval arithmetic
  candidates.py   # pluggable candidate-set selection (full/topn/adaptive)
  model.py        # TinyLM transformer
  device.py       # GPU/CPU selection and mixed-precision helpers
  data.py         # TinyStories loading & batching
  train.py        # training loop (GPU + AMP aware)
  codec.py        # the arithmetic encoder/decoder
  cli.py          # llmcodec train|encode|decode|roundtrip
tests/            # pytest suite
examples/         # runnable demo
configs/          # example training config
```

## Limitations and FAQ

**What happens if a character isn't in the candidate set?**
With `full` this never happens. With `topn`/`adaptive` it can, and `encode`
raises `NotEncodableError` (the CLI prints a hint and exits non-zero). Increase
`--n`/`--coverage` or use `--mode full`.

**Why does the code get long (small interval) for long sentences?**
Each character roughly multiplies the interval width by that character's
probability under the model. A 30-character sentence with ~1/20 probability per
step shrinks the interval by ~20³⁰ — the `Fraction` handles this exactly, which
is why the code can be a huge fraction rather than a float.

**Does the model need to be good for this to work?**
No — only for the code to be *compact*. Round-tripping is exact for any model.
A better model simply produces wider intervals (better compression).

**Can I change the vocabulary?**
Yes: `Vocab.build(chars=..., specials=...)`. Keep the special-token ids stable
if you want to reuse checkpoints, and remember the codec derives `emittable_ids`
from single-character tokens plus `<eos>`/`<unk>`.

**Why emit the midpoint rather than the lower bound?**
The midpoint stays as far as possible from both interval boundaries, so tiny
perturbations (e.g. rounding the code to a float) are least likely to push it
into a neighbouring interval.

## License

MIT — see [LICENSE](LICENSE).
