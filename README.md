# llmcodec

Encode a sentence as **a single real number in `[0, 1]`** — and decode it back —
using a small language model as the engine of an arithmetic coder.

The model is a compact decoder-only transformer (default: **6 layers, 512 wide**)
trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).
Its next-token probability distribution drives a classic arithmetic-coding
recursion: each character of the input selects a sub-interval of the current
probability interval, and the final interval is the code.

> **Why a language model?** A good model assigns high probability to plausible
> continuations, so likely sentences consume *less* of the `[0, 1]` range and end
> up with wider (more robust, shorter-in-bits) intervals. The coder itself is
> lossless and works with *any* model — even a randomly initialised one.

---

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
from the same prefixes (no side information needed), find the slice of the
current interval containing `r`, emit that token, narrow the interval to the
slice, and repeat until `<eos>`.

The interval arithmetic is done with **exact rationals**, so encoding and
decoding are perfectly inverse — no floating-point drift, even for long inputs.

### Candidate selection is pluggable

The candidate set must be a deterministic function of the model distribution so
that encoder and decoder agree. Three strategies ship in `llmcodec.candidates`:

| mode       | behaviour                                                                 | lossless? |
|------------|---------------------------------------------------------------------------|-----------|
| `full`     | every token with non-zero probability (**default**)                       | yes       |
| `topn`     | the `N` most probable tokens, laid out alphabetically                     | no\*      |
| `adaptive` | keep adding tokens (up to `max_n`) until `coverage` of the mass is reached | no\*      |

\* With `topn`/`adaptive`, encoding raises `NotEncodableError` if the true next
token is not in the candidate set. `full` never fails.

---

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

`runs/default/train_config.json` records the exact configuration for
reproducibility.

## Encode / decode

```bash
llmcodec encode "once upon a time" --model runs/default/model.pt
# value     : 0.123456789012
# fraction  : 123456789012/...

llmcodec decode 0.123456789012 --model runs/default/model.pt
# text      : 'once upon a time'

llmcodec roundtrip "the cat sat" --model runs/default/model.pt
# match     : True
```

With the `topn` / `adaptive` modes add `--mode topn --n 5` or
`--mode adaptive --coverage 0.999`.

## Python API

```python
from llmcodec import Codec, TinyLM, build_selector
from llmcodec.vocab import DEFAULT_VOCAB

model = TinyLM.load("runs/default/model.pt")
codec = Codec(model, DEFAULT_VOCAB, build_selector("full"))

enc = codec.encode("the cat sat on the mat.")
print(enc.as_float)          # a real in [0, 1]
print(codec.decode(enc.value))   # 'the cat sat on the mat.'
```

See [`examples/encode_decode.py`](examples/encode_decode.py) for a runnable demo.

## Tests

```bash
pytest -m "not slow"   # fast unit + property tests (no training/network)
pytest -m slow         # includes a short end-to-end training run
```

The codec tests assert **exact** round-trips with an untrained model, which is
the strongest possible correctness check for the coder.

## Layout

```
src/llmcodec/
  vocab.py        # the 43-token vocabulary
  interval.py     # exact rational interval arithmetic
  candidates.py   # pluggable candidate-set selection (full/topn/adaptive)
  model.py        # TinyLM transformer
  data.py         # TinyStories loading & batching
  train.py        # training loop
  codec.py        # the arithmetic encoder/decoder
  cli.py          # llmcodec train|encode|decode|roundtrip
```

## License

MIT — see [LICENSE](LICENSE).
