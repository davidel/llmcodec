"""Encode and decode sentences with a pretrained Hugging Face model.

Usage:
    python examples/pretrained_hf.py [--hf-model gpt2] [--dtype float32]

Requires the ``hf`` extra (``pip install -e ".[hf]"``).  The first run
downloads the model checkpoint (about 550 MB for ``gpt2``); later runs load
from the local cache.

The demo encodes a few sentences with an adaptive candidate selector,
reports the exact interval width as a compression figure (bits per
character), and verifies that every sentence decodes back exactly.
"""

import argparse
import math

from llmcodec.candidates import build_selector
from llmcodec.codec import Codec
from llmcodec.pretrained import HFCausalLM, HFTokenizer

SENTENCES = [
    "The cat sat on the mat.",
    "Large language models compress text surprisingly well.",
    "Arithmetic coding turns a sentence into a single real number.",
]


def parse_args():
  """Parse the demo arguments."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--hf-model",
                      default="gpt2",
                      help="Hugging Face model id (default: gpt2)")
  parser.add_argument("--dtype",
                      default="float32",
                      help="inference dtype: float32, float16 or bfloat16")
  parser.add_argument("--max-n",
                      type=int,
                      default=None,
                      help="adaptive candidate budget (default: the model's whole "
                      "vocabulary, so every token stays encodable)")
  return parser.parse_args()


def main():
  """Load model and tokenizer, then roundtrip the demo sentences."""
  args = parse_args()
  print(f"[demo] loading {args.hf_model} (dtype={args.dtype})")
  model = HFCausalLM.load(args.hf_model, dtype=args.dtype, device="auto")
  vocab = HFTokenizer.load(args.hf_model)
  max_n = args.max_n if args.max_n is not None else vocab.vocab_size
  codec = Codec(model,
                vocab,
                build_selector("adaptive", max_n=max_n, coverage=1.0),
                device="cpu")

  print(f"selector: {codec.selector.describe()}")
  print(f"{'text':<44} {'bits':>7} {'bits/char':>10}  roundtrip")
  print("-" * 78)

  ok = True
  for text in SENTENCES:
    enc = codec.encode(text)
    decoded = codec.decode(enc.value)
    match = decoded == text
    ok &= match
    # The interval width is the probability mass the model would assign to
    # everything else it considers possible at the same prefix; -log2 of it
    # is the number of bits the code needs.
    bits = -math.log2(float(enc.interval.width))
    print(f"{text!r:<44} {bits:>7.1f} {bits / len(text):>10.3f}  {match}")

  print("\nall round-trips exact:", ok)
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
