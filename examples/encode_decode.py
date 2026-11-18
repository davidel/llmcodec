"""Encode and decode sentences with a trained llmcodec model.

Usage:
    python examples/encode_decode.py [path/to/model.pt]

If no checkpoint is given it trains a tiny model on the built-in offline corpus
so the script is runnable end-to-end without network access.
"""

import os
import sys

from llmcodec import Codec, TinyLM, build_selector
from llmcodec.model import ModelConfig
from llmcodec.train import TrainConfig, train
from llmcodec.vocab import DEFAULT_VOCAB

SENTENCES = [
    "once upon a time",
    "the cat sat on the mat.",
    "hi",
    "a little girl had 3 red balls.",
]


def get_model(path):
  if path and os.path.exists(path):
    print(f"[demo] loading {path}")
    return TinyLM.load(path)
  print("[demo] no checkpoint given -- training a tiny model on the sample corpus")
  cfg = TrainConfig(
      source="sample",
      steps=300,
      batch_size=8,
      block_size=32,
      warmup_steps=20,
      out_dir="runs/demo",
      eval_every=0,
      log_every=100,
      model=ModelConfig(
          vocab_size=DEFAULT_VOCAB.vocab_size,
          d_model=64,
          n_layers=2,
          n_heads=4,
          d_ff=128,
      ),
  )
  return TinyLM.load(train(cfg))


def main():
  model_path = sys.argv[1] if len(sys.argv) > 1 else None
  model = get_model(model_path)
  codec = Codec(model, DEFAULT_VOCAB, build_selector("full"))

  print(f"\nselector: {codec.selector.describe()}")
  print(f"{'text':<34} {'value':>18} {'interval width':>18}  roundtrip")
  print("-" * 90)

  ok = True
  for text in SENTENCES:
    enc = codec.encode(text)
    decoded = codec.decode(enc.value)
    match = decoded == text.lower()
    ok &= match
    print(f"{text!r:<34} {enc.as_float:>18.12f} "
          f"{float(enc.interval.width):>18.3e}  {match}")

  print("\nall round-trips exact:", ok)
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
