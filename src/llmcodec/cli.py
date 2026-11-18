"""Command-line interface.

Exposes four subcommands:

``llmcodec train``
    Train the tiny language model.

``llmcodec encode TEXT``
    Encode a sentence to a real number in ``[0, 1]``.

``llmcodec decode VALUE``
    Decode a real number in ``[0, 1]`` back to a sentence.

``llmcodec roundtrip TEXT``
    Encode then decode and report whether the result matches the input.
"""

import argparse
import logging
from fractions import Fraction

from .candidates import build_selector
from .codec import Codec, NotEncodableError
from .model import TinyLM
from .train import TrainConfig
from .train import train as train_run
from .vocab import DEFAULT_VOCAB


def _make_codec(args):
  """Build a :class:`~llmcodec.codec.Codec` from parsed CLI arguments.

  Args:
    args: Parsed arguments carrying ``--model`` and the selector options.

  Returns:
    A ready-to-use codec.
  """
  device = getattr(args, "device", "auto")
  model = TinyLM.load(args.model, map_location=device)
  selector = build_selector(
      args.mode,
      n=getattr(args, "n", 5),
      max_n=getattr(args, "max_n", 16),
      coverage=getattr(args, "coverage", 0.999),
  )
  return Codec(model, DEFAULT_VOCAB, selector, device=device)


def _cmd_train(args):
  """Handle the ``train`` subcommand."""
  cfg = TrainConfig(
      source=args.source,
      max_stories=args.max_stories,
      steps=args.steps,
      batch_size=args.batch_size,
      block_size=args.block_size,
      lr=args.lr,
      warmup_steps=args.warmup_steps,
      log_every=args.log_every,
      eval_every=args.eval_every,
      out_dir=args.out_dir,
      device=args.device,
      amp=args.amp,
      amp_dtype=args.amp_dtype,
      num_threads=args.num_threads,
      seed=args.seed,
  )
  train_run(cfg)
  return 0


def _cmd_encode(args):
  """Handle the ``encode`` subcommand."""
  codec = _make_codec(args)
  try:
    enc = codec.encode(args.text)
  except NotEncodableError as exc:
    print(f"encode failed: {exc}")
    print("hint: use --mode full (lossless) or a larger --n / --coverage")
    return 1
  print(f"text      : {args.text!r}")
  print(f"value     : {float(enc.value):.12f}")
  print(f"fraction  : {enc.value}")
  print(f"interval  : {enc.interval}")
  print(f"steps     : {enc.n_steps}")
  print(f"selector  : {codec.selector.describe()}")
  return 0


def _cmd_decode(args):
  """Handle the ``decode`` subcommand."""
  codec = _make_codec(args)
  value = Fraction(args.value) if args.exact else Fraction(float(args.value))
  try:
    text = codec.decode(value)
  except (NotEncodableError, RuntimeError) as exc:
    print(f"decode failed: {exc}")
    return 1
  print(f"value     : {float(value):.12f}")
  print(f"text      : {text!r}")
  return 0


def _cmd_roundtrip(args):
  """Handle the ``roundtrip`` subcommand."""
  codec = _make_codec(args)
  try:
    enc = codec.encode(args.text)
    decoded = codec.decode(enc.value)
  except NotEncodableError as exc:
    print(f"roundtrip failed: {exc}")
    print("hint: use --mode full (lossless) or a larger --n / --coverage")
    return 1
  ok = decoded == args.text.lower()
  print(f"text      : {args.text!r}")
  print(f"value     : {float(enc.value):.12f}")
  print(f"decoded   : {decoded!r}")
  print(f"match     : {ok}")
  return 0 if ok else 1


def add_codec_args(parser):
  """Add the shared model/selector arguments to a codec subcommand.

  Args:
    parser: The subparser to extend.
  """
  parser.add_argument("--model",
                      required=True,
                      help="path to a trained model.pt checkpoint")
  parser.add_argument(
      "--mode",
      choices=["full", "topn", "adaptive"],
      default="full",
      help="candidate-selection strategy: 'full' is lossless, 'topn' and "
      "'adaptive' are lossy but use a smaller alphabet",
  )
  parser.add_argument("--n",
                      type=int,
                      default=5,
                      help="number of candidates kept in 'topn' mode")
  parser.add_argument("--max-n",
                      type=int,
                      default=16,
                      help="maximum number of candidates in 'adaptive' mode")
  parser.add_argument("--coverage",
                      type=float,
                      default=0.999,
                      help="target probability mass in 'adaptive' mode")
  parser.add_argument(
      "--device",
      default="auto",
      help="torch device for inference: 'auto' (CUDA if available), 'cpu' or 'cuda'",
  )


def build_parser():
  """Construct the top-level argument parser with all subcommands.

  Returns:
    The configured :class:`argparse.ArgumentParser`.
  """
  parser = argparse.ArgumentParser(
      prog="llmcodec",
      description="Encode sentences as real numbers in [0, 1] with a tiny LM.",
  )
  sub = parser.add_subparsers(dest="command", required=True)

  p_train = sub.add_parser("train", help="train the tiny language model")
  p_train.add_argument(
      "--source",
      choices=["tinystories", "sample"],
      default="tinystories",
      help="training corpus: 'tinystories' downloads it, 'sample' is offline",
  )
  p_train.add_argument("--max-stories",
                       type=int,
                       default=200_000,
                       help="cap on the number of stories to load")
  p_train.add_argument("--steps",
                       type=int,
                       default=50_000,
                       help="number of optimizer steps")
  p_train.add_argument("--batch-size",
                       type=int,
                       default=64,
                       help="blocks per optimizer step")
  p_train.add_argument("--block-size",
                       type=int,
                       default=256,
                       help="training context length in tokens")
  p_train.add_argument("--lr", type=float, default=3e-4, help="peak learning rate")
  p_train.add_argument("--warmup-steps",
                       type=int,
                       default=500,
                       help="linear warmup length")
  p_train.add_argument("--log-every",
                       type=int,
                       default=50,
                       help="print a loss line every N steps")
  p_train.add_argument("--eval-every",
                       type=int,
                       default=500,
                       help="run validation every N steps (0 disables)")
  p_train.add_argument("--out-dir",
                       default="runs/default",
                       help="directory for checkpoint and config")
  p_train.add_argument("--device",
                       default="auto",
                       help="torch device: 'auto' (CUDA if available), 'cpu' or 'cuda'")
  p_train.add_argument("--amp",
                       action=argparse.BooleanOptionalAction,
                       default=True,
                       help="enable mixed-precision autocast (default: enabled)")
  p_train.add_argument(
      "--amp-dtype",
      choices=["auto", "float16", "bfloat16", "float32"],
      default="auto",
      help="autocast dtype (default: float16 on CUDA, bfloat16 on CPU)")
  p_train.add_argument("--num-threads",
                       type=int,
                       default=0,
                       help="torch CPU threads (0 = default)")
  p_train.add_argument("--seed", type=int, default=1337, help="RNG seed")
  p_train.set_defaults(func=_cmd_train)

  p_enc = sub.add_parser("encode", help="encode a sentence to a real in [0, 1]")
  p_enc.add_argument("text", help="the sentence to encode")
  add_codec_args(p_enc)
  p_enc.set_defaults(func=_cmd_encode)

  p_dec = sub.add_parser("decode", help="decode a real in [0, 1] to a sentence")
  p_dec.add_argument("value", help="a real number in [0, 1]")
  p_dec.add_argument(
      "--exact",
      action="store_true",
      help="interpret VALUE as an exact fraction such as 3/7 instead of a float",
  )
  add_codec_args(p_dec)
  p_dec.set_defaults(func=_cmd_decode)

  p_rt = sub.add_parser("roundtrip",
                        help="encode then decode and report whether it matches")
  p_rt.add_argument("text", help="the sentence to round-trip")
  add_codec_args(p_rt)
  p_rt.set_defaults(func=_cmd_roundtrip)

  return parser


def main(argv=None):
  """Parse arguments and dispatch to the selected subcommand.

  Args:
    argv: Argument list (defaults to ``sys.argv[1:]``).

  Returns:
    The process exit code.
  """
  logging.basicConfig(level=logging.INFO, format="[llmcodec] %(message)s")
  parser = build_parser()
  args = parser.parse_args(argv)
  return args.func(args)


if __name__ == "__main__":  # pragma: no cover
  raise SystemExit(main())
