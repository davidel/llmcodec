"""Command-line interface.

Exposes six subcommands:

``llmcodec train``
    Train the tiny language model.

``llmcodec encode TEXT``
    Encode a sentence to a real number in ``[0, 1]``.

``llmcodec decode VALUE``
    Decode a real number in ``[0, 1]`` back to a sentence.

``llmcodec roundtrip TEXT``
    Encode then decode and report whether the result matches the input.

``llmcodec encode-bits TEXT``
    Encode a sentence to a dyadic :class:`~llmcodec.interval.BitCode`.

``llmcodec decode-bits BITS``
    Decode a :class:`~llmcodec.interval.BitCode` back to a sentence.
"""

import argparse
import hashlib
import json
import logging
import os
from fractions import Fraction

from .candidates import build_selector
from .codec import Codec, NotEncodableError
from .interval import BitCode
from .model import TinyLM
from .pretrained import HFTokenizer
from .train import add_train_arguments, config_from_args
from .train import train as train_run
from .vocab import DEFAULT_VOCAB

DEFAULT_SELECTOR_BY_VOCAB = 1000


def _load_tokenizer(args):
  """Resolve the tokenizer for a codec or training run.

  Priority: an explicit ``--tokenizer`` beats the tokenizer recorded in the
  checkpoint's ``train_config.json``, which beats the built-in character
  vocabulary.  A recorded tokenizer is re-fetched from its identity and
  verified against the content hash stored at training time, so a silently
  drifted vocabulary fails loudly instead of corrupting decodes.

  Args:
    args: Parsed arguments; ``tokenizer`` may be absent (train subparser).

  Returns:
    A :class:`~llmcodec.vocab.Vocab` or an
    :class:`~llmcodec.pretrained.HFTokenizer`.

  Raises:
    ValueError: If a recorded tokenizer's content hash does not match the
      tokenizer that was re-fetched under the recorded identity.
  """
  name = getattr(args, "tokenizer", None)
  if name:
    return HFTokenizer.load(name,
                            revision=getattr(args, "revision", None),
                            device=getattr(args, "device", "auto"))
  recorded = _recorded_tokenizer(args)
  if recorded is not None:
    name, revision, digest = recorded
    tokenizer = HFTokenizer.load(name,
                                 revision=revision,
                                 device=getattr(args, "device", "auto"))
    if digest is not None:
      actual = hashlib.sha256("\x1f".join(tokenizer.itos).encode()).hexdigest()
      if actual != digest:
        raise ValueError(f"tokenizer {name!r} does not match the one recorded at "
                         "training time (content hash mismatch); pass --tokenizer "
                         "explicitly to override")
    return tokenizer
  return DEFAULT_VOCAB


def _recorded_tokenizer(args):
  """Read the tokenizer identity recorded beside a trained checkpoint.

  Args:
    args: Parsed arguments carrying ``--model`` (may be absent).

  Returns:
    A ``(name, revision, sha256)`` tuple, or ``None`` when the checkpoint
    has no recorded subword tokenizer.
  """
  model_path = getattr(args, "model", None)
  if not model_path:
    return None
  config_path = os.path.join(os.path.dirname(os.path.abspath(model_path)),
                             "train_config.json")
  try:
    with open(config_path) as fh:
      record = json.load(fh)
  except (OSError, json.JSONDecodeError):
    return None
  block = record.get("tokenizer")
  if not isinstance(block, dict) or "name" not in block:
    return None
  return block["name"], block.get("revision"), block.get("sha256")


def _load_codec_model(args):
  """Load the model backend: an HF causal LM or a TinyLM checkpoint.

  Args:
    args: Parsed arguments carrying ``--model`` and the HF options.

  Returns:
    A :class:`~llmcodec.model.TinyLM` (nn.Module) or an
    :class:`~llmcodec.pretrained.HFCausalLM`.

  Raises:
    ValueError: If both or neither of ``--model`` and ``--hf-model`` are
      given.
  """
  hf_name = getattr(args, "hf_model", None)
  if hf_name and getattr(args, "model", None):
    raise ValueError("--model and --hf-model are mutually exclusive")
  if hf_name:
    from .pretrained import HFCausalLM

    return HFCausalLM.load(hf_name,
                           revision=getattr(args, "revision", None),
                           dtype=getattr(args, "hf_dtype", "float32"),
                           device=getattr(args, "device", "auto"))
  if not getattr(args, "model", None):
    raise ValueError("one of --model or --hf-model is required")
  return TinyLM.load(args.model, map_location=getattr(args, "device", "auto"))


def _make_codec(args):
  """Build a :class:`~llmcodec.codec.Codec` from parsed CLI arguments.

  Args:
    args: Parsed arguments carrying ``--model``, the tokenizer choice and
      the selector options.

  Returns:
    A ready-to-use codec.

  Raises:
    ValueError: If the requested selector mode cannot serve the tokenizer,
      for example the lossless ``full`` mode with a large vocabulary.
  """
  device = getattr(args, "device", "auto")
  model = _load_codec_model(args)
  vocab = _load_tokenizer(args)
  if args.mode == "full" and vocab.vocab_size >= DEFAULT_SELECTOR_BY_VOCAB:
    raise ValueError(
        f"the lossless 'full' selector needs one interval per token; with a "
        f"vocabulary of {vocab.vocab_size} that is impractically slow -- "
        "use '--mode adaptive' or '--mode topn' instead")
  selector = build_selector(
      args.mode,
      n=getattr(args, "n", 5),
      max_n=getattr(args, "max_n", 16),
      coverage=getattr(args, "coverage", 0.999),
  )
  return Codec(model, vocab, selector, device=device)


def _cmd_train(args):
  """Handle the ``train`` subcommand."""
  try:
    cfg = config_from_args(args)
    vocab = _load_tokenizer(args)
  except ValueError as exc:
    print(f"train failed: {exc}")
    return 1
  train_run(cfg, vocab=vocab)
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
  # The built-in vocabulary lower-cases; subword tokenizers do not, so the
  # expected text depends on which tokenizer produced the code.
  tokenizer_name = getattr(args, "tokenizer", None)
  if not tokenizer_name:
    recorded = _recorded_tokenizer(args)
    tokenizer_name = recorded[0] if recorded else None
  expected = args.text if tokenizer_name else args.text.lower()
  ok = decoded == expected
  print(f"text      : {args.text!r}")
  print(f"value     : {float(enc.value):.12f}")
  print(f"decoded   : {decoded!r}")
  print(f"match     : {ok}")
  return 0 if ok else 1


def _parse_bits(text):
  """Parse a ``0``/``1`` string into a :class:`~llmcodec.interval.BitCode`.

  The length of the string is the grid depth ``n`` and its binary value is the
  cell index ``k``; leading zeros are therefore significant.

  Args:
    text: A non-empty string of ``0`` and ``1`` characters.

  Returns:
    The corresponding :class:`~llmcodec.interval.BitCode`.

  Raises:
    ValueError: If ``text`` is empty or contains a character other than
      ``0``/``1``.
  """
  if not text or set(text) - {"0", "1"}:
    raise ValueError("bits must be a non-empty string of 0s and 1s")
  return BitCode(len(text), int(text, 2))


def _cmd_encode_bits(args):
  """Handle the ``encode-bits`` subcommand."""
  codec = _make_codec(args)
  try:
    code = codec.encode_bits(args.text)
  except NotEncodableError as exc:
    print(f"encode-bits failed: {exc}")
    print("hint: use --mode full (lossless) or a larger --n / --coverage")
    return 1
  print(f"text      : {args.text!r}")
  print(f"bits      : {code.bits}")
  print(f"n         : {code.n}")
  print(f"k         : {code.k}")
  print(f"value     : {float(code.value):.12f}")
  print(f"fraction  : {code.value}")
  print(f"cell      : {code.cell}")
  print(f"selector  : {codec.selector.describe()}")
  return 0


def _cmd_decode_bits(args):
  """Handle the ``decode-bits`` subcommand."""
  codec = _make_codec(args)
  try:
    code = _parse_bits(args.bits)
    text = codec.decode_bits(code)
  except (ValueError, RuntimeError) as exc:
    # ValueError covers both an unparsable bit string and NotEncodableError.
    print(f"decode-bits failed: {exc}")
    return 1
  print(f"bits      : {code.bits}")
  print(f"value     : {float(code.value):.12f}")
  print(f"text      : {text!r}")
  return 0


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
  parser.add_argument(
      "--tokenizer",
      default=None,
      help="Hugging Face tokenizer to use instead of the built-in character "
      "vocabulary, e.g. 'gpt2' (with a matching HF model, or a TinyLM "
      "trained with the same tokenizer; on 'train' it selects the "
      "tokenization of the training corpus and is recorded in "
      "train_config.json so decode can re-fetch it)",
  )
  parser.add_argument(
      "--hf-model",
      default=None,
      help="Hugging Face causal LM to use as the distribution source instead "
      "of a trained checkpoint, e.g. 'gpt2' (mutually exclusive with "
      "--model; requires the llmcodec[hf] extra)",
  )
  parser.add_argument(
      "--revision",
      default=None,
      help="revision to pin when loading the HF model and tokenizer "
      "(branch, tag or commit hash)",
  )
  parser.add_argument(
      "--hf-dtype",
      choices=["float32", "float16", "bfloat16", "auto"],
      default="float32",
      help="inference dtype for the HF model (default: float32 for "
      "reproducibility; encode and decode must agree)",
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
  add_train_arguments(p_train)
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

  p_encb = sub.add_parser("encode-bits", help="encode a sentence to a dyadic BitCode")
  p_encb.add_argument("text", help="the sentence to encode")
  add_codec_args(p_encb)
  p_encb.set_defaults(func=_cmd_encode_bits)

  p_decb = sub.add_parser("decode-bits",
                          help="decode a dyadic BitCode back to a sentence")
  p_decb.add_argument("bits", help="a 0/1 string; its length is the grid depth N")
  add_codec_args(p_decb)
  p_decb.set_defaults(func=_cmd_decode_bits)

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
