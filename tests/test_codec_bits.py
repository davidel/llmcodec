"""End-to-end tests for the dyadic (bit-code) encoding API.

``Codec.encode_bits`` / ``Codec.decode_bits`` are the binary counterpart of the
real-number API.  As with the plain codec, correctness does not depend on a
trained model: the arithmetic coding is lossless for any fixed model, so an
untrained one is enough to test the round-trip.
"""

from fractions import Fraction

import pytest
import torch

from llmcodec.candidates import build_selector
from llmcodec.codec import Codec
from llmcodec.interval import BitCode
from llmcodec.model import ModelConfig, TinyLM
from llmcodec.vocab import DEFAULT_VOCAB


@pytest.fixture(scope="module")
def model():
  torch.manual_seed(0)
  cfg = ModelConfig(vocab_size=DEFAULT_VOCAB.vocab_size,
                    d_model=32,
                    n_layers=2,
                    n_heads=2,
                    d_ff=64)
  m = TinyLM(cfg)
  m.eval()
  return m


def _codec(model, mode="full", **kw):
  return Codec(model, DEFAULT_VOCAB, build_selector(mode, **kw))


@pytest.mark.parametrize(
    "text",
    [
        "hi",
        "hello, world.",
        "once upon a time",
        "abc 123",
        "a",
        "the quick brown fox jumps over the lazy dog",
    ],
)
def test_encode_bits_roundtrip_is_exact(model, text):
  """The bit code decodes to exactly the text the real code decodes to."""
  codec = _codec(model, "full")
  code = codec.encode_bits(text)
  assert isinstance(code, BitCode)
  assert codec.decode_bits(code) == text


def test_encode_bits_matches_interval_dyadic(model):
  """``encode_bits`` is exactly ``encode(...).interval.dyadic()``."""
  codec = _codec(model, "full")
  enc = codec.encode("hello world")
  assert codec.encode_bits("hello world") == enc.interval.dyadic()


def test_encode_bits_cell_is_inside_interval(model):
  """The code's cell lies inside the final interval the encoder produced."""
  codec = _codec(model, "full")
  enc = codec.encode("hello world")
  cell = codec.encode_bits("hello world").cell
  assert enc.interval.lower <= cell.lower
  assert cell.upper <= enc.interval.upper


def test_encode_bits_value_decodes_with_plain_decode(model):
  """The code's value is a point the plain decoder also accepts."""
  codec = _codec(model, "full")
  code = codec.encode_bits("the cat")
  assert codec.decode(code.value) == "the cat"


def test_decode_bits_rejects_non_bitcode(model):
  """``decode_bits`` refuses values that are not a ``BitCode``."""
  codec = _codec(model, "full")
  with pytest.raises(ValueError):
    codec.decode_bits(Fraction(1, 2))
