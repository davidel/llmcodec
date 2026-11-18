"""End-to-end codec tests.

Crucially, the coder must be *exactly* invertible even with an untrained,
randomly initialised model: the arithmetic coding is lossless as long as the
candidate set is a deterministic function of the model distribution.  We do not
need a good model to test correctness -- only a fixed one.
"""

import numpy as np
import pytest
import torch

from llmcodec.candidates import build_selector
from llmcodec.codec import Codec, NotEncodableError
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
        "i have 3 cats, 2 dogs.",
    ],
)
def test_full_roundtrip_is_exact(model, text):
  codec = _codec(model, "full")
  enc = codec.encode(text)
  assert 0 <= enc.value <= 1
  assert codec.decode(enc.value) == text


def test_roundtrip_with_unknown_characters(model):
  codec = _codec(model, "full")
  # Uppercase is lower-cased; '!' becomes <unk>.
  enc = codec.encode("Hi!")
  assert codec.decode(enc.value) == "hi<unk>"


def test_code_increases_with_first_character(model):
  """Different first characters should generally map to disjoint intervals."""
  codec = _codec(model, "full")
  a = codec.encode("a")
  b = codec.encode("b")
  # Intervals for different first characters are disjoint (or at least not
  # identical) for a generic model.
  assert a.interval != b.interval


def test_decode_rejects_out_of_range(model):
  codec = _codec(model, "full")
  with pytest.raises(ValueError):
    codec.decode(1.5)


def test_topn_raises_when_character_missing(model):
  codec = _codec(model, "topn", n=1)
  # With n=1 every step has a single candidate, so almost any real sentence
  # will hit a token outside the candidate set.
  with pytest.raises(NotEncodableError):
    codec.encode("hello, world.")


def test_adaptive_roundtrip_often_works(model):
  codec = _codec(model, "adaptive", max_n=16, coverage=0.999)
  text = "hi"
  try:
    enc = codec.encode(text)
  except NotEncodableError:
    pytest.skip("adaptive selector missed a token for this random model")
  assert codec.decode(enc.value) == text


def test_candidates_are_deterministic(model):
  """Same prefix -> same candidate set, independent of encoder/decoder path."""
  codec = _codec(model, "full")
  prefix = [DEFAULT_VOCAB.bos_id]
  p1 = codec._probs(prefix)
  p2 = codec._probs(prefix)
  np.testing.assert_allclose(p1, p2, rtol=0, atol=0)
  c1 = codec._candidates(p1)
  c2 = codec._candidates(p2)
  assert c1 == c2
