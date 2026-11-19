"""Tests for the codec's end-of-sequence guarantee.

A selector can legitimately leave ``<eos>`` out of the candidate set: a
strong model assigns the sentence-final ``<eos>`` so little probability
that it falls below a top-n or coverage cutoff, and the codec then cannot
terminate.  ``Codec._candidates`` therefore appends EOS with its true
quantized probability whenever a selector dropped it.

The tests use a fixed stub model (exact probability vector, no torch
randomness) and a stub selector that deliberately omits EOS, so every
weight asserted below is exact and the suite stays network-free.
"""

import pytest
import torch

from llmcodec.candidates import WEIGHT_SCALE, build_selector
from llmcodec.codec import Codec
from llmcodec.model import ModelConfig, TinyLM

# pad, bos, eos, unk, a, b, c
TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "a", "b", "c"]
BOS, EOS, UNK = 1, 2, 3
# Exact distribution: p(EOS) = 0.004 -> quantized weight rint(0.004 * 2**20).
PROBS = [0.05, 0.05, 0.004, 0.0, 0.30, 0.29, 0.306]
EOS_WEIGHT = 4194


class _FixedModel:
  """A model stub returning one fixed next-token distribution."""

  vocab_size = len(TOKENS)
  device = "cpu"

  def next_token_probs(self, input_ids, amp_dtype=None):
    del input_ids, amp_dtype
    return torch.tensor(PROBS, dtype=torch.float32)


class _StubVocab:
  """A minimal vocabulary over ``TOKENS`` with the usual special ids."""

  def encode(self, text, add_bos=False, add_eos=False):
    del add_bos, add_eos
    return [TOKENS.index(ch) for ch in text]

  def decode(self, ids, skip_special=False):
    del skip_special
    return "".join(TOKENS[i] for i in ids)

  @property
  def itos(self):
    return list(TOKENS)

  @property
  def vocab_size(self):
    return len(TOKENS)

  @property
  def bos_id(self):
    return BOS

  @property
  def eos_id(self):
    return EOS

  @property
  def unk_id(self):
    return UNK

  def token_id(self, token):
    return TOKENS.index(token)


class _TopKWithoutEos:
  """A selector that never places EOS, however probable it is."""

  def __init__(self, k=3):
    self.k = k

  def select(self, probs, eligible):
    order = sorted((i for i in eligible if i != EOS), key=lambda i: -probs[i])
    chosen = order[:self.k]
    weights = [max(1, int(round(probs[i] * WEIGHT_SCALE))) for i in chosen]
    from llmcodec.candidates import CandidateSet

    return CandidateSet(chosen, weights)

  def describe(self):
    return {"type": "topk-without-eos", "k": self.k}


@pytest.fixture
def codec():
  return Codec(_FixedModel(),
               _StubVocab(),
               _TopKWithoutEos(),
               emittable_ids=range(len(TOKENS)),
               device="cpu")


def test_eos_is_appended_with_its_true_weight(codec):
  """EOS lands in the set with exactly ``rint(p(EOS) * WEIGHT_SCALE)``."""
  import numpy as np

  candidates = codec._candidates(np.array(PROBS, dtype=np.float32))
  assert candidates.index_of(EOS) is not None
  assert candidates.token_ids[-1] == EOS  # appended at a fixed position
  assert candidates.weights[-1] == EOS_WEIGHT


def test_roundtrip_when_selector_omits_eos(codec):
  """Encoding terminates and decodes exactly despite the hostile selector."""
  enc = codec.encode("abcabc")
  assert codec.decode(enc.value) == "abcabc"
  assert codec.decode_bits(codec.encode_bits("abcabc")) == "abcabc"


def test_guarantee_is_a_noop_when_eos_is_covered(codec):
  """A selector that already includes EOS sees its set returned untouched."""
  wrapped_calls = []

  class _Spy:

    def __init__(self):
      self.inner = build_selector("full")

    def select(self, probs, eligible):
      result = self.inner.select(probs, eligible)
      wrapped_calls.append(result)
      return result

    def describe(self):
      return self.inner.describe()

  import numpy as np

  spy_codec = Codec(_FixedModel(),
                    _StubVocab(),
                    _Spy(),
                    emittable_ids=range(len(TOKENS)),
                    device="cpu")
  result = spy_codec._candidates(np.array(PROBS, dtype=np.float32))
  assert result is wrapped_calls[0]  # the very object the selector built


def test_zero_probability_eos_still_terminates():
  """Even with p(EOS) = 0 the codec appends a minimal weight and terminates."""

  class _ZeroEosModel(_FixedModel):

    def next_token_probs(self, input_ids, amp_dtype=None):
      del input_ids, amp_dtype
      probs = list(PROBS)
      probs[EOS] = 0.0
      return torch.tensor(probs, dtype=torch.float32)

  codec = Codec(_ZeroEosModel(),
                _StubVocab(),
                _TopKWithoutEos(),
                emittable_ids=range(len(TOKENS)),
                device="cpu")
  enc = codec.encode("abcabc")
  assert codec.decode(enc.value) == "abcabc"


def test_real_tiny_model_roundtrip_when_selector_omits_eos():
  """The guarantee holds with a real model forward pass in the loop."""

  class _AllButEos:
    """Covers every content token but never places EOS."""

    def select(self, probs, eligible):
      chosen = [i for i in eligible if i != EOS]
      weights = [max(1, int(round(probs[i] * WEIGHT_SCALE))) for i in chosen]
      from llmcodec.candidates import CandidateSet

      return CandidateSet(chosen, weights)

    def describe(self):
      return {"type": "all-but-eos"}

  torch.manual_seed(0)
  model = TinyLM(
      ModelConfig(vocab_size=len(TOKENS),
                  d_model=16,
                  n_layers=1,
                  n_heads=2,
                  d_ff=32,
                  max_seq_len=32))
  codec = Codec(model, _StubVocab(), _AllButEos(), device="cpu")
  enc = codec.encode("abcabc")
  assert codec.decode(enc.value) == "abcabc"
  assert codec.decode_bits(codec.encode_bits("abcabc")) == "abcabc"
