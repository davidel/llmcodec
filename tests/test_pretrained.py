"""Tests for the Hugging Face model adapter.

The adapter tests build a tiny randomly-initialized ``GPT2LMHeadModel`` in
memory, so they never download anything from the hub.  They are skipped when
``transformers`` is not installed (the fast CI suite does not require it).
"""

import sys

import pytest
import torch

from llmcodec.model import ModelConfig, TinyLM
from llmcodec.pretrained import HFCausalLM, NextTokenModel, _resolve_dtype

VOCAB = 43


def _hf_adapter(context_len=None, model=None):
  """Build a tiny random GPT-2 wrapped in the adapter (no downloads).

  Args:
    context_len: Optional maximum prefix length for the adapter.
    model: Optionally wrap an existing module instead of building a new
      random one, so two adapters can share identical weights.
  """
  pytest.importorskip("transformers")
  from transformers import GPT2Config, GPT2LMHeadModel

  if model is None:
    config = GPT2Config(vocab_size=VOCAB,
                        n_positions=16,
                        n_embd=32,
                        n_layer=2,
                        n_head=4)
    model = GPT2LMHeadModel(config)
  return HFCausalLM(model,
                    name="test-gpt2",
                    dtype="float32",
                    device="cpu",
                    context_len=context_len)


def test_tiny_lm_satisfies_protocol():
  """The built-in model already implements the codec's model interface."""
  tiny = TinyLM(
      ModelConfig(vocab_size=VOCAB,
                  d_model=32,
                  n_layers=2,
                  n_heads=4,
                  d_ff=64,
                  max_seq_len=32))
  assert isinstance(tiny, NextTokenModel)


def test_next_token_probs_shape_and_normalization():
  """A 1-D prefix yields a finite ``(vocab_size,)`` distribution summing to 1."""
  hf = _hf_adapter()
  ids = torch.randint(0, VOCAB, (5,))
  probs = hf.next_token_probs(ids)
  assert probs.shape == (VOCAB,)
  assert probs.dtype == torch.float32
  assert torch.isfinite(probs).all()
  assert (probs >= 0).all()
  assert probs.sum().item() == pytest.approx(1.0, abs=1e-5)


def test_next_token_probs_batched():
  """2-D input yields per-row distributions; identical rows agree exactly."""
  hf = _hf_adapter()
  ids = torch.randint(0, VOCAB, (2, 4))
  probs = hf.next_token_probs(ids)
  assert probs.shape == (2, VOCAB)
  assert probs[0].sum().item() == pytest.approx(1.0, abs=1e-5)
  assert torch.equal(probs[0], probs[1]) is (torch.equal(ids[0], ids[1]))


def test_truncates_long_prefix():
  """Prefixes longer than ``context_len`` are cut to their last tokens.

  Without truncation GPT-2 (``n_positions=16``) would raise on a 20-token
  prefix; with ``context_len=8`` the result must match a manual forward on
  the final 8 tokens.
  """
  hf = _hf_adapter(context_len=8)
  ids = torch.randint(0, VOCAB, (20,))
  probs = hf.next_token_probs(ids)
  reference = _hf_adapter(model=hf.model)
  assert probs.shape == (VOCAB,)
  assert torch.equal(probs, reference.next_token_probs(ids[-8:]))


def test_amp_dtype_kwarg_is_accepted():
  """The adapter tolerates the codec's ``amp_dtype`` argument (ignored)."""
  hf = _hf_adapter()
  ids = torch.randint(0, VOCAB, (5,))
  assert torch.equal(hf.next_token_probs(ids),
                     hf.next_token_probs(ids, amp_dtype=torch.float16))


def test_describe_reports_configuration():
  """``describe`` exposes the adapter settings for logging."""
  hf = _hf_adapter(context_len=8)
  info = hf.describe()
  assert info["type"] == "hf"
  assert info["name"] == "test-gpt2"
  assert info["dtype"] == "float32"
  assert info["context_len"] == 8
  assert info["vocab_size"] == VOCAB


def test_vocab_size_property():
  """``vocab_size`` mirrors the model config."""
  assert _hf_adapter().vocab_size == VOCAB


def test_resolve_dtype_rejects_unknown():
  """Unknown dtype names fail with a helpful message."""
  with pytest.raises(ValueError, match="unknown dtype"):
    _resolve_dtype("int8", torch.device("cpu"))


def test_resolve_dtype_auto_on_cpu():
  """``auto`` falls back to float32 on CPU."""
  assert _resolve_dtype("auto", torch.device("cpu")) is torch.float32


def test_load_without_transformers_hint(monkeypatch):
  """Missing ``transformers`` produces an install hint, not a bare traceback."""
  monkeypatch.setitem(sys.modules, "transformers", None)
  with pytest.raises(ImportError, match=r"llmcodec\[hf\]"):
    HFCausalLM.load("gpt2", device="cpu")
