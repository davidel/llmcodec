import torch

from llmcodec.model import ModelConfig, TinyLM
from llmcodec.vocab import DEFAULT_VOCAB


def _small_config():
  return ModelConfig(vocab_size=DEFAULT_VOCAB.vocab_size,
                     d_model=32,
                     n_layers=2,
                     n_heads=2,
                     d_ff=64)


def test_forward_shape():
  model = TinyLM(_small_config())
  x = torch.randint(0, DEFAULT_VOCAB.vocab_size, (2, 5))
  logits = model(x)
  assert logits.shape == (2, 5, DEFAULT_VOCAB.vocab_size)


def test_causality():
  """Changing a future token must not affect earlier logits."""
  torch.manual_seed(0)
  model = TinyLM(_small_config())
  model.eval()
  x = torch.randint(0, DEFAULT_VOCAB.vocab_size, (1, 6))
  x2 = x.clone()
  x2[0, -1] = (x2[0, -1] + 1) % DEFAULT_VOCAB.vocab_size
  with torch.no_grad():
    a = model(x)
    b = model(x2)
  assert torch.allclose(a[0, :-1], b[0, :-1], atol=1e-5)


def test_next_token_probs_sum_to_one():
  model = TinyLM(_small_config())
  ids = torch.tensor([DEFAULT_VOCAB.bos_id, DEFAULT_VOCAB.token_id("a")])
  probs = model.next_token_probs(ids)
  assert probs.shape == (DEFAULT_VOCAB.vocab_size,)
  assert abs(probs.sum().item() - 1.0) < 1e-5


def test_save_load_roundtrip(tmp_path):
  model = TinyLM(_small_config())
  path = tmp_path / "m.pt"
  model.save(str(path))
  loaded = TinyLM.load(str(path))
  ids = torch.tensor([DEFAULT_VOCAB.bos_id])
  with torch.no_grad():
    assert torch.allclose(model.next_token_probs(ids), loaded.next_token_probs(ids))


def test_max_seq_len_guard():
  import pytest

  cfg = _small_config()
  cfg.max_seq_len = 4
  model = TinyLM(cfg)
  with pytest.raises(ValueError):
    model(torch.randint(0, 10, (1, 8)))
