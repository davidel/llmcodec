"""Unit tests for the training helpers that do not need a full run.

These cover the pieces of :mod:`llmcodec.train` that the end-to-end smoke test
exercises only implicitly: the learning-rate schedule, the configuration
round-trip, token preparation and the checkpoint written by ``train``.
"""

import json
import os

import pytest
import torch

from llmcodec.model import ModelConfig
from llmcodec.train import TrainConfig, _lr_at, _prepare_tokens, train
from llmcodec.vocab import DEFAULT_VOCAB


def test_train_config_to_dict_includes_model():
  """``to_dict`` flattens the scalar fields and nests the model config."""
  cfg = TrainConfig(steps=10, lr=0.5, model=ModelConfig(vocab_size=43))
  data = cfg.to_dict()
  assert data["steps"] == 10
  assert data["lr"] == 0.5
  assert data["model"] == cfg.model.to_dict()


def test_train_config_to_dict_is_json_serializable():
  """The dumped config must be JSON-friendly so training can log it."""
  cfg = TrainConfig()
  json.dumps(cfg.to_dict())


def test_lr_schedule_warms_up_linearly():
  """During warmup the learning rate rises linearly from lr/warmup."""
  cfg = TrainConfig(lr=1.0, warmup_steps=10, steps=100, min_lr=0.0)
  assert _lr_at(0, cfg) == pytest.approx(0.1)
  assert _lr_at(9, cfg) == pytest.approx(1.0)
  # Monotonic non-decreasing across warmup.
  rates = [_lr_at(step, cfg) for step in range(cfg.warmup_steps)]
  assert rates == sorted(rates)


def test_lr_schedule_decays_to_min_lr():
  """After warmup the rate decays cosine-style and bottoms out at ``min_lr``."""
  cfg = TrainConfig(lr=1.0, min_lr=0.01, warmup_steps=10, steps=110)
  # At the end of warmup and start of decay the rate is at its peak.
  assert _lr_at(10, cfg) == pytest.approx(1.0)
  # Halfway through the decay the cosine term has halved the excess.
  assert 0.01 < _lr_at(60, cfg) < 1.0
  # At and beyond the final step the rate is pinned to the floor.
  assert _lr_at(cfg.steps, cfg) == pytest.approx(cfg.min_lr)
  assert _lr_at(cfg.steps + 100, cfg) == pytest.approx(cfg.min_lr)


def test_lr_schedule_is_monotonic_after_warmup():
  """The decay phase never increases the learning rate."""
  cfg = TrainConfig(lr=1.0, min_lr=0.01, warmup_steps=5, steps=105)
  decay = [_lr_at(step, cfg) for step in range(cfg.warmup_steps, cfg.steps)]
  assert decay == sorted(decay, reverse=True)


def test_prepare_tokens_sample_source_is_offline():
  """The built-in sample corpus needs no network and yields valid ids."""
  cfg = TrainConfig(source="sample")
  tokens = _prepare_tokens(cfg, DEFAULT_VOCAB)
  assert tokens.dim() == 1
  assert tokens.numel() > 0
  assert tokens.dtype == torch.long
  assert int(tokens.min()) >= 0
  assert int(tokens.max()) < DEFAULT_VOCAB.vocab_size


def test_prepare_tokens_rejects_unknown_source():
  """An unknown source is a configuration error, not a silent fallback."""
  cfg = TrainConfig(source="does-not-exist")
  with pytest.raises(ValueError):
    _prepare_tokens(cfg, DEFAULT_VOCAB)


@pytest.mark.slow
def test_train_writes_checkpoint_and_config(tmp_path):
  """``train`` writes both ``model.pt`` and ``train_config.json``."""
  out_dir = str(tmp_path / "run")
  cfg = TrainConfig(
      source="sample",
      steps=5,
      batch_size=4,
      block_size=32,
      warmup_steps=2,
      out_dir=out_dir,
      device="cpu",
      num_threads=2,
      eval_every=0,
      log_every=1000,
      model=ModelConfig(vocab_size=DEFAULT_VOCAB.vocab_size,
                        d_model=16,
                        n_layers=1,
                        n_heads=2,
                        d_ff=32),
  )
  ckpt = train(cfg)
  assert os.path.isfile(ckpt)
  config_path = os.path.join(out_dir, "train_config.json")
  assert os.path.isfile(config_path)
  with open(config_path) as fh:
    dumped = json.load(fh)
  assert dumped["train"]["steps"] == 5
  assert dumped["vocab"]["itos"] == list(DEFAULT_VOCAB.itos)
