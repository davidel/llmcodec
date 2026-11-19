"""Unit tests for the training helpers that do not need a full run.

These cover the pieces of :mod:`llmcodec.train` that the end-to-end smoke test
exercises only implicitly: the learning-rate schedule, the configuration
round-trip, token preparation and the checkpoint written by ``train``.
"""

import json
import os
from types import SimpleNamespace

import pytest
import torch

from llmcodec.model import ModelConfig
from llmcodec.train import (
    TrainConfig,
    _lr_at,
    _prepare_tokens,
    add_train_arguments,
    config_from_args,
    load_config,
    train,
)
from llmcodec.vocab import DEFAULT_VOCAB


def test_from_dict_roundtrip():
  """``from_dict(to_dict())`` reproduces the configuration."""
  cfg = TrainConfig(steps=11,
                    lr=0.25,
                    source="sample",
                    model=ModelConfig(vocab_size=43,
                                      d_model=32,
                                      n_layers=2,
                                      n_heads=2,
                                      d_ff=64))
  assert TrainConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()


def test_from_dict_accepts_wrapped_form():
  """The ``{"train": ..., "vocab": ...}`` shape loads and ignores the vocab."""
  wrapped = {"train": TrainConfig(steps=9).to_dict(), "vocab": {"size": 43}}
  assert TrainConfig.from_dict(wrapped).steps == 9


def test_from_dict_ignores_unknown_keys():
  """Unknown keys are dropped, matching ``ModelConfig.from_dict``."""
  assert TrainConfig.from_dict({"steps": 4, "bogus": 1}).steps == 4


def test_validate_rejects_bad_architecture():
  """``validate`` turns misconfigurations into actionable errors."""
  with pytest.raises(ValueError, match="divisible"):
    TrainConfig(model=ModelConfig(vocab_size=43, n_heads=5)).validate()
  with pytest.raises(ValueError, match="exceeds max_seq_len"):
    TrainConfig(block_size=300).validate()
  with pytest.raises(ValueError, match="vocabulary size"):
    TrainConfig(model=ModelConfig(vocab_size=10)).validate(DEFAULT_VOCAB)
  with pytest.raises(ValueError, match="dropout"):
    TrainConfig(model=ModelConfig(vocab_size=43, dropout=1.0)).validate()
  with pytest.raises(ValueError, match="steps"):
    TrainConfig(steps=0).validate()
  with pytest.raises(ValueError, match="batch_size"):
    TrainConfig(batch_size=0).validate()


def test_config_from_args_defaults():
  """A namespace with only ``config=None`` yields the default config."""
  cfg = config_from_args(SimpleNamespace(config=None))
  assert cfg.to_dict() == TrainConfig().to_dict()


def test_config_from_args_flag_overrides(tmp_path):
  """Explicit flags beat ``--config`` values; unset flags keep file values."""
  path = tmp_path / "cfg.json"
  path.write_text(json.dumps({"steps": 123, "lr": 0.01, "batch_size": 8}))
  args = SimpleNamespace(config=str(path), steps=7)
  cfg = config_from_args(args)
  assert cfg.steps == 7
  assert cfg.lr == 0.01
  assert cfg.batch_size == 8


def test_config_from_args_missing_file():
  """A bad ``--config`` path raises a clear error."""
  with pytest.raises(ValueError, match="could not read config file"):
    config_from_args(SimpleNamespace(config="does/not/exist.json"))


def test_config_from_args_invalid_json(tmp_path):
  """Malformed JSON raises a clear error."""
  path = tmp_path / "bad.json"
  path.write_text("{not json")
  with pytest.raises(ValueError, match="not valid JSON"):
    config_from_args(SimpleNamespace(config=str(path)))


def test_cli_flags_reach_model_config():
  """The architecture flags populate ``ModelConfig`` through the CLI parser."""
  import argparse

  parser = argparse.ArgumentParser()
  add_train_arguments(parser)
  args = parser.parse_args([
      "--d-model", "64", "--n-layers", "2", "--n-heads", "4", "--d-ff", "128",
      "--max-seq-len", "64", "--dropout", "0.1", "--steps", "5"
  ])
  cfg = config_from_args(args)
  assert cfg.model.d_model == 64
  assert cfg.model.n_layers == 2
  assert cfg.model.n_heads == 4
  assert cfg.model.d_ff == 128
  assert cfg.model.max_seq_len == 64
  assert cfg.model.dropout == 0.1
  assert cfg.steps == 5
  assert cfg.model.vocab_size == 43


def test_load_config_configs_default():
  """The shipped ``configs/default.json`` loads as-is."""
  cfg = load_config("configs/default.json")
  assert cfg.steps == 50_000
  assert cfg.model.d_model == 512
  assert cfg.model.n_layers == 6


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
