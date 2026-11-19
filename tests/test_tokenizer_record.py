"""Tests for tokenizer persistence in ``train_config.json``.

The identity record (name, revision, content hash) and its decode-side
re-verification are exercised without network and without transformers:
the training runs use the offline sample corpus with a hand-rolled
tokenizer, and the CLI reader tests use temporary JSON files.
"""

import hashlib
import json
from argparse import Namespace

import pytest

from llmcodec import cli
from llmcodec.model import ModelConfig
from llmcodec.train import TrainConfig, train
from llmcodec.vocab import DEFAULT_VOCAB


class _NamedTokenizer:
  """Minimal tokenizer carrying the identity attributes ``train`` records."""

  def __init__(self, name="toy-tokenizer", revision=None):
    self.name = name
    self.revision = revision
    self._tokens = ["<pad>", "<bos>", "<eos>", "<unk>", "a", "b", "c"]

  def encode(self, text, add_bos=False, add_eos=False):
    del add_bos, add_eos  # the tiny corpus needs no real special handling
    return [4 + "abc".index(ch) for ch in text if ch in "abc"]

  def decode(self, ids, skip_special=False):
    del skip_special
    return "".join(self._tokens[i] for i in ids)

  @property
  def itos(self):
    return self._tokens

  @property
  def vocab_size(self):
    return len(self._tokens)

  @property
  def bos_id(self):
    return 1

  @property
  def eos_id(self):
    return 2

  @property
  def unk_id(self):
    return 3

  @property
  def pad_id(self):
    return 0


def _tiny_config(tmp_path, vocab_size=7):
  """A one-step training config writing into ``tmp_path``.

  The model vocabulary must match the tokenizer's, as ``validate`` requires.
  """
  return TrainConfig(source="sample",
                     steps=1,
                     batch_size=2,
                     block_size=8,
                     eval_every=0,
                     eval_batches=1,
                     model=ModelConfig(vocab_size=vocab_size,
                                       d_model=16,
                                       n_layers=1,
                                       n_heads=2,
                                       d_ff=32,
                                       max_seq_len=32),
                     out_dir=str(tmp_path))


def _read_record(tmp_path):
  with open(tmp_path / "train_config.json") as fh:
    return json.load(fh)


def test_char_level_record_stays_inline(tmp_path):
  """The built-in vocabulary is inlined under ``vocab`` as before."""
  train(_tiny_config(tmp_path, vocab_size=DEFAULT_VOCAB.vocab_size),
        vocab=DEFAULT_VOCAB)
  record = _read_record(tmp_path)
  assert record["vocab"] == DEFAULT_VOCAB.to_dict()
  assert "tokenizer" not in record


def test_named_tokenizer_recorded_by_identity(tmp_path):
  """A named tokenizer is recorded as identity plus content hash."""
  vocab = _NamedTokenizer(name="toy-tokenizer", revision="v1")
  train(_tiny_config(tmp_path), vocab=vocab)
  record = _read_record(tmp_path)
  block = record["tokenizer"]
  assert block["name"] == "toy-tokenizer"
  assert block["revision"] == "v1"
  digest = hashlib.sha256("\x1f".join(vocab.itos).encode()).hexdigest()
  assert block["sha256"] == digest
  assert "vocab" not in record


def test_recorded_tokenizer_reader(tmp_path):
  """The CLI reader returns the identity tuple or ``None``."""
  path = tmp_path / "model.pt"
  path.write_bytes(b"")  # only the sibling JSON matters
  config = tmp_path / "train_config.json"
  config.write_text(
      json.dumps(
          {"tokenizer": {
              "name": "gpt2",
              "revision": "abc123",
              "sha256": "deadbeef"
          }}))
  args = Namespace(model=str(path))
  assert cli._recorded_tokenizer(args) == ("gpt2", "abc123", "deadbeef")

  config.write_text(json.dumps({"vocab": {"itos": ["a"]}}))
  assert cli._recorded_tokenizer(args) is None

  args_no_model = Namespace(model=None)
  assert cli._recorded_tokenizer(args_no_model) is None


def test_recorded_tokenizer_hash_mismatch_is_rejected(tmp_path, monkeypatch):
  """A drifted vocabulary fails loudly instead of corrupting decodes."""

  class _DriftedLoader:

    def __init__(self, drifted_tokens):
      self._tokens = drifted_tokens

    def load(self, name, revision=None, device=None):
      del name, revision, device
      return _NamedTokenizer()

  drifted = _NamedTokenizer()
  drifted._tokens = ["<pad>", "<bos>", "<eos>", "<unk>", "x", "y", "z"]
  monkeypatch.setattr(cli, "HFTokenizer", _DriftedLoader(drifted))

  path = tmp_path / "model.pt"
  path.write_bytes(b"")
  config = tmp_path / "train_config.json"
  digest = hashlib.sha256("\x1f".join(drifted.itos).encode()).hexdigest()
  config.write_text(
      json.dumps({"tokenizer": {
          "name": "gpt2",
          "revision": None,
          "sha256": digest
      }}))
  with pytest.raises(ValueError, match="content hash mismatch"):
    cli._load_tokenizer(
        Namespace(model=str(path), tokenizer=None, revision=None, device="cpu"))


def test_explicit_tokenizer_beats_recorded_identity(tmp_path, monkeypatch):
  """``--tokenizer`` wins over whatever the checkpoint recorded."""
  loaded = []

  class _SpyLoader:

    def load(self, name, revision=None, device=None):
      loaded.append(name)
      return _NamedTokenizer(name=name)

  monkeypatch.setattr(cli, "HFTokenizer", _SpyLoader())
  args = Namespace(model=str(tmp_path / "model.pt"),
                   tokenizer="explicit-tokenizer",
                   revision=None,
                   device="cpu")
  vocab = cli._load_tokenizer(args)
  assert loaded == ["explicit-tokenizer"]
  assert vocab.name == "explicit-tokenizer"
