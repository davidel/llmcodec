"""End-to-end CLI test: train a tiny model offline, then round-trip via the CLI.

Marked ``slow`` because it trains.  It uses the built-in sample corpus, so it
never touches the network.
"""

import pytest

from llmcodec.cli import main
from llmcodec.model import ModelConfig
from llmcodec.train import TrainConfig, train
from llmcodec.vocab import DEFAULT_VOCAB


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
  """Train a very small model once and return its checkpoint path."""
  out_dir = str(tmp_path_factory.mktemp("cli_run"))
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
  return train(cfg)


@pytest.mark.slow
def test_cli_roundtrip(checkpoint, capsys):
  """``llmcodec roundtrip`` reports a successful match for a short sentence."""
  code = main(["roundtrip", "--model", checkpoint, "--mode", "full", "the cat"])
  assert code == 0
  out = capsys.readouterr().out
  assert "match     : True" in out


@pytest.mark.slow
def test_cli_encode_bits(checkpoint, capsys):
  """``encode-bits`` prints the bit string, length and cell."""
  code = main(["encode-bits", "--model", checkpoint, "--mode", "full", "the cat"])
  assert code == 0
  out = capsys.readouterr().out
  assert "bits      : " in out
  assert "n         : " in out
  assert "cell      : " in out


@pytest.mark.slow
def test_cli_decode_bits_roundtrip(checkpoint, capsys):
  """A BitCode printed by ``encode-bits`` decodes back via ``decode-bits``."""
  assert main(["encode-bits", "--model", checkpoint, "--mode", "full", "the cat"]) == 0
  bits = None
  for line in capsys.readouterr().out.splitlines():
    if line.startswith("bits      : "):
      bits = line.split(": ", 1)[1].strip()
  assert bits
  code = main(["decode-bits", "--model", checkpoint, "--mode", "full", bits])
  assert code == 0
  out = capsys.readouterr().out
  assert "text      : 'the cat'" in out


@pytest.mark.slow
def test_cli_decode_bits_rejects_garbage(checkpoint, capsys):
  """A non-binary argument makes ``decode-bits`` fail with a hint."""
  code = main(["decode-bits", "--model", checkpoint, "--mode", "full", "0102"])
  assert code == 1
  assert "decode-bits failed" in capsys.readouterr().out
