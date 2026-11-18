"""A tiny end-to-end test: train briefly, then round-trip a sentence.

Marked ``slow`` because it actually trains.  Run with ``pytest -m slow``.
"""

import pytest

from llmcodec.candidates import build_selector
from llmcodec.codec import Codec
from llmcodec.model import ModelConfig
from llmcodec.train import TrainConfig, train
from llmcodec.vocab import DEFAULT_VOCAB


@pytest.mark.slow
def test_train_then_roundtrip(tmp_path):
    cfg = TrainConfig(
        source="sample",
        steps=30,
        batch_size=8,
        block_size=32,
        warmup_steps=5,
        out_dir=str(tmp_path / "run"),
        device="cpu",
        num_threads=2,
        eval_every=0,
        log_every=1000,
        model=ModelConfig(
            vocab_size=DEFAULT_VOCAB.vocab_size, d_model=32, n_layers=2, n_heads=2, d_ff=64
        ),
    )
    ckpt = train(cfg)
    from llmcodec.model import TinyLM

    model = TinyLM.load(ckpt)
    codec = Codec(model, DEFAULT_VOCAB, build_selector("full"))
    text = "the cat"
    enc = codec.encode(text)
    assert codec.decode(enc.value) == text
