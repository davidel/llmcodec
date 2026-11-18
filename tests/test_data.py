import torch

from llmcodec.data import BlockDataset, build_token_stream, iter_batches, sample_stories
from llmcodec.vocab import DEFAULT_VOCAB


def test_build_token_stream_wraps_stories():
    stream = build_token_stream(sample_stories()[:2], DEFAULT_VOCAB)
    assert stream.dtype == torch.long
    assert stream[0].item() == DEFAULT_VOCAB.bos_id
    assert DEFAULT_VOCAB.eos_id in stream.tolist()


def test_block_dataset_shapes():
    stream = build_token_stream(sample_stories(), DEFAULT_VOCAB)
    ds = BlockDataset(stream, block_size=16)
    assert len(ds) > 0
    x, y = ds[0]
    assert x.shape == (16,)
    assert y.shape == (16,)
    assert torch.equal(x[1:], y[:-1])


def test_block_dataset_too_short():
    import pytest

    with pytest.raises(ValueError):
        BlockDataset(torch.tensor([1, 2, 3]), block_size=16)


def test_iter_batches():
    stream = build_token_stream(sample_stories(), DEFAULT_VOCAB)
    ds = BlockDataset(stream, block_size=16)
    batches = list(iter_batches(ds, batch_size=4, shuffle=False))
    assert len(batches) >= 1
    x, y = batches[0]
    assert x.shape[1] == 16
    assert x.shape == y.shape
