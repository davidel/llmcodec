"""Dataset loading and batching.

The default corpus is `TinyStories <https://huggingface.co/datasets/roneneldan/TinyStories>`_.
Each story is lower-cased, tokenized with :class:`~llmcodec.vocab.Vocab`, wrapped
in ``<bos> ... <eos>`` and concatenated into one long token stream.  The stream
is then chopped into fixed-length blocks for next-token training.

A small built-in corpus is also available so the package (and its tests) can run
without network access.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch

from .vocab import Vocab

# A handful of TinyStories-style sentences used for offline smoke tests.
SAMPLE_STORIES: tuple[str, ...] = (
    "Once upon a time, there was a little cat. The cat was very small.",
    "Tom had a red ball. He played with the ball all day, and he was happy.",
    "The dog ran in the park. It saw a bird, and the bird flew away.",
    "Sara found a shiny rock. She put it in her pocket and went home.",
    "One day, a big fish swam in the sea. It met a small crab.",
    "Ben wanted a cake. His mom made a cake with 3 candles.",
    "The sun was hot. The kids drank water and played in the yard.",
    "A little girl lost her shoe. Her friend helped her find it.",
)


def load_tinystories(
    split: str = "train",
    max_stories: int | None = None,
) -> list[str]:
    """Download (if needed) and return TinyStories stories as strings."""
    from datasets import load_dataset  # imported lazily: heavy dependency

    ds = load_dataset("roneneldan/TinyStories", split=split)
    if max_stories is not None:
        ds = ds.select(range(min(max_stories, len(ds))))
    return [row["text"] for row in ds]


def sample_stories() -> list[str]:
    """Return the small built-in corpus (no network required)."""
    return list(SAMPLE_STORIES)


def build_token_stream(
    texts: Sequence[str],
    vocab: Vocab,
    *,
    add_bos: bool = True,
    add_eos: bool = True,
) -> torch.Tensor:
    """Concatenate ``texts`` into a single ``long`` token tensor.

    Stories are separated by ``<bos>``/``<eos>`` so that the model learns both
    sentence boundaries.
    """
    ids: list[int] = []
    for text in texts:
        ids.extend(vocab.encode(text, add_bos=add_bos, add_eos=add_eos))
    return torch.tensor(ids, dtype=torch.long)


@dataclass
class BlockDataset:
    """A ``torch``-style dataset of next-token prediction blocks.

    Each item is ``(x, y)`` where ``x`` is a window of ``block_size`` tokens and
    ``y`` is the same window shifted by one position.
    """

    tokens: torch.Tensor
    block_size: int

    def __post_init__(self) -> None:
        if self.tokens.numel() < self.block_size + 1:
            raise ValueError("token stream is too short for the requested block_size")

    def __len__(self) -> int:
        return (self.tokens.numel() - 1) // self.block_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.block_size
        end = start + self.block_size
        x = self.tokens[start:end]
        y = self.tokens[start + 1 : end + 1]
        return x, y

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for i in range(len(self)):
            yield self[i]


def iter_batches(
    dataset: BlockDataset,
    batch_size: int,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield ``(x, y)`` mini-batches of shape ``(batch, block_size)``."""
    order = (
        torch.randperm(len(dataset), generator=generator) if shuffle else torch.arange(len(dataset))
    )
    for start in range(0, len(dataset), batch_size):
        idx = order[start : start + batch_size]
        xs = torch.stack([dataset[int(i)][0] for i in idx])
        ys = torch.stack([dataset[int(i)][1] for i in idx])
        yield xs, ys
