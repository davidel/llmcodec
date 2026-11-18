"""Dataset loading, tokenization and batching.

The default corpus is `TinyStories
<https://huggingface.co/datasets/roneneldan/TinyStories>`_.  Each story is
lower-cased, tokenized with :class:`~llmcodec.vocab.Vocab`, wrapped in
``<bos> ... <eos>`` and concatenated into one long token stream.  The stream is
then chopped into fixed-length blocks for next-token training.

A small built-in corpus (:data:`SAMPLE_STORIES`) is also provided so the package
and its tests can run without network access.
"""

import torch

# A handful of TinyStories-style sentences used for offline smoke tests and for
# the ``--source sample`` training mode.
SAMPLE_STORIES = (
    "Once upon a time, there was a little cat. The cat was very small.",
    "Tom had a red ball. He played with the ball all day, and he was happy.",
    "The dog ran in the park. It saw a bird, and the bird flew away.",
    "Sara found a shiny rock. She put it in her pocket and went home.",
    "One day, a big fish swam in the sea. It met a small crab.",
    "Ben wanted a cake. His mom made a cake with 3 candles.",
    "The sun was hot. The kids drank water and played in the yard.",
    "A little girl lost her shoe. Her friend helped her find it.",
)


def load_tinystories(split="train", max_stories=None):
  """Download (if needed) and return TinyStories stories as strings.

  The ``datasets`` package is imported lazily so that it remains an optional
  dependency for users who only need the offline corpus.

  Args:
    split: The dataset split to load, e.g. ``"train"`` or ``"validation"``.
    max_stories: If given, keep only the first ``max_stories`` stories.

  Returns:
    The list of story strings.
  """
  from datasets import load_dataset  # heavy, optional dependency

  ds = load_dataset("roneneldan/TinyStories", split=split)
  if max_stories is not None:
    ds = ds.select(range(min(max_stories, len(ds))))
  return [row["text"] for row in ds]


def sample_stories():
  """Return the small built-in corpus (no network required)."""
  return list(SAMPLE_STORIES)


def build_token_stream(texts, vocab, add_bos=True, add_eos=True):
  """Concatenate ``texts`` into a single ``long`` token tensor.

  Each story is optionally wrapped in ``<bos>``/``<eos>`` so the model learns
  where sentences begin and end -- both markers matter to the codec, which
  starts from ``<bos>`` and terminates on ``<eos>``.

  Args:
    texts: The story strings.
    vocab: The vocabulary used to tokenize.
    add_bos: Prepend ``<bos>`` to each story.
    add_eos: Append ``<eos>`` to each story.

  Returns:
    A 1-D ``long`` tensor holding the concatenated token ids.
  """
  ids = []
  for text in texts:
    ids.extend(vocab.encode(text, add_bos=add_bos, add_eos=add_eos))
  return torch.tensor(ids, dtype=torch.long)


class BlockDataset:
  """A dataset of next-token prediction blocks.

  Each item is ``(x, y)`` where ``x`` is a window of ``block_size`` tokens and
  ``y`` is the same window shifted one position to the right, so that training
  on ``y`` given ``x`` is standard next-token prediction.

  Attributes:
    tokens: The flat token stream.
    block_size: The length of each window.
  """

  def __init__(self, tokens, block_size):
    """Store the stream and validate that it is long enough for one block.

    Raises:
      ValueError: If the token stream is shorter than ``block_size + 1``.
    """
    if tokens.numel() < block_size + 1:
      raise ValueError("token stream is too short for the requested block_size")
    self.tokens = tokens
    self.block_size = block_size

  def __len__(self):
    """Return the number of (non-overlapping) blocks."""
    return (self.tokens.numel() - 1) // self.block_size

  def __getitem__(self, index):
    """Return the ``index``-th ``(x, y)`` block pair.

    Args:
      index: The block index.

    Returns:
      A tuple ``(x, y)`` of 1-D tensors of length ``block_size``.
    """
    start = index * self.block_size
    end = start + self.block_size
    x = self.tokens[start:end]
    y = self.tokens[start + 1:end + 1]
    return x, y

  def __iter__(self):
    """Iterate over the blocks in order."""
    for i in range(len(self)):
      yield self[i]


def iter_batches(dataset, batch_size, shuffle=True, generator=None):
  """Yield mini-batches of stacked blocks.

  Args:
    dataset: The block dataset to sample from.
    batch_size: The number of blocks per batch.
    shuffle: Whether to shuffle the block order.
    generator: Optional RNG used for reproducible shuffling.

  Yields:
    Tuples ``(x, y)`` of shape ``(batch, block_size)``.
  """
  if shuffle:
    order = torch.randperm(len(dataset), generator=generator)
  else:
    order = torch.arange(len(dataset))
  for start in range(0, len(dataset), batch_size):
    idx = order[start:start + batch_size]
    xs = torch.stack([dataset[int(i)][0] for i in idx])
    ys = torch.stack([dataset[int(i)][1] for i in idx])
    yield xs, ys
