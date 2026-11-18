"""Candidate-set selection strategies for the arithmetic codec.

At every coding step the codec needs a *candidate alphabet*: an ordered list of
tokens together with positive integer weights that determine how the current
probability interval is subdivided.  The single most important property of a
selector is **determinism**: the candidate set must be a pure function of the
model's next-token distribution (and nothing else).  Because the encoder and the
decoder compute the same distribution from the same prefix, they then derive the
same candidate set independently, and no side information has to travel with the
code.

Three strategies are provided:

``full``
    Keep every eligible token with non-zero probability.  This is *lossless*:
    any token the model can emit has a slice, so encoding never fails.  It is
    the default.

``topn``
    Keep the ``n`` most probable eligible tokens.  This mirrors the classic
    "take the top N, order them alphabetically" scheme.  Encoding raises
    :class:`~llmcodec.codec.NotEncodableError` when the true next token is not in
    the top ``n``.

``adaptive``
    Like ``topn`` but keep adding tokens (up to ``max_n``) until their combined
    probability reaches ``coverage``.  This bounds the failure rate while using
    a small alphabet when the model is confident and a larger one when it is
    not.

In every strategy the surviving tokens are laid out in **token-id order**.
For this vocabulary that is alphabetical for the character tokens, which is the
ordering requested by the design.
"""

from collections import namedtuple

import numpy as np

# Probabilities are quantized onto this many integer buckets.  A larger scale
# tracks the model distribution more faithfully but inflates the interval
# denominators (and therefore the cost of the exact-rational arithmetic).
WEIGHT_SCALE = 1 << 20


class CandidateSet(namedtuple("CandidateSet", "token_ids weights")):
  """An ordered candidate alphabet with positive integer weights.

  The weights are proportional to the (renormalized) candidate probabilities.
  Only their ratios matter, so any positive scaling is equivalent; the codec
  passes them straight to :meth:`~llmcodec.interval.Interval.subdivide`.

  Attributes:
    token_ids: The candidate token ids, in the order they are laid out.
    weights: The positive integer weight of each candidate, aligned with
      ``token_ids``.
  """

  __slots__ = ()

  def __new__(cls, token_ids, weights):
    """Validate that ids and weights are aligned and positive.

    Raises:
      ValueError: If the two sequences differ in length, if the set is empty, or
        if any weight is not strictly positive.
    """
    if len(token_ids) != len(weights):
      raise ValueError("token_ids and weights must have equal length")
    if not token_ids:
      raise ValueError("candidate set must be non-empty")
    if any(w <= 0 for w in weights):
      raise ValueError("all weights must be strictly positive")
    return super().__new__(cls, tuple(token_ids), tuple(weights))

  def __len__(self):
    """Return the number of candidates."""
    return len(self.token_ids)

  def index_of(self, token_id):
    """Return the position of ``token_id`` in the candidate list, or ``None``."""
    try:
      return self.token_ids.index(token_id)
    except ValueError:
      return None


def _quantize(probs):
  """Convert probabilities into strictly positive integer weights.

  Values are rounded to the nearest bucket and floored at one, so a surviving
  candidate always gets a non-zero slice (which keeps the interval arithmetic
  well defined even for extremely small probabilities).

  Args:
    probs: A 1-D array of probabilities in ``[0, 1]``.

  Returns:
    An integer array of the same shape, with every entry at least ``1``.
  """
  weights = np.rint(probs * WEIGHT_SCALE).astype(np.int64)
  weights[weights < 1] = 1
  return weights


class CandidateSelector:
  """Base class for candidate-set strategies.

  Subclasses implement :meth:`select`, which maps a full next-token probability
  vector plus the set of *eligible* token ids to a :class:`CandidateSet`.
  """

  name = "base"

  def select(self, probs, eligible):
    """Return the candidate set for the given distribution.

    Args:
      probs: The full next-token probability vector, indexed by token id.
      eligible: The token ids the selector may choose from.

    Returns:
      The chosen :class:`CandidateSet`.

    Raises:
      NotImplementedError: Always, in the base class.
    """
    raise NotImplementedError

  def describe(self):
    """Return a JSON-friendly description of the strategy and its settings."""
    return {"strategy": self.name}


class FullVocabSelector(CandidateSelector):
  """Keep every eligible token with non-zero probability.

  This selector is lossless: it never discards a token the model can emit, so
  encoding can never fail.  It is the default because correctness does not
  depend on the quality of the model.
  """

  name = "full"

  def select(self, probs, eligible):
    """Select all eligible tokens with non-zero probability, sorted by id.

    Args:
      probs: The next-token probability vector.
      eligible: The token ids to consider.

    Returns:
      A candidate set covering every eligible token with ``prob > 0``.

    Raises:
      ValueError: If the model assigns zero probability to every eligible token.
    """
    ids = [t for t in eligible if probs[t] > 0.0]
    if not ids:
      raise ValueError("model assigned zero probability to every eligible token")
    ids.sort()
    weights = _quantize(probs[ids])
    return CandidateSet(ids, [int(w) for w in weights])


class TopNSelector(CandidateSelector):
  """Keep the ``n`` most probable eligible tokens.

  Ties in probability are broken by token id so that the result is fully
  deterministic.

  Attributes:
    n: The maximum number of candidates to keep.
    name: The strategy name (``"topn"``).
  """

  name = "topn"

  def __init__(self, n=5):
    self.n = n

  def select(self, probs, eligible):
    """Select the ``n`` most probable eligible tokens, then sort by id.

    Args:
      probs: The next-token probability vector.
      eligible: The token ids to consider.

    Returns:
      A candidate set of at most ``n`` tokens.

    Raises:
      ValueError: If the model assigns zero probability to every eligible token.
    """
    cand = [t for t in eligible if probs[t] > 0.0]
    if not cand:
      raise ValueError("model assigned zero probability to every eligible token")
    cand.sort(key=lambda t: (-probs[t], t))
    chosen = cand[:self.n]
    chosen.sort()
    weights = _quantize(probs[chosen])
    return CandidateSet(chosen, [int(w) for w in weights])

  def describe(self):
    """Return the strategy name and ``n``."""
    return {"strategy": self.name, "n": self.n}


class AdaptiveSelector(CandidateSelector):
  """Grow the candidate set until it captures ``coverage`` of the mass.

  Tokens are added in descending probability order until either the cumulative
  probability reaches ``coverage`` or ``max_n`` tokens have been chosen.

  Attributes:
    max_n: The hard cap on the number of candidates.
    coverage: The target cumulative probability mass, in ``(0, 1]``.
    name: The strategy name (``"adaptive"``).
  """

  name = "adaptive"

  def __init__(self, max_n=16, coverage=0.999):
    self.max_n = max_n
    self.coverage = coverage

  def select(self, probs, eligible):
    """Select enough top tokens to reach ``coverage`` (capped at ``max_n``).

    Args:
      probs: The next-token probability vector.
      eligible: The token ids to consider.

    Returns:
      A candidate set capturing at least ``coverage`` of the eligible mass, or
      ``max_n`` tokens, whichever comes first.

    Raises:
      ValueError: If the model assigns zero probability to every eligible token.
    """
    cand = [t for t in eligible if probs[t] > 0.0]
    if not cand:
      raise ValueError("model assigned zero probability to every eligible token")
    cand.sort(key=lambda t: (-probs[t], t))
    chosen = []
    acc = 0.0
    for t in cand:
      chosen.append(t)
      acc += float(probs[t])
      if acc >= self.coverage or len(chosen) >= self.max_n:
        break
    chosen.sort()
    weights = _quantize(probs[chosen])
    return CandidateSet(chosen, [int(w) for w in weights])

  def describe(self):
    """Return the strategy name, ``max_n`` and ``coverage``."""
    return {"strategy": self.name, "max_n": self.max_n, "coverage": self.coverage}


def build_selector(mode, **kwargs):
  """Build a selector from a mode name.

  Args:
    mode: One of ``"full"``, ``"topn"`` or ``"adaptive"`` (case-insensitive).
    **kwargs: Strategy-specific settings.  ``topn`` reads ``n``; ``adaptive``
      reads ``max_n`` and ``coverage``.  Unknown keys are ignored.

  Returns:
    The constructed :class:`CandidateSelector`.

  Raises:
    ValueError: If ``mode`` is not one of the known strategies.
  """
  mode = mode.lower()
  if mode == "full":
    return FullVocabSelector()
  if mode == "topn":
    return TopNSelector(n=int(kwargs.get("n", 5)))
  if mode == "adaptive":
    return AdaptiveSelector(
        max_n=int(kwargs.get("max_n", 16)),
        coverage=float(kwargs.get("coverage", 0.999)),
    )
  raise ValueError(f"unknown selector mode {mode!r}")
