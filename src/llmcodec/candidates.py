"""Candidate-set selection strategies.

At every coding step the codec needs a *candidate alphabet*: an ordered list of
tokens together with integer weights that determine how the current probability
interval is subdivided.  The candidate set must be a **deterministic function of
the model's next-token distribution only** so that the encoder and the decoder
-- which compute the same distribution from the same prefix -- agree without any
side-channel information.

Three strategies are provided:

``full``
    Every eligible token with non-zero probability.  This is *lossless*: any
    character the model can emit has a slice, so encoding never fails.  This is
    the default.

``topn``
    The ``n`` most probable tokens (prob > 0), renormalized.  Matches the
    "top N, ordered alphabetically" scheme from the design.  Encoding can fail
    when the true next token falls outside the top ``n``.

``adaptive``
    Like ``topn`` but keeps adding tokens (up to ``max_n``) until their combined
    probability reaches ``coverage``.  This bounds the failure rate while using
    a smaller alphabet when the model is confident.

In every strategy the surviving tokens are laid out in **token-id order**
(which, for this vocabulary, is alphabetical for the character tokens), exactly
as specified.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# Number of integer buckets a probability is quantized into.  Bigger = more
# faithful to the model distribution but larger interval denominators.
WEIGHT_SCALE = 1 << 20


@dataclass(frozen=True)
class CandidateSet:
    """An ordered candidate alphabet with integer weights."""

    token_ids: tuple[int, ...]
    weights: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.weights):
            raise ValueError("token_ids and weights must have equal length")
        if not self.token_ids:
            raise ValueError("candidate set must be non-empty")
        if any(w <= 0 for w in self.weights):
            raise ValueError("all weights must be strictly positive")

    def __len__(self) -> int:
        return len(self.token_ids)

    def index_of(self, token_id: int) -> int | None:
        try:
            return self.token_ids.index(token_id)
        except ValueError:
            return None


def _quantize(probs: np.ndarray) -> np.ndarray:
    """Turn probabilities into positive integer weights."""
    weights = np.rint(probs * WEIGHT_SCALE).astype(np.int64)
    weights[weights < 1] = 1  # every surviving token gets at least one bucket
    return weights


class CandidateSelector:
    """Base class for candidate-set strategies."""

    name: str = "base"

    def select(self, probs: np.ndarray, eligible: Sequence[int]) -> CandidateSet:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"strategy": self.name}


@dataclass
class FullVocabSelector(CandidateSelector):
    """Keep every eligible token with non-zero probability (lossless)."""

    name: str = "full"

    def select(self, probs: np.ndarray, eligible: Sequence[int]) -> CandidateSet:
        ids = [t for t in eligible if probs[t] > 0.0]
        if not ids:
            raise ValueError("model assigned zero probability to every eligible token")
        ids.sort()
        weights = _quantize(probs[ids])
        return CandidateSet(tuple(ids), tuple(int(w) for w in weights))


@dataclass
class TopNSelector(CandidateSelector):
    """Keep the ``n`` most probable eligible tokens (prob > 0)."""

    n: int = 5
    name: str = "topn"

    def select(self, probs: np.ndarray, eligible: Sequence[int]) -> CandidateSet:
        cand = [t for t in eligible if probs[t] > 0.0]
        if not cand:
            raise ValueError("model assigned zero probability to every eligible token")
        # Rank by probability (desc), break ties by token id for determinism.
        cand.sort(key=lambda t: (-probs[t], t))
        chosen = cand[: self.n]
        chosen.sort()  # lay out alphabetically / by token id
        weights = _quantize(probs[chosen])
        return CandidateSet(tuple(chosen), tuple(int(w) for w in weights))

    def describe(self) -> dict:
        return {"strategy": self.name, "n": self.n}


@dataclass
class AdaptiveSelector(CandidateSelector):
    """Grow the candidate set until ``coverage`` of the mass is captured."""

    max_n: int = 16
    coverage: float = 0.999
    name: str = "adaptive"

    def select(self, probs: np.ndarray, eligible: Sequence[int]) -> CandidateSet:
        cand = [t for t in eligible if probs[t] > 0.0]
        if not cand:
            raise ValueError("model assigned zero probability to every eligible token")
        cand.sort(key=lambda t: (-probs[t], t))
        chosen: list[int] = []
        acc = 0.0
        for t in cand:
            chosen.append(t)
            acc += float(probs[t])
            if acc >= self.coverage or len(chosen) >= self.max_n:
                break
        chosen.sort()
        weights = _quantize(probs[chosen])
        return CandidateSet(tuple(chosen), tuple(int(w) for w in weights))

    def describe(self) -> dict:
        return {"strategy": self.name, "max_n": self.max_n, "coverage": self.coverage}


def build_selector(mode: str, **kwargs) -> CandidateSelector:
    """Factory for the three built-in strategies."""
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
