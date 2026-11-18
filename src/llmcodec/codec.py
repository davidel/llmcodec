"""The arithmetic codec: encode a sentence to a real in ``[0, 1]`` and back.

Encoding
--------
Starting from the full interval ``[0, 1]`` and the prefix ``<bos>``:

1. Ask the model for the next-token distribution given the current prefix.
2. Ask the :class:`~llmcodec.candidates.CandidateSelector` for an ordered
   candidate alphabet (token ids + integer weights) derived *only* from that
   distribution.
3. Subdivide the current interval into one slice per candidate, in token-id
   (alphabetical) order, with widths proportional to the weights.
4. Let the true next token choose a slice; that slice becomes the new current
   interval.
5. Append the true token to the prefix and repeat, consuming the input
   characters, then finally consuming ``<eos>``.

The emitted code is the midpoint of the final interval.  Decoding reverses the
process: it replays the exact same candidate sets from the same prefixes, finds
the slice containing the (rescaled) code, emits the corresponding token, and
repeats until ``<eos>``.

Because the candidate sets depend only on the model distribution, the decoder
never needs to be told which tokens the encoder chose.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import torch

from .candidates import CandidateSelector, build_selector
from .interval import Interval
from .model import TinyLM
from .vocab import Vocab


class NotEncodableError(ValueError):
    """Raised when a token has no slice in the current candidate set.

    This can only happen with lossy selectors (``topn``/``adaptive``) when the
    true next token falls outside the selected candidates.  The ``full``
    selector never raises.
    """

    def __init__(self, token_id: int, position: int, candidate_ids: Sequence[int]):
        self.token_id = token_id
        self.position = position
        self.candidate_ids = tuple(candidate_ids)
        super().__init__(
            f"token {token_id} at position {position} has no slice among "
            f"candidates {tuple(candidate_ids)}"
        )


@dataclass
class Encoded:
    """The result of encoding a sentence."""

    value: Fraction
    text: str
    interval: Interval
    n_steps: int

    @property
    def as_float(self) -> float:
        return float(self.value)


class Codec:
    """Ties a model, a vocabulary and a candidate selector into a codec."""

    def __init__(
        self,
        model: TinyLM,
        vocab: Vocab,
        selector: CandidateSelector | None = None,
        *,
        emittable_ids: Sequence[int] | None = None,
        max_decode_len: int = 512,
        device: str = "cpu",
    ):
        self.model = model
        self.vocab = vocab
        self.selector = selector if selector is not None else build_selector("full")
        self.device = device
        self.max_decode_len = max_decode_len
        self.model.eval()

        if emittable_ids is None:
            # Everything the message alphabet can contain: the character
            # tokens plus <eos> and <unk>.  <pad>/<bos> are prompt-only.
            chars = {vocab.token_id(c) for c in vocab.itos if len(c) == 1}
            emittable_ids = sorted(chars | {vocab.eos_id, vocab.unk_id})
        self.emittable_ids = tuple(emittable_ids)
        self._emittable_set = set(self.emittable_ids)

    # -- helpers -------------------------------------------------------------
    def _probs(self, prefix_ids: Sequence[int]) -> np.ndarray:
        ids = torch.tensor(list(prefix_ids), dtype=torch.long, device=self.device)
        return self.model.next_token_probs(ids).cpu().numpy()

    def _candidates(self, probs: np.ndarray):
        return self.selector.select(probs, self.emittable_ids)

    # -- encoding ------------------------------------------------------------
    def encode(self, text: str) -> Encoded:
        """Encode ``text`` into a real number in ``[0, 1]``."""
        char_ids = self.vocab.encode(text)  # lower-cased, unknowns -> <unk>
        prefix = [self.vocab.bos_id]
        interval = Interval.unit()

        for position, token_id in enumerate(char_ids):
            cand = self._candidates(self._probs(prefix))
            idx = cand.index_of(token_id)
            if idx is None:
                raise NotEncodableError(token_id, position, cand.token_ids)
            interval = interval.subdivide(cand.weights, idx)
            prefix.append(token_id)

        # Consume the end-of-sentence marker.
        cand = self._candidates(self._probs(prefix))
        idx = cand.index_of(self.vocab.eos_id)
        if idx is None:
            raise NotEncodableError(self.vocab.eos_id, len(char_ids), cand.token_ids)
        interval = interval.subdivide(cand.weights, idx)

        return Encoded(
            value=interval.midpoint(),
            text=text,
            interval=interval,
            n_steps=len(char_ids) + 1,
        )

    def encode_float(self, text: str) -> float:
        return float(self.encode(text).value)

    # -- decoding ------------------------------------------------------------
    def decode(self, value: Fraction | float | int) -> str:
        """Decode a real number in ``[0, 1]`` back into a sentence."""
        code = value if isinstance(value, Fraction) else Fraction(value)
        if not (0 <= code <= 1):
            raise ValueError("code must lie in [0, 1]")

        prefix = [self.vocab.bos_id]
        interval = Interval.unit()
        out: list[int] = []

        for _ in range(self.max_decode_len):
            cand = self._candidates(self._probs(prefix))
            # The code stays in absolute [0, 1] coordinates: the interval simply
            # narrows around it, so ``find_slice`` always operates on the code's
            # current position.  This is exactly inverse to encoding.
            idx = interval.find_slice(cand.weights, code)
            token_id = cand.token_ids[idx]

            if token_id == self.vocab.eos_id:
                return self.vocab.decode(out, skip_special=False)

            interval = interval.subdivide(cand.weights, idx)
            out.append(token_id)
            prefix.append(token_id)

        raise RuntimeError(f"decoding exceeded max_decode_len={self.max_decode_len} without <eos>")

    # -- introspection -------------------------------------------------------
    def describe(self) -> dict:
        return {
            "selector": self.selector.describe(),
            "emittable_ids": list(self.emittable_ids),
            "vocab_size": self.vocab.vocab_size,
        }
