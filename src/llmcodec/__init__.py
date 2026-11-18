"""llmcodec -- encode sentences as real numbers in ``[0, 1]`` with a small LM.

The package trains a tiny character-level transformer on a corpus (TinyStories
by default) and uses its next-token distribution to drive an arithmetic coder:
a sentence becomes a single real number, and that number decodes back to the
sentence.

Public API
----------
``Vocab`` / ``DEFAULT_VOCAB``
    The fixed 42-token vocabulary.
``TinyLM`` / ``ModelConfig``
    6-layer, 512-wide causal transformer.
``Codec`` / ``Encoded`` / ``NotEncodableError``
    The encoder/decoder.
``build_selector`` and the selector classes
    Pluggable candidate-set strategies (``full``, ``topn``, ``adaptive``).
"""

from .candidates import (
    AdaptiveSelector,
    CandidateSelector,
    CandidateSet,
    FullVocabSelector,
    TopNSelector,
    build_selector,
)
from .codec import Codec, Encoded, NotEncodableError
from .interval import Interval
from .model import ModelConfig, TinyLM
from .vocab import DEFAULT_VOCAB, Vocab

__all__ = [
    "AdaptiveSelector",
    "CandidateSet",
    "CandidateSelector",
    "Codec",
    "DEFAULT_VOCAB",
    "Encoded",
    "FullVocabSelector",
    "Interval",
    "ModelConfig",
    "NotEncodableError",
    "TinyLM",
    "TopNSelector",
    "Vocab",
    "build_selector",
]

__version__ = "0.1.0"
