"""llmcodec -- encode sentences as real numbers in ``[0, 1]`` with a small LM.

The package trains a tiny character-level transformer on a corpus (TinyStories
by default) and uses its next-token distribution to drive an arithmetic coder: a
sentence becomes a single real number, and that number decodes back to the
sentence.

Public API:
  :class:`~llmcodec.vocab.Vocab` / :data:`~llmcodec.vocab.DEFAULT_VOCAB`
    The fixed 43-token vocabulary.
  :class:`~llmcodec.model.TinyLM` / :class:`~llmcodec.model.ModelConfig`
    The 6-layer, 512-wide causal transformer.
  :class:`~llmcodec.codec.Codec` / :class:`~llmcodec.codec.Encoded` /
  :class:`~llmcodec.codec.NotEncodableError`
    The encoder/decoder.
  :func:`~llmcodec.candidates.build_selector` and the selector classes
    Pluggable candidate-set strategies (``full``, ``topn``, ``adaptive``).
  :mod:`llmcodec.device`
    GPU/CPU device selection and mixed-precision helpers.
"""

from .candidates import (
    WEIGHT_SCALE,
    AdaptiveSelector,
    CandidateSelector,
    CandidateSet,
    FullVocabSelector,
    TopNSelector,
    build_selector,
)
from .codec import Codec, Encoded, NotEncodableError
from .device import (
    autocast_context,
    make_grad_scaler,
    resolve_amp_dtype,
    resolve_device,
)
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
    "WEIGHT_SCALE",
    "autocast_context",
    "build_selector",
    "make_grad_scaler",
    "resolve_amp_dtype",
    "resolve_device",
]

__version__ = "0.1.0"
