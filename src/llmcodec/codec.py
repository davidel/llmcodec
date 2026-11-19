"""The arithmetic codec: encode a sentence to a real in ``[0, 1]`` and back.

The codec treats a language model as a source model for arithmetic coding.  The
model supplies a next-token probability distribution at every step; the coder
turns a sentence into the single real number that uniquely identifies the
sequence of choices the sentence makes.

Encoding
--------
Start with the interval ``[0, 1]`` and the prefix ``<bos>``.  For each token to
be encoded (the input characters in order, then a final ``<eos>``):

1. Ask the model for the next-token distribution given the current prefix.
2. Ask the :class:`~llmcodec.candidates.CandidateSelector` for an ordered
   candidate alphabet (token ids plus integer weights) derived *only* from that
   distribution.
3. Subdivide the current interval into one slice per candidate, in token-id
   (alphabetical) order, with widths proportional to the weights.
4. Let the true next token choose a slice; that slice becomes the new interval.
5. Append the true token to the prefix and repeat.

The emitted code is the midpoint of the final interval.  Every value strictly
inside the interval decodes to the same sentence, so the midpoint is a robust
representative.

Decoding
--------
Given a real ``r`` in ``[0, 1]``, replay the exact same candidate sets from the
same prefixes.  At each step, find the slice of the current interval that
contains ``r``, emit the corresponding token, narrow the interval to that slice,
and continue.  Decoding stops at ``<eos>``.  The code is never rescaled: it
stays in absolute ``[0, 1]`` coordinates while the interval shrinks around it,
which is precisely the inverse of the encoding step.

Because the candidate set is a deterministic function of the model
distribution, the decoder never needs to be told which tokens the encoder chose.
"""

from collections import namedtuple
from fractions import Fraction

import numpy as np
import torch

from .candidates import WEIGHT_SCALE, CandidateSet, build_selector
from .device import resolve_device
from .interval import BitCode, Interval


class NotEncodableError(ValueError):
  """Raised when a token has no slice in the current candidate set.

  This can only happen with lossy selectors (``topn``/``adaptive``) when the
  true next token falls outside the selected candidates.  The ``full`` selector
  never raises because it keeps every token the model can emit.

  Attributes:
    token_id: The token that could not be placed.
    position: Its position in the input (character index, or the ``<eos>`` step).
    candidate_ids: The candidate token ids that were available instead.
    token: The human-readable text of the offending token, when available.
    candidates: The human-readable text of the candidate tokens, when available.
  """

  def __init__(self, token_id, position, candidate_ids, token=None, candidates=None):
    """Record the offending token and the available candidates.

    Args:
      token_id: The id of the token that could not be placed.
      position: Its position in the input.
      candidate_ids: The candidate token ids that were available instead.
      token: Optional human-readable text of the offending token.  The codec
        passes this so the message reads ``'q'`` rather than a bare id.
      candidates: Optional human-readable text of the candidate tokens.
    """
    self.token_id = token_id
    self.position = position
    self.candidate_ids = tuple(candidate_ids)
    self.token = token
    self.candidates = tuple(candidates) if candidates is not None else None
    token_repr = f"{token!r} (id {token_id})" if token is not None else token_id
    if self.candidates is not None:
      candidate_repr = list(self.candidates)
    else:
      candidate_repr = list(self.candidate_ids)
    super().__init__(f"token {token_repr} at position {position} has no slice "
                     f"among candidates {candidate_repr}")


class Encoded(namedtuple("Encoded", "value text interval n_steps")):
  """The result of encoding a sentence.

  Attributes:
    value: The code, as an exact :class:`~fractions.Fraction` in ``[0, 1]``.
    text: The original (un-normalized) input text.
    interval: The final interval; any point inside it decodes to ``text``.
    n_steps: The number of coding steps, including the final ``<eos>`` step.
  """

  __slots__ = ()

  @property
  def as_float(self):
    """The code as a Python ``float`` (lossy -- prefer :attr:`value`)."""
    return float(self.value)


class Codec:
  """Ties a model, a vocabulary and a candidate selector into a codec.

  Attributes:
    model: The language model supplying next-token distributions.
    vocab: The vocabulary used to map text to token ids.
    selector: The strategy that turns a distribution into a candidate set.
    device: The torch device used for model inference.
    max_decode_len: Safety cap on the number of decoding steps.
    emittable_ids: The token ids the codec is allowed to place.
  """

  def __init__(self,
               model,
               vocab,
               selector=None,
               *,
               emittable_ids=None,
               max_decode_len=512,
               device="auto",
               amp_dtype=None):
    """Initialize the codec.

    Args:
      model: The language model.  It is moved to the resolved device and put
        into evaluation mode.
      vocab: The vocabulary.
      selector: The candidate-selection strategy.  Defaults to the lossless
        ``full`` selector.
      emittable_ids: The token ids the codec may place.  Defaults to the
        message characters plus ``<eos>`` and ``<unk>``; ``<pad>`` and ``<bos>``
        are prompt-only and therefore excluded.
      max_decode_len: Maximum number of tokens to decode before giving up.
      device: The torch device for inference.  ``"auto"`` (the default) uses
        CUDA when available and the CPU otherwise.
      amp_dtype: Optional autocast dtype for CUDA inference.  Defaults to
        ``float16`` on CUDA and ``bfloat16`` elsewhere.
    """
    self.model = model
    self.vocab = vocab
    self.selector = selector if selector is not None else build_selector("full")
    self.device = resolve_device(device)
    self.amp_dtype = amp_dtype
    self.max_decode_len = max_decode_len
    if isinstance(self.model, torch.nn.Module):
      self.model.to(self.device)
      self.model.eval()
    else:
      # The wrapper manages its own device and evaluation mode (for example
      # llmcodec.pretrained.HFCausalLM); do not assume an nn.Module.
      if getattr(self.model, "device", None) is not None:
        self.device = torch.device(self.model.device)

    if emittable_ids is None:
      token_id = getattr(vocab, "token_id", None)
      if token_id is not None:
        # Character-level vocabularies can emit every single-char token plus
        # <eos> and <unk>.  <pad>/<bos> are prompt-only and never emitted.
        chars = {token_id(c) for c in vocab.itos if len(c) == 1}
        emittable_ids = sorted(chars | {vocab.eos_id, vocab.unk_id})
      else:
        # Subword vocabularies have no per-character table: every token is
        # emittable except the prompt-only specials (<bos>/<pad>).
        excluded = {
            i for i in (vocab.bos_id, getattr(vocab, "pad_id", None)) if i is not None
        }
        emittable_ids = sorted(set(range(vocab.vocab_size)) - excluded)
    self.emittable_ids = tuple(emittable_ids)

    model_size = getattr(model, "vocab_size", None)
    if model_size is None:
      model_size = getattr(getattr(model, "config", None), "vocab_size", None)
    if model_size is not None and model_size != vocab.vocab_size:
      raise ValueError(f"model vocabulary size {model_size} does not match tokenizer "
                       f"vocabulary size {vocab.vocab_size}: the codec would index the "
                       "distribution with ids from the wrong table")

  def _bos_seed(self):
    """Return the id that seeds the coding prefix.

    Normally ``<bos>``.  For tokenizers without a BOS token (GPT-2 style
    byte-level BPE) the standard convention applies: EOS doubles as the
    sequence-start token, so it seeds the prefix instead.  The seed is part
    of the coded format: encode and decode must agree on it, which they do
    automatically because both derive it from the same tokenizer.

    Raises:
      ValueError: If the tokenizer has neither BOS nor EOS, so no seed
        exists at all.
    """
    seed = self.vocab.bos_id
    if seed is None:
      seed = self.vocab.eos_id
    if seed is None:
      raise ValueError(
          "tokenizer has neither <bos> nor <eos>; the codec cannot seed its "
          "prefix -- pick a tokenizer that defines at least one of them")
    return seed

  def _probs(self, prefix_ids):
    """Return the next-token distribution for ``prefix_ids`` as a numpy array.

    Args:
      prefix_ids: The token ids seen so far.

    Returns:
      A ``(vocab_size,)`` probability vector.
    """
    ids = torch.tensor(list(prefix_ids), dtype=torch.long, device=self.device)
    probs = self.model.next_token_probs(ids, amp_dtype=self.amp_dtype)
    return probs.cpu().numpy()

  def _candidates(self, probs):
    """Return the candidate set for a distribution.

    The end-of-sequence token is always part of the set: if the selector
    left it out (a strong model can assign the sentence-final ``<eos>``
    less probability than a selector's coverage cutoff), it is appended
    with its true quantized probability.  Without this guarantee the codec
    could fail mid-decode, unable to place ``<eos>`` no matter how the
    selector is configured.

    The guarantee is a no-op whenever EOS is already covered, so sets that
    the plain selectors produce are never modified.

    Args:
      probs: The next-token probability vector.

    Returns:
      The :class:`~llmcodec.candidates.CandidateSet` chosen by the selector,
      with EOS guaranteed to be part of it.

    Raises:
      ValueError: If the selector omitted EOS and it has no usable
        probability in the distribution.
    """
    candidates = self.selector.select(probs, self.emittable_ids)
    eos_id = self.vocab.eos_id
    if eos_id is None or candidates.index_of(eos_id) is not None:
      return candidates
    weight = int(np.rint(probs[eos_id] * WEIGHT_SCALE))
    if weight < 1:
      # A selector that drops EOS has implicitly claimed the probability
      # mass; take the smallest share that keeps the interval arithmetic
      # well defined, so termination stays possible even for p(EOS) = 0.
      weight = 1
    token_ids = list(candidates.token_ids) + [eos_id]
    weights = list(candidates.weights) + [weight]
    return CandidateSet(token_ids, weights)

  def _not_encodable(self, token_id, position, candidate_ids):
    """Build a :class:`NotEncodableError` with human-readable token text.

    Args:
      token_id: The id of the token that could not be placed.
      position: Its position in the input.
      candidate_ids: The candidate token ids that were available instead.

    Returns:
      A :class:`NotEncodableError` whose message names the tokens rather than
      only their ids.
    """
    return NotEncodableError(
        token_id,
        position,
        candidate_ids,
        token=self.vocab.itos[token_id],
        candidates=[self.vocab.itos[i] for i in candidate_ids],
    )

  def encode(self, text):
    """Encode ``text`` into a real number in ``[0, 1]``.

    The text is lower-cased and any character outside the message alphabet
    becomes ``<unk>``.  The characters are consumed in order, followed by a
    final ``<eos>``.

    Args:
      text: The sentence to encode.

    Returns:
      An :class:`Encoded` holding the code, the interval, and the step count.

    Raises:
      NotEncodableError: If a token has no slice under the current selector.
    """
    char_ids = self.vocab.encode(text)
    prefix = [self._bos_seed()]
    interval = Interval.unit()

    for position, token_id in enumerate(char_ids):
      cand = self._candidates(self._probs(prefix))
      idx = cand.index_of(token_id)
      if idx is None:
        raise self._not_encodable(token_id, position, cand.token_ids)
      interval = interval.subdivide(cand.weights, idx)
      prefix.append(token_id)

    # Consume the end-of-sentence marker to terminate the sequence.
    cand = self._candidates(self._probs(prefix))
    idx = cand.index_of(self.vocab.eos_id)
    if idx is None:
      raise self._not_encodable(self.vocab.eos_id, len(char_ids), cand.token_ids)
    interval = interval.subdivide(cand.weights, idx)

    return Encoded(
        value=interval.midpoint(),
        text=text,
        interval=interval,
        n_steps=len(char_ids) + 1,
    )

  def encode_float(self, text):
    """Encode ``text`` and return the code as a ``float``.

    Args:
      text: The sentence to encode.

    Returns:
      The code as a Python ``float`` in ``[0, 1]``.
    """
    return float(self.encode(text).value)

  def encode_bits(self, text):
    """Encode ``text`` as a :class:`~llmcodec.interval.BitCode`.

    This is the binary form of :meth:`encode`.  The final interval is quantized
    onto the coarsest dyadic cell fully contained in it, and the cell is
    returned as ``(n, k)``: ``n`` bits whose value is ``k / 2**n``.  Any point of
    the cell -- in particular the code's own value -- lies inside the interval,
    so the bits decode back to the same sentence.

    The code is the shortest containment-safe form of the interval: its length
    is within two bits of the information-theoretic optimum ``ceil(-log2 w)``.

    Args:
      text: The sentence to encode.

    Returns:
      The :class:`~llmcodec.interval.BitCode` for ``text``.

    Raises:
      NotEncodableError: If a token has no slice under the current selector.
    """
    return self.encode(text).interval.dyadic()

  def decode_bits(self, code):
    """Decode a :class:`~llmcodec.interval.BitCode` back into a sentence.

    This is the inverse of :meth:`encode_bits`: the code names the dyadic cell
    ``[k / 2**n, (k + 1) / 2**n]``, and its lower edge ``k / 2**n`` is a point
    inside the interval the encoder produced, so :meth:`decode` recovers the
    sentence.

    Args:
      code: The :class:`~llmcodec.interval.BitCode` to decode.

    Returns:
      The decoded sentence.

    Raises:
      ValueError: If ``code`` is not a :class:`~llmcodec.interval.BitCode`.
      RuntimeError: If decoding exceeds ``max_decode_len`` without seeing
        ``<eos>`` (which indicates a code that does not correspond to any
        encoded sentence).
    """
    if not isinstance(code, BitCode):
      raise ValueError("decode_bits expects a BitCode")
    return self.decode(code.value)

  def decode(self, value):
    """Decode a real number in ``[0, 1]`` back into a sentence.

    Args:
      value: The code, as a :class:`~fractions.Fraction`, ``float`` or ``int``.

    Returns:
      The decoded sentence.

    Raises:
      ValueError: If ``value`` lies outside ``[0, 1]``.
      RuntimeError: If decoding exceeds ``max_decode_len`` without seeing
        ``<eos>`` (which indicates a code that does not correspond to any
        encoded sentence).
    """
    code = value if isinstance(value, Fraction) else Fraction(value)
    if not (0 <= code <= 1):
      raise ValueError("code must lie in [0, 1]")

    prefix = [self._bos_seed()]
    interval = Interval.unit()
    out = []

    for _ in range(self.max_decode_len):
      cand = self._candidates(self._probs(prefix))
      # The code stays in absolute [0, 1] coordinates: the interval simply
      # narrows around it, so find_slice always operates on the code's current
      # position.  This is exactly inverse to encoding.
      idx = interval.find_slice(cand.weights, code)
      token_id = cand.token_ids[idx]

      if token_id == self.vocab.eos_id:
        return self.vocab.decode(out, skip_special=False)

      interval = interval.subdivide(cand.weights, idx)
      out.append(token_id)
      prefix.append(token_id)

    raise RuntimeError(
        f"decoding exceeded max_decode_len={self.max_decode_len} without <eos>")

  def describe(self):
    """Return a JSON-friendly description of the codec configuration."""
    return {
        "selector": self.selector.describe(),
        "emittable_ids": list(self.emittable_ids),
        "vocab_size": self.vocab.vocab_size,
        "device": str(self.device),
    }
