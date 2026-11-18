"""Vocabulary definition and text-to-token-id mapping.

The vocabulary is deliberately tiny.  A small, well-conditioned alphabet keeps
the arithmetic coder simple and makes the model easy to train to a useful level
of accuracy on a modest corpus.  It consists of three groups of tokens:

Message characters (39 tokens)
    The 26 lowercase letters ``a``-``z``, the 10 digits ``0``-``9``, the space
    character, a period ``.`` and a comma ``,``.  These are the only characters
    the codec can represent faithfully.

Special tokens (4 tokens)
    ``<pad>`` (padding for batched training sequences), ``<bos>`` (beginning of
    sentence), ``<eos>`` (end of sentence) and ``<unk>`` (any character outside
    the message alphabet).

All text is lower-cased before tokenization, and any character that is not a
message character becomes ``<unk>``.  This means the mapping
``text -> ids -> text`` is only the identity for already-lower-case text drawn
from the message alphabet.

The token-id assignment is fixed and deterministic: the four special tokens
occupy ids ``0..3`` in :data:`SPECIAL_TOKENS` order, followed by the message
characters in their natural order.  Fixing the layout makes checkpoints
portable across machines and runs.
"""

import string
from collections import namedtuple

# The message alphabet, in the exact order used to assign token ids.  Keeping
# this as a module-level string makes it easy to inspect and to reuse in tests.
CHARS = string.ascii_lowercase + string.digits + " .,"

# The special tokens, in the exact order used to assign token ids.
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN)


class Vocab(namedtuple("Vocab", "itos")):
  """A tiny, deterministic vocabulary of characters and special tokens.

  The vocabulary is immutable and compares by value, so two ``Vocab`` objects
  with the same ``itos`` are interchangeable.  Use :meth:`build` to construct
  the standard vocabulary and :meth:`from_dict` to reload one from a checkpoint.

  Attributes:
    itos: The token strings indexed by token id (id -> string).  Its length is
      the vocabulary size.
  """

  __slots__ = ()

  @classmethod
  def build(cls, chars=CHARS, specials=SPECIAL_TOKENS):
    """Build a vocabulary from a character string and a list of specials.

    The specials are placed first (in the given order) and the characters after
    them, so the resulting id layout is ``specials + chars``.

    Args:
      chars: The message characters, one per token, in id order.
      specials: The special tokens, in id order.

    Returns:
      A new :class:`Vocab`.

    Raises:
      ValueError: If any token string occurs more than once.

    Example:
      >>> vocab = Vocab.build()
      >>> vocab.vocab_size
      43
      >>> vocab.token_id("a")
      4
      >>> vocab.bos_id
      1
    """
    itos = tuple(specials) + tuple(chars)
    if len(set(itos)) != len(itos):
      raise ValueError("duplicate tokens in vocabulary")
    return cls(itos=itos)

  def __len__(self):
    """Return the number of tokens in the vocabulary."""
    return len(self.itos)

  @property
  def vocab_size(self):
    """The number of tokens in the vocabulary."""
    return len(self.itos)

  @property
  def stoi(self):
    """The string-to-id mapping (rebuilt on each access; the vocab is tiny)."""
    return {tok: i for i, tok in enumerate(self.itos)}

  def __contains__(self, token):
    """Return whether ``token`` is part of the vocabulary."""
    return token in self.itos

  def token_id(self, token):
    """Return the id of ``token``.

    Args:
      token: The token string, e.g. ``"a"`` or ``"<eos>"``.

    Returns:
      The corresponding token id.

    Raises:
      KeyError: If ``token`` is not in the vocabulary.
    """
    try:
      return self.stoi[token]
    except KeyError as exc:  # pragma: no cover - defensive
      raise KeyError(f"unknown token {token!r}") from exc

  @property
  def pad_id(self):
    """The id of the ``<pad>`` token."""
    return self.token_id(PAD_TOKEN)

  @property
  def bos_id(self):
    """The id of the ``<bos>`` token."""
    return self.token_id(BOS_TOKEN)

  @property
  def eos_id(self):
    """The id of the ``<eos>`` token."""
    return self.token_id(EOS_TOKEN)

  @property
  def unk_id(self):
    """The id of the ``<unk>`` token."""
    return self.token_id(UNK_TOKEN)

  def normalize(self, text):
    """Apply the codec's only text preprocessing step.

    Args:
      text: Arbitrary input text.

    Returns:
      ``text`` lower-cased.
    """
    return text.lower()

  def encode(self, text, add_bos=False, add_eos=False):
    """Map ``text`` to a list of token ids.

    Unknown characters are mapped to :data:`UNK_TOKEN`.  By default no special
    tokens are added; pass ``add_bos``/``add_eos`` to wrap the sequence, which is
    what the training data pipeline does.

    Args:
      text: The text to encode (lower-cased internally).
      add_bos: If true, prepend the ``<bos>`` id.
      add_eos: If true, append the ``<eos>`` id.

    Returns:
      The list of token ids.

    Example:
      >>> Vocab.build().encode("cat")
      [6, 4, 23]
      >>> Vocab.build().encode("cat", add_bos=True, add_eos=True)
      [1, 6, 4, 23, 2]
    """
    stoi = self.stoi
    unk = self.unk_id
    ids = []
    if add_bos:
      ids.append(self.bos_id)
    for ch in self.normalize(text):
      ids.append(stoi.get(ch, unk))
    if add_eos:
      ids.append(self.eos_id)
    return ids

  def decode(self, ids, skip_special=False):
    """Map token ids back to a string.

    Special tokens are rendered verbatim (for example ``<bos>``) unless
    ``skip_special`` is set, in which case they are dropped.  Note that this is
    the plain inverse of :meth:`encode` and does not know about ``<unk>``
    collapsing: several distinct inputs can map to the same output.

    Args:
      ids: The token ids to decode.
      skip_special: If true, omit special tokens from the output.

    Returns:
      The decoded string.
    """
    special = set(SPECIAL_TOKENS)
    out = []
    for i in ids:
      tok = self.itos[i]
      if skip_special and tok in special:
        continue
      out.append(tok)
    return "".join(out)

  def to_dict(self):
    """Serialize the vocabulary to a JSON-friendly dictionary."""
    return {"itos": list(self.itos)}

  @classmethod
  def from_dict(cls, data):
    """Rebuild a vocabulary from the output of :meth:`to_dict`."""
    return cls(itos=tuple(data["itos"]))


# The vocabulary used throughout the package unless a caller supplies another.
DEFAULT_VOCAB = Vocab.build()
