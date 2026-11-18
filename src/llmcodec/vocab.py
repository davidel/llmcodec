"""Vocabulary definition and text <-> token-id mapping.

The vocabulary is intentionally tiny so that the arithmetic coder has a small,
well-conditioned alphabet to work with:

* 26 lowercase letters ``a-z``
* 10 digits ``0-9``
* the space character
* ``.`` and ``,``

Every other character is mapped to the ``<unk>`` special token.  Text is always
lower-cased before it is tokenized.

In addition to the character tokens we reserve a handful of special tokens:
``<pad>`` (used to pad batched training sequences), ``<bos>`` (beginning of
sentence), ``<eos>`` (end of sentence) and ``<unk>``.
"""

from __future__ import annotations

import string
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

# --- the character alphabet -------------------------------------------------

CHARS: str = string.ascii_lowercase + string.digits + " .,"

# --- special tokens ---------------------------------------------------------

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS: tuple[str, ...] = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN)


@dataclass(frozen=True)
class Vocab:
    """A tiny, deterministic vocabulary.

    The token id assignment is fixed: the special tokens occupy the first four
    ids (in :data:`SPECIAL_TOKENS` order) followed by :data:`CHARS` in their
    natural order.  This makes checkpoints portable and reproducible.
    """

    itos: tuple[str, ...]

    # -- construction --------------------------------------------------------
    @classmethod
    def build(cls, chars: str = CHARS, specials: Sequence[str] = SPECIAL_TOKENS) -> Vocab:
        itos = tuple(specials) + tuple(chars)
        if len(set(itos)) != len(itos):
            raise ValueError("duplicate tokens in vocabulary")
        return cls(itos=itos)

    # -- basic introspection -------------------------------------------------
    def __len__(self) -> int:
        return len(self.itos)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def stoi(self) -> dict[str, int]:
        return {tok: i for i, tok in enumerate(self.itos)}

    def __contains__(self, token: str) -> bool:
        return token in self.itos

    def token_id(self, token: str) -> int:
        try:
            return self.stoi[token]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"unknown token {token!r}") from exc

    @property
    def pad_id(self) -> int:
        return self.token_id(PAD_TOKEN)

    @property
    def bos_id(self) -> int:
        return self.token_id(BOS_TOKEN)

    @property
    def eos_id(self) -> int:
        return self.token_id(EOS_TOKEN)

    @property
    def unk_id(self) -> int:
        return self.token_id(UNK_TOKEN)

    # -- conversion ----------------------------------------------------------
    def normalize(self, text: str) -> str:
        """Lower-case ``text`` (the only preprocessing we apply)."""
        return text.lower()

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Map ``text`` to a list of token ids.

        Unknown characters become :data:`UNK_TOKEN`.  ``<bos>``/``<eos>`` are
        *not* added by default; pass the flags to wrap the sequence.
        """
        stoi = self.stoi
        unk = self.unk_id
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for ch in self.normalize(text):
            ids.append(stoi.get(ch, unk))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], *, skip_special: bool = False) -> str:
        """Map token ids back to a string.

        Special tokens are rendered verbatim (e.g. ``<bos>``) unless
        ``skip_special`` is set, in which case they are dropped.
        """
        special = set(SPECIAL_TOKENS)
        out: list[str] = []
        for i in ids:
            tok = self.itos[i]
            if skip_special and tok in special:
                continue
            out.append(tok)
        return "".join(out)

    # -- (de)serialization ---------------------------------------------------
    def to_dict(self) -> dict:
        return {"itos": list(self.itos)}

    @classmethod
    def from_dict(cls, data: dict) -> Vocab:
        return cls(itos=tuple(data["itos"]))


DEFAULT_VOCAB = Vocab.build()
