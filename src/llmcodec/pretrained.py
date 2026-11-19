"""Adapters for pre-trained language models as codec backends.

The arithmetic codec only needs one thing from its model: a next-token
probability distribution over the vocabulary.  Anything that provides
:meth:`next_token_probs` can drive the codec, which is exactly the interface
:class:`~llmcodec.model.TinyLM` already exposes.  This module adds an adapter
for Hugging Face causal language models so a pre-trained checkpoint (GPT-2,
Pythia, SmolLM, ...) can be used as the distribution source, with the matching
tokenizer supplying the vocabulary.

Determinism matters here: the codec requires that encoding and decoding see
*identical* probability vectors, or the intervals will not match.  The adapter
therefore always computes the softmax in float32 (regardless of the model's
inference dtype) and expects the same device/dtype/revision on both sides.
"""

from abc import abstractmethod
from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F


@runtime_checkable
class NextTokenModel(Protocol):
  """Structural interface the codec expects from a model.

  Implementations provide:

  ``next_token_probs(input_ids, amp_dtype=None)``
      Given a 1-D or 2-D ``torch.long`` tensor of token ids, return the
      softmax distribution for the last position as a float32 tensor whose
      last dimension is the vocabulary size.  Rows sum to one.
  ``vocab_size``
      The number of logits the model emits.
  ``device``
      The ``torch.device`` the model lives on.

  :class:`~llmcodec.model.TinyLM` already satisfies this protocol.
  """

  __slots__ = ()


class HFCausalLM:
  """Wraps a Hugging Face causal LM as a codec distribution source.

  The wrapped model is the raw checkpoint: no fine-tuning is done here, the
  adapter only normalizes inference so the codec sees a well-behaved
  distribution -- logits shifted to float32, softmaxed, and returned on CPU
  by the caller.

  Attributes:
    model: The underlying ``torch.nn.Module``.
    name: The model id or path it was loaded from.
    revision: The requested revision (defaults to the library's, ``main``).
    dtype: The inference dtype used when loading the weights.
    device: The device the model was moved to.
    context_len: Maximum prefix length fed to the model.
  """

  def __init__(self,
               model,
               name,
               revision=None,
               dtype="float32",
               device=None,
               context_len=None):
    """Wrap an already-constructed causal LM.

    Args:
      model: A Hugging Face causal LM (or any module whose forward returns
        logits with shape ``(..., vocab_size)``).
      name: The model id or path, kept for :meth:`describe`.
      revision: The revision the weights came from, if known.
      dtype: The inference dtype name (``float32``, ``float16``,
        ``bfloat16``); the softmax is still computed in float32.
      device: Device the model was moved to, if it was.
      context_len: Maximum prefix length per forward pass.  ``None`` uses
        the model's ``max_position_embeddings``.
    """
    self.model = model
    self.name = name
    self.revision = revision
    self.dtype = dtype
    self.device = torch.device(device) if device else next(model.parameters()).device
    self.context_len = context_len or int(
        getattr(model.config, "max_position_embeddings", 1024) or 1024)
    self.model.eval()

  @classmethod
  def load(cls,
           name_or_path,
           revision=None,
           dtype="float32",
           device=None,
           context_len=None):
    """Download (or load) a causal LM and wrap it.

    Args:
      name_or_path: A Hugging Face model id, e.g. ``"gpt2"``, or a local
        path to a checkpoint directory.
      revision: Optional git revision (branch, tag or commit hash) to pin.
      dtype: Inference dtype: ``"float32"``, ``"float16"``, ``"bfloat16"``
        or ``"auto"``.  Defaults to ``"float32"`` for reproducibility;
        the codec must run encode and decode with the same value.
      device: ``"auto"`` (default), ``"cpu"``, ``"cuda"`` or a torch device
        string.  ``"auto"`` resolves to CUDA when available.
      context_len: Optional override for the maximum prefix length.

    Returns:
      An :class:`HFCausalLM` in evaluation mode.

    Raises:
      ImportError: If the ``transformers`` package is not installed
        (install it with ``pip install "llmcodec[hf]"``).
    """
    try:
      from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover - depends on extras
      raise ImportError("Hugging Face models need the 'transformers' package; "
                        "install it with: pip install \"llmcodec[hf]\"") from exc
    from llmcodec.device import resolve_device

    resolved = resolve_device(device or "auto")
    torch_dtype = _resolve_dtype(dtype, resolved)
    kwargs = {"dtype": torch_dtype}
    if revision is not None:
      kwargs["revision"] = revision
    model = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs)
    model.to(resolved)
    return cls(model,
               name=name_or_path,
               revision=revision,
               dtype=dtype,
               device=str(resolved),
               context_len=context_len)

  @property
  def vocab_size(self):
    """Number of logits the model emits."""
    config = self.model.config
    return int(
        getattr(config, "vocab_size",
                self.model.get_output_embeddings().out_features))

  @torch.no_grad()
  def next_token_probs(self, input_ids, amp_dtype=None):
    """Return the softmax next-token distribution for the last position.

    Matches :meth:`llmcodec.model.TinyLM.next_token_probs`: the logits are
    computed in the model's inference dtype, but the softmax always runs in
    float32 so probabilities are stable across dtype choices.  Prefixes
    longer than :attr:`context_len` are truncated to their last
    ``context_len`` tokens (the model never sees more than it was trained
    for).

    Args:
      input_ids: 1-D or 2-D ``torch.long`` tensor of token ids.
      amp_dtype: Accepted for interface compatibility; ignored.

    Returns:
      A float32 tensor: ``(vocab_size,)`` for 1-D input, ``(B, vocab_size)``
      for 2-D input.  Rows sum to one.
    """
    del amp_dtype  # unused; the HF forward pass manages its own precision
    squeeze = input_ids.dim() == 1
    if squeeze:
      input_ids = input_ids.unsqueeze(0)
    if input_ids.shape[1] > self.context_len:
      input_ids = input_ids[:, -self.context_len:]
    input_ids = input_ids.to(self.device)
    logits = self.model(input_ids).logits[:, -1, :]
    probs = F.softmax(logits.float(), dim=-1)
    return probs.squeeze(0) if squeeze else probs

  def describe(self):
    """Return a JSON-friendly description of the adapter configuration."""
    return {
        "type": "hf",
        "name": self.name,
        "revision": self.revision,
        "dtype": self.dtype,
        "device": str(self.device),
        "context_len": self.context_len,
        "vocab_size": self.vocab_size,
    }


def _resolve_dtype(dtype, device):
  """Map a dtype name to a ``torch.dtype``, honouring ``"auto"``.

  Args:
    dtype: One of ``"float32"``, ``"float16"``, ``"bfloat16"``, ``"auto"``
      or an existing ``torch.dtype``.

  Returns:
    The corresponding ``torch.dtype``.

  Raises:
    ValueError: If ``dtype`` names an unknown dtype, or ``bfloat16`` is
      requested on hardware without bfloat16 support.
  """
  if isinstance(dtype, torch.dtype):
    return dtype
  mapping = {
      "float32": torch.float32,
      "fp32": torch.float32,
      "float16": torch.float16,
      "fp16": torch.float16,
      "bfloat16": torch.bfloat16,
      "bf16": torch.bfloat16,
  }
  if dtype == "auto":
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
      return torch.bfloat16
    return torch.float32
  if dtype not in mapping:
    raise ValueError(f"unknown dtype {dtype!r}; expected float32, float16, bfloat16 or "
                     "'auto'")
  torch_dtype = mapping[dtype]
  if torch_dtype is torch.bfloat16 and device.type == "cuda" and \
      not torch.cuda.is_bf16_supported():
    raise ValueError("bfloat16 is not supported on this GPU; "
                     "use float32 or float16")
  return torch_dtype


def load_hf_model(name_or_path, revision=None, dtype="float32", device=None):
  """Load a Hugging Face causal LM as a codec model backend.

  Convenience wrapper around :meth:`HFCausalLM.load` without the
  ``context_len`` override.

  Args:
    name_or_path: Hugging Face model id or local checkpoint path.
    revision: Optional revision to pin.
    dtype: Inference dtype (see :meth:`HFCausalLM.load`).
    device: ``"auto"``, ``"cpu"``, ``"cuda"`` or a torch device string.

  Returns:
    An :class:`HFCausalLM` ready to be passed to
    :class:`~llmcodec.codec.Codec`.
  """
  return HFCausalLM.load(name_or_path, revision=revision, dtype=dtype, device=device)


@runtime_checkable
class TokenCodec(Protocol):
  """Structural interface the codec expects from a tokenizer.

  Implementations provide:

  ``encode(text, add_bos=False, add_eos=False)``
      Map text to a list of token ids.
  ``decode(ids, skip_special=False)``
      Map ids back to text.
  ``vocab_size``, ``bos_id``, ``eos_id``, ``unk_id``
      Sizes and special ids; ``vocab_size`` must match the model's.
  ``itos``
      Optional id-to-token table used only for error messages.

  :class:`~llmcodec.vocab.Vocab` already satisfies this protocol.
  """

  @abstractmethod
  def encode(self, text, add_bos=False, add_eos=False):
    ...

  @abstractmethod
  def decode(self, ids, skip_special=False):
    ...

  @property
  @abstractmethod
  def vocab_size(self):
    ...

  @property
  @abstractmethod
  def bos_id(self):
    ...

  @property
  @abstractmethod
  def eos_id(self):
    ...

  @property
  @abstractmethod
  def unk_id(self):
    ...


class HFTokenizer:
  """Wraps a Hugging Face ``PreTrainedTokenizerFast`` as a codec tokenizer.

  BOS/EOS management stays with the codec exactly as in the character-level
  path, so the wrapper requests plain encodings and clears any special ids
  the tokenizer would have added itself.  Unk is typically ``None`` for
  byte-level BPE tokenizers; the codec's unknown-input behaviour then comes
  from the tokenizer's own byte fallback instead.

  Attributes:
    tokenizer: The wrapped fast tokenizer.
    name: The tokenizer id or path it was loaded from.
    revision: The requested revision, if any.
    add_bos_token: Whether the underlying tokenizer adds BOS by default.
  """

  def __init__(self, tokenizer, name=None, revision=None, device=None):
    """Wrap an already-loaded tokenizer.

    Args:
      tokenizer: A ``PreTrainedTokenizerFast`` (or compatible fast tokenizer).
      name: The tokenizer id or path, kept for :meth:`describe`.
      revision: The revision the tokenizer came from, if known.
      device: Unused; accepted so call sites can pass the codec's device
        uniformly.
    """
    del device  # tokenization is device-independent
    self.tokenizer = tokenizer
    self.name = name
    self.revision = revision
    self.add_bos_token = bool(getattr(tokenizer, "add_bos_token", False)) or \
        getattr(tokenizer, "bos_token_id", None) in (
            tokenizer.model_input_names or [])

  def __getattr__(self, name):
    """Pass attribute access through to the wrapped tokenizer.

    This exposes ``bos_token_id``/``eos_token_id``/``unk_token_id`` and any
    other attribute of the underlying tokenizer, which is how the codec
    reaches ``bos_id``/``eos_id``/``unk_id``/``vocab_size`` without a forest
    of forwarding properties.
    """
    return getattr(self.tokenizer, name)

  def __getitem__(self, token_id):
    """Expose ``itos``-style indexed access for error messages."""
    return self.tokenizer.convert_ids_to_tokens(int(token_id))

  def __iter__(self):
    """Allow ``for c in vocab.itos``-style iteration for the char heuristic."""
    return iter(self.tokenizer.convert_ids_to_tokens(range(len(self.tokenizer))))

  def __len__(self):
    """Size of the vocabulary, mirroring ``Vocab.__len__``."""
    return self.vocab_size

  @classmethod
  def load(cls, name_or_path, revision=None, device=None):
    """Load a tokenizer from the hub or a local path.

    Args:
      name_or_path: A tokenizer id, e.g. ``"gpt2"``, or a local path.
      revision: Optional revision to pin.
      device: Unused; accepted for call-site uniformity.

    Returns:
      An :class:`HFTokenizer`.

    Raises:
      ImportError: If the ``transformers`` package is not installed
        (install it with ``pip install "llmcodec[hf]"``).
    """
    try:
      from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on extras
      raise ImportError("Hugging Face tokenizers need the 'transformers' package; "
                        "install it with: pip install \"llmcodec[hf]\"") from exc
    kwargs = {}
    if revision is not None:
      kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    return cls(tokenizer, name=name_or_path, revision=revision)

  @property
  def vocab_size(self):
    """Size of the token id space the model's logits must cover.

    ``len(tokenizer)`` rather than ``tokenizer.vocab_size``: the latter can
    exclude *added* special tokens, whose ids are still valid inputs to the
    model and must stay indexable into its distribution.
    """
    return len(self.tokenizer)

  @property
  def bos_id(self):
    """The BOS id, or ``None`` when the tokenizer has none.

    The codec requires a BOS to seed its prefix, so loading a tokenizer
    without BOS fails at :meth:`load` time rather than at encode time.
    """
    return self.tokenizer.bos_token_id

  @property
  def eos_id(self):
    """The EOS id, or ``None`` when the tokenizer has none."""
    return self.tokenizer.eos_token_id

  @property
  def unk_id(self):
    """The unk id, or ``None`` when the tokenizer has unknown tokens."""
    return self.tokenizer.unk_token_id

  @property
  def pad_id(self):
    """The pad id, or ``None`` when the tokenizer has no pad token."""
    return self.tokenizer.pad_token_id

  @property
  def itos(self):
    """Id-to-token strings, for error messages and the char heuristic.

    Truncated to the tokenizer's true vocabulary; padded re-exported
    tokenizers can report a padded ``len()`` greater than the real vocab.
    """
    size = self.vocab_size
    return self.tokenizer.convert_ids_to_tokens(range(size))

  def encode(self, text, add_bos=False, add_eos=False):
    """Map ``text`` to a list of token ids.

    Args:
      text: The text to tokenize.
      add_bos: Prepend the BOS id.  Unusable when the tokenizer has no BOS
        (e.g. GPT-2); raise instead of silently encoding a wrong prefix.
      add_eos: Append the EOS id.  Same constraint as ``add_bos``.

    Returns:
      The list of token ids, with any tokenizer-added special ids removed.
    """
    ids = self.tokenizer.encode(text, add_special_tokens=False)
    if add_bos:
      if self.bos_id is None:
        raise ValueError("tokenizer has no bos token; the codec cannot seed its prefix")
      ids = [self.bos_id] + ids
    if add_eos:
      if self.eos_id is None:
        raise ValueError("tokenizer has no eos token; cannot append one")
      ids = ids + [self.eos_id]
    return ids

  def decode(self, ids, skip_special=False):
    """Map ids back to text via the tokenizer's own decoder.

    Args:
      ids: Token ids.
      skip_special: Drop special tokens before decoding.

    Returns:
      The decoded string.
    """
    if skip_special:
      special = {i for i in (self.bos_id, self.eos_id, self.unk_id) if i is not None}
      ids = [i for i in ids if i not in special]
    return self.tokenizer.decode(list(ids), skip_special_tokens=False)

  def describe(self):
    """Return a JSON-friendly description of the tokenizer configuration."""
    return {
        "type": "hf",
        "name": self.name,
        "revision": self.revision,
        "vocab_size": self.vocab_size,
    }


__all__ = ["NextTokenModel", "HFCausalLM", "TokenCodec", "HFTokenizer", "load_hf_model"]
