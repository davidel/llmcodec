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


__all__ = ["NextTokenModel", "HFCausalLM", "load_hf_model"]
