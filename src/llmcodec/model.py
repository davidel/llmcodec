"""A small decoder-only transformer language model.

The architecture is deliberately compact.  The default configuration is a
6-layer, 512-wide transformer, because the goal of this package is not to push
the state of the art in language modelling but to obtain a well-behaved
*next-token distribution* that the arithmetic codec can exploit: a sharper
distribution over plausible continuations means likely sentences consume less of
the ``[0, 1]`` range.

The model is a standard pre-norm causal transformer built from PyTorch's
:class:`~torch.nn.TransformerEncoder` with an explicit causal mask.  It predicts
the next token given a prefix, which is exactly the interface the codec needs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .device import autocast_context, resolve_amp_dtype, resolve_device


class ModelConfig:
  """Hyper-parameters for :class:`TinyLM`.

  This is a plain, mutable settings object: it is convenient to tweak a field
  (for example ``config.max_seq_len``) before building a model.

  Attributes:
    vocab_size: Number of tokens in the vocabulary.
    d_model: Embedding / residual-stream width.
    n_layers: Number of transformer blocks.
    n_heads: Number of attention heads (must divide ``d_model``).
    d_ff: Width of the feed-forward sublayer.
    max_seq_len: Maximum sequence length the positional embedding supports.
    dropout: Dropout probability (0 disables dropout entirely).
  """

  FIELDS = ("vocab_size", "d_model", "n_layers", "n_heads", "d_ff", "max_seq_len",
            "dropout")

  def __init__(self,
               vocab_size=43,
               d_model=512,
               n_layers=6,
               n_heads=8,
               d_ff=2048,
               max_seq_len=256,
               dropout=0.0):
    self.vocab_size = vocab_size
    self.d_model = d_model
    self.n_layers = n_layers
    self.n_heads = n_heads
    self.d_ff = d_ff
    self.max_seq_len = max_seq_len
    self.dropout = dropout

  def to_dict(self):
    """Return the configuration as a JSON-friendly dictionary."""
    return {name: getattr(self, name) for name in self.FIELDS}

  @classmethod
  def from_dict(cls, data):
    """Build a configuration from a dictionary, ignoring unknown keys."""
    return cls(**{k: v for k, v in data.items() if k in cls.FIELDS})


class TinyLM(nn.Module):
  """A small causal transformer language model.

  The forward pass returns per-position logits over the vocabulary.  Because
  the attention mask is causal, the logits at position ``i`` depend only on
  tokens ``0..i``, so the last position yields the next-token distribution for
  the whole prefix -- the quantity the codec consumes.
  """

  def __init__(self, config):
    """Initialize the model.

    Args:
      config: The architecture hyper-parameters (:class:`ModelConfig`).
    """
    super().__init__()
    self.config = config

    self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
    self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
    self.embed_dropout = nn.Dropout(config.dropout)

    layer = nn.TransformerEncoderLayer(
        d_model=config.d_model,
        nhead=config.n_heads,
        dim_feedforward=config.d_ff,
        dropout=config.dropout,
        batch_first=True,
        norm_first=True,
        activation="gelu",
    )
    # ``enable_nested_tensor=False`` avoids a spurious warning with norm_first.
    self.encoder = nn.TransformerEncoder(layer,
                                         num_layers=config.n_layers,
                                         enable_nested_tensor=False)
    self.ln_f = nn.LayerNorm(config.d_model)
    self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    self.apply(self._init_weights)

  @staticmethod
  def _init_weights(module):
    """Initialize weights with a small normal distribution.

    Args:
      module: The module to initialize in-place.
    """
    if isinstance(module, nn.Linear):
      nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
      nn.init.ones_(module.weight)
      nn.init.zeros_(module.bias)

  def forward(self, input_ids):
    """Compute next-token logits for every position.

    Args:
      input_ids: A ``long`` tensor of shape ``(batch, seq)``.

    Returns:
      Logits of shape ``(batch, seq, vocab_size)``.

    Raises:
      ValueError: If ``seq`` exceeds ``config.max_seq_len``.
    """
    batch, seq = input_ids.shape
    if seq > self.config.max_seq_len:
      raise ValueError(
          f"sequence length {seq} exceeds max_seq_len {self.config.max_seq_len}")
    positions = torch.arange(seq, device=input_ids.device).unsqueeze(0)
    x = self.token_embedding(input_ids) + self.position_embedding(positions)
    x = self.embed_dropout(x)

    # Causal mask: position ``i`` may only attend to positions ``j <= i``.
    mask = torch.triu(torch.ones(seq, seq, device=input_ids.device, dtype=torch.bool),
                      diagonal=1)
    x = self.encoder(x, mask=mask)
    x = self.ln_f(x)
    return self.lm_head(x)

  @torch.no_grad()
  def next_token_probs(self, input_ids, amp_dtype=None):
    """Return the softmax next-token distribution for the last position.

    This is the codec's workhorse: given the current prefix it returns the
    probability the model assigns to each token being the next one.  When the
    model and input live on a CUDA device the forward pass runs under autocast
    (float16) for speed, but the softmax is always computed in float32 so the
    probabilities the coder consumes stay precise.

    Args:
      input_ids: A 1-D ``(seq,)`` or 2-D ``(batch, seq)`` ``long`` tensor.
      amp_dtype: Optional autocast dtype override.  Defaults to ``float16`` on
        CUDA and ``bfloat16`` elsewhere.

    Returns:
      A ``(vocab_size,)`` tensor for 1-D input, or a ``(batch, vocab_size)``
      tensor for 2-D input.  Rows sum to one.
    """
    squeeze = input_ids.dim() == 1
    if squeeze:
      input_ids = input_ids.unsqueeze(0)
    device = input_ids.device
    dtype = amp_dtype if amp_dtype is not None else resolve_amp_dtype("auto", device)
    with autocast_context(device, dtype, enabled=(device.type == "cuda")):
      logits = self.forward(input_ids)[:, -1, :]
    probs = F.softmax(logits.float(), dim=-1)
    return probs.squeeze(0) if squeeze else probs

  def save(self, path):
    """Save the configuration and weights to ``path``.

    Args:
      path: Destination file path.
    """
    torch.save({"config": self.config.to_dict(), "state_dict": self.state_dict()}, path)

  @classmethod
  def load(cls, path, map_location=None):
    """Load a model previously written by :meth:`save`.

    Args:
      path: The checkpoint path.
      map_location: Device to map the tensors onto.  Defaults to ``"auto"``,
        which resolves to CUDA when available and CPU otherwise.

    Returns:
      The loaded model, moved to the resolved device and set to evaluation mode.
    """
    device = resolve_device(map_location or "auto")
    ckpt = torch.load(path, map_location="cpu")
    model = cls(ModelConfig.from_dict(ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model
