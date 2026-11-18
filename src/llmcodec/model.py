"""Small decoder-only transformer language model.

The architecture is deliberately compact -- the default configuration is a
6-layer, 512-wide transformer -- because the goal of this package is not to push
state-of-the-art language modelling but to obtain a well-behaved *next-token
distribution* that the arithmetic codec can exploit.

The model predicts the next token given a prefix.  We use PyTorch's built-in
:class:`~torch.nn.TransformerEncoder` with a causal mask, which keeps the
implementation short while remaining a standard pre-norm transformer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Hyper-parameters for :class:`TinyLM`."""

    vocab_size: int = 43
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    max_seq_len: int = 256
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ModelConfig:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class TinyLM(nn.Module):
    """A small causal transformer language model."""

    def __init__(self, config: ModelConfig):
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
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.n_layers, enable_nested_tensor=False
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return logits of shape ``(batch, seq, vocab_size)``."""
        batch, seq = input_ids.shape
        if seq > self.config.max_seq_len:
            raise ValueError(f"sequence length {seq} exceeds max_seq_len {self.config.max_seq_len}")
        positions = torch.arange(seq, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embed_dropout(x)

        # Causal mask: position ``i`` may only attend to ``j <= i``.
        mask = torch.triu(
            torch.ones(seq, seq, device=input_ids.device, dtype=torch.bool), diagonal=1
        )
        x = self.encoder(x, mask=mask)
        x = self.ln_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def next_token_probs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Softmax next-token distribution for the *last* position.

        Accepts a 1-D ``(seq,)`` or 2-D ``(batch, seq)`` tensor and returns a
        ``(vocab_size,)`` or ``(batch, vocab_size)`` tensor.
        """
        squeeze = input_ids.dim() == 1
        if squeeze:
            input_ids = input_ids.unsqueeze(0)
        logits = self.forward(input_ids)[:, -1, :]
        probs = F.softmax(logits.float(), dim=-1)
        return probs.squeeze(0) if squeeze else probs

    # -- persistence ---------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({"config": self.config.to_dict(), "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, map_location: str | None = "cpu") -> TinyLM:
        ckpt = torch.load(path, map_location=map_location)
        model = cls(ModelConfig.from_dict(ckpt["config"]))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model
