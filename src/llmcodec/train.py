"""Training loop for :class:`~llmcodec.model.TinyLM`.

Trains a character-level causal LM on TinyStories (or the built-in offline
corpus) with AdamW, a warmup+cosine learning-rate schedule and gradient
clipping.  The trained checkpoint can be loaded directly by the codec.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F

from .data import (
    BlockDataset,
    build_token_stream,
    iter_batches,
    load_tinystories,
    sample_stories,
)
from .model import ModelConfig, TinyLM
from .vocab import DEFAULT_VOCAB, Vocab


@dataclass
class TrainConfig:
    """Everything needed to reproduce a training run."""

    # data
    source: str = "tinystories"  # "tinystories" or "sample"
    split: str = "train"
    max_stories: int | None = 200_000

    # sequence / batch
    block_size: int = 256
    batch_size: int = 64

    # optimisation
    steps: int = 50_000
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # bookkeeping
    seed: int = 1337
    device: str = "cpu"
    log_every: int = 50
    eval_every: int = 500
    eval_batches: int = 20
    out_dir: str = "runs/default"
    num_threads: int = 0  # 0 -> leave PyTorch default

    model: ModelConfig = field(default_factory=ModelConfig)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["model"] = self.model.to_dict()
        return data


def _lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


def _prepare_tokens(cfg: TrainConfig, vocab: Vocab) -> torch.Tensor:
    if cfg.source == "sample":
        texts = sample_stories()
    elif cfg.source == "tinystories":
        texts = load_tinystories(split=cfg.split, max_stories=cfg.max_stories)
    else:
        raise ValueError(f"unknown data source {cfg.source!r}")
    return build_token_stream(texts, vocab)


@torch.no_grad()
def evaluate(
    model: TinyLM,
    dataset: BlockDataset,
    cfg: TrainConfig,
    generator: torch.Generator,
) -> float:
    model.eval()
    losses: list[float] = []
    for i, (x, y) in enumerate(iter_batches(dataset, cfg.batch_size, True, generator)):
        if i >= cfg.eval_batches:
            break
        x, y = x.to(cfg.device), y.to(cfg.device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def train(cfg: TrainConfig | None = None, *, vocab: Vocab | None = None) -> str:
    """Run training and return the path to the saved checkpoint."""
    cfg = cfg or TrainConfig()
    vocab = vocab or DEFAULT_VOCAB

    if cfg.num_threads > 0:
        torch.set_num_threads(cfg.num_threads)
    torch.manual_seed(cfg.seed)

    os.makedirs(cfg.out_dir, exist_ok=True)

    tokens = _prepare_tokens(cfg, vocab)
    dataset = BlockDataset(tokens, cfg.block_size)

    # Held-out split (last 5%) for a lightweight validation signal.
    n_val = max(1, len(dataset) // 20)
    val_dataset = BlockDataset(tokens[-(n_val + 1) * cfg.block_size :], cfg.block_size)
    train_dataset = BlockDataset(tokens[: -(n_val + 1) * cfg.block_size], cfg.block_size)

    model = TinyLM(cfg.model).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[llmcodec] vocab={vocab.vocab_size} params={n_params:,} tokens={tokens.numel():,}")
    print(f"[llmcodec] train_blocks={len(train_dataset):,} val_blocks={len(val_dataset):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    generator = torch.Generator().manual_seed(cfg.seed)
    ckpt_path = os.path.join(cfg.out_dir, "model.pt")

    running: list[float] = []
    t0 = time.time()
    step = 0
    while step < cfg.steps:
        for x, y in iter_batches(train_dataset, cfg.batch_size, True, generator):
            if step >= cfg.steps:
                break
            lr = _lr_at(step, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr

            x, y = x.to(cfg.device), y.to(cfg.device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            running.append(loss.item())
            if step % cfg.log_every == 0:
                avg = sum(running[-cfg.log_every :]) / len(running[-cfg.log_every :])
                dt = time.time() - t0
                print(
                    f"[llmcodec] step {step:>6}/{cfg.steps} "
                    f"loss {avg:.4f} lr {lr:.2e} elapsed {dt:6.1f}s"
                )
            if cfg.eval_every and step > 0 and step % cfg.eval_every == 0:
                val = evaluate(model, val_dataset, cfg, generator)
                print(f"[llmcodec] step {step:>6} val_loss {val:.4f}")
            step += 1

    model.save(ckpt_path)
    with open(os.path.join(cfg.out_dir, "train_config.json"), "w") as fh:
        json.dump({"train": cfg.to_dict(), "vocab": vocab.to_dict()}, fh, indent=2)

    final_val = evaluate(model, val_dataset, cfg, generator)
    print(f"[llmcodec] done. final val_loss {final_val:.4f}. checkpoint -> {ckpt_path}")
    return ckpt_path


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train the llmcodec TinyLM.")
    parser.add_argument("--source", choices=["tinystories", "sample"], default="tinystories")
    parser.add_argument("--max-stories", type=int, default=200_000)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out-dir", default="runs/default")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    cfg = TrainConfig(
        source=args.source,
        max_stories=args.max_stories,
        steps=args.steps,
        batch_size=args.batch_size,
        block_size=args.block_size,
        lr=args.lr,
        out_dir=args.out_dir,
        device=args.device,
        num_threads=args.num_threads,
        seed=args.seed,
    )
    train(cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
