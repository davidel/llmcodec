"""Training loop for :class:`~llmcodec.model.TinyLM`.

Trains a character-level causal language model on TinyStories (or the built-in
offline corpus) with AdamW, a warmup-plus-cosine learning-rate schedule and
gradient clipping.  The resulting checkpoint can be loaded directly by the codec
with :meth:`~llmcodec.model.TinyLM.load`.

The run automatically uses a GPU when one is available and falls back to the CPU
otherwise, and supports mixed-precision autocast (see
:mod:`llmcodec.device`).  A short validation split is carved off the end of the
token stream so the run reports a held-out loss without needing a second
download.
"""

import argparse
import json
import logging
import math
import os
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .data import (
    BlockDataset,
    build_token_stream,
    iter_batches,
    load_tinystories,
    sample_stories,
)
from .device import (
    autocast_context,
    make_grad_scaler,
    resolve_amp_dtype,
    resolve_device,
)
from .model import ModelConfig, TinyLM
from .vocab import DEFAULT_VOCAB

logger = logging.getLogger(__name__)


class TrainConfig:
  """Everything needed to reproduce a training run.

  A plain, mutable settings object.  ``model`` holds the architecture
  hyper-parameters and defaults to a fresh :class:`~llmcodec.model.ModelConfig`.

  Attributes:
    source: Data source, ``"tinystories"`` or ``"sample"``.
    split: Dataset split to load when ``source == "tinystories"``.
    max_stories: Cap on the number of stories (``None`` for all).
    block_size: Training context length.
    batch_size: Number of blocks per optimization step.
    steps: Total number of optimizer steps.
    lr: Peak learning rate.
    min_lr: Final learning rate at the end of the cosine schedule.
    warmup_steps: Linear warmup length.
    weight_decay: AdamW weight decay.
    beta1: AdamW first-moment decay.
    beta2: AdamW second-moment decay.
    grad_clip: Maximum gradient norm.
    seed: RNG seed for reproducibility.
    device: Torch device string.  ``"auto"`` picks CUDA when available.
    amp: Whether to use mixed-precision autocast (CUDA or CPU).
    amp_dtype: Autocast dtype name (``"auto"``, ``"float16"``, ``"bfloat16"`` or
      ``"float32"``).  ``"auto"`` is ``float16`` on CUDA and ``bfloat16`` on CPU.
    log_every: Print a loss line every this many steps.
    eval_every: Run validation every this many steps (0 disables).
    eval_batches: Number of batches per validation pass.
    out_dir: Directory for the checkpoint and config dump.
    num_threads: Torch CPU thread count (0 leaves the default).
    model: The model architecture configuration.
  """

  FIELDS = ("source", "split", "max_stories", "block_size", "batch_size", "steps", "lr",
            "min_lr", "warmup_steps", "weight_decay", "beta1", "beta2", "grad_clip",
            "seed", "device", "amp", "amp_dtype", "log_every", "eval_every",
            "eval_batches", "out_dir", "num_threads")

  def __init__(self,
               source="tinystories",
               split="train",
               max_stories=200_000,
               block_size=256,
               batch_size=64,
               steps=50_000,
               lr=3e-4,
               min_lr=3e-5,
               warmup_steps=500,
               weight_decay=0.1,
               beta1=0.9,
               beta2=0.95,
               grad_clip=1.0,
               seed=1337,
               device="auto",
               amp=True,
               amp_dtype="auto",
               log_every=50,
               eval_every=500,
               eval_batches=20,
               out_dir="runs/default",
               num_threads=0,
               model=None):
    self.source = source
    self.split = split
    self.max_stories = max_stories
    self.block_size = block_size
    self.batch_size = batch_size
    self.steps = steps
    self.lr = lr
    self.min_lr = min_lr
    self.warmup_steps = warmup_steps
    self.weight_decay = weight_decay
    self.beta1 = beta1
    self.beta2 = beta2
    self.grad_clip = grad_clip
    self.seed = seed
    self.device = device
    self.amp = amp
    self.amp_dtype = amp_dtype
    self.log_every = log_every
    self.eval_every = eval_every
    self.eval_batches = eval_batches
    self.out_dir = out_dir
    self.num_threads = num_threads
    self.model = model if model is not None else ModelConfig()

  def to_dict(self):
    """Return the configuration as a JSON-friendly dictionary."""
    data = {name: getattr(self, name) for name in self.FIELDS}
    data["model"] = self.model.to_dict()
    return data

  @classmethod
  def from_dict(cls, data, base=None):
    """Build a configuration from a dictionary, ignoring unknown keys.

    The dictionary may be flat (the shape of :meth:`to_dict`) or wrapped as
    ``{"train": {...}, "vocab": {...}}``, which is the shape written to
    ``train_config.json`` by :func:`train`.  The ``vocab`` block is ignored
    because the vocabulary is fixed by the codec.

    Args:
      data: A mapping of configuration fields.  A ``"model"`` sub-mapping holds
        the architecture hyper-parameters.
      base: An optional existing :class:`TrainConfig` to use as the starting
        point.  Fields absent from ``data`` keep their ``base`` value.  Defaults
        to a fresh :class:`TrainConfig`, so absent fields take the defaults.

    Returns:
      A new :class:`TrainConfig` instance.
    """
    if "train" in data:
      data = data["train"]
    base = base or cls()
    fields = {name: data[name] for name in cls.FIELDS if name in data}
    model = base.model
    if "model" in data:
      model = ModelConfig.from_dict(data["model"])
    return cls(**{**fields, "model": model})

  def validate(self, vocab=None):
    """Check the configuration for combinations that would fail later.

    Catching these here turns a confusing crash deep inside ``torch`` into an
    actionable message.

    Args:
      vocab: The vocabulary the model will use.  When given, its size must match
        ``model.vocab_size`` or the checkpoint would not decode.

    Raises:
      ValueError: If the architecture or training settings are inconsistent.
    """
    model = self.model
    if model.d_model % model.n_heads != 0:
      raise ValueError(f"d_model ({model.d_model}) must be divisible by "
                       f"n_heads ({model.n_heads})")
    if self.block_size > model.max_seq_len:
      raise ValueError(f"block_size ({self.block_size}) exceeds max_seq_len "
                       f"({model.max_seq_len})")
    if vocab is not None and model.vocab_size != vocab.vocab_size:
      raise ValueError(f"model.vocab_size ({model.vocab_size}) does not match "
                       f"the vocabulary size ({vocab.vocab_size})")
    if not 0.0 <= model.dropout < 1.0:
      raise ValueError(f"dropout ({model.dropout}) must be in [0, 1)")
    if self.steps < 1:
      raise ValueError(f"steps ({self.steps}) must be positive")
    if self.batch_size < 1:
      raise ValueError(f"batch_size ({self.batch_size}) must be positive")


def _read_config_json(path):
  """Read a training-config JSON file into a plain dictionary.

  Args:
    path: Path to the JSON file.  Both the flat shape of
      :meth:`TrainConfig.to_dict` and the wrapped ``{"train": ..., "vocab": ...}``
      shape written by :func:`train` are accepted.

  Returns:
    The configuration mapping, with any outer ``"train"`` wrapper removed.

  Raises:
    ValueError: If the file cannot be read, is not valid JSON, or is not an
      object.
  """
  try:
    with open(path, "r", encoding="utf-8") as fh:
      data = json.load(fh)
  except OSError as exc:
    raise ValueError(f"could not read config file {path!r}: {exc}") from exc
  except json.JSONDecodeError as exc:
    raise ValueError(f"config file {path!r} is not valid JSON: {exc}") from exc
  if not isinstance(data, dict):
    raise ValueError(f"config file {path!r} must contain a JSON object")
  if "train" in data:
    data = data["train"]
    if not isinstance(data, dict):
      raise ValueError(f"config file {path!r} has a non-object 'train' section")
  return data


def load_config(path):
  """Load a :class:`TrainConfig` from a JSON file.

  Args:
    path: Path to a JSON config file.  See :func:`_read_config_json` for the
      accepted shapes.

  Returns:
    A new :class:`TrainConfig` populated from the file.

  Raises:
    ValueError: If the file cannot be read or parsed.
  """
  return TrainConfig.from_dict(_read_config_json(path))


def merged_config(args):
  """Merge an optional ``--config`` file with the explicit CLI flags.

  Precedence is ``TrainConfig`` defaults < ``--config`` file < explicit flags.
  Every train flag defaults to ``None`` so that "not given" stays
  distinguishable from "set to the default value"; only non-``None`` attributes
  are carried over.  Architecture flags are nested under ``"model"``.

  Args:
    args: A parsed argument namespace.  Missing attributes are treated as unset.

  Returns:
    A :class:`types.SimpleNamespace` holding the merged raw configuration.
  """
  config_path = getattr(args, "config", None)
  raw = dict(_read_config_json(config_path)) if config_path is not None else {}
  raw["model"] = dict(raw.get("model", {}))
  for name in TrainConfig.FIELDS:
    value = getattr(args, name, None)
    if value is not None:
      raw[name] = value
  for name in ModelConfig.FIELDS:
    value = getattr(args, name, None)
    if value is not None:
      raw["model"][name] = value
  return SimpleNamespace(**raw)


def config_from_args(args):
  """Build a :class:`TrainConfig` from parsed CLI arguments.

  Args:
    args: A parsed argument namespace, as produced by :func:`add_train_arguments`.

  Returns:
    A new :class:`TrainConfig` with the ``--config`` file and explicit flags
    merged in.

  Raises:
    ValueError: If the ``--config`` file cannot be read or parsed.
  """
  return TrainConfig.from_dict(vars(merged_config(args)))


def add_train_arguments(parser):
  """Register the ``train`` flags on ``parser``.

  Every flag defaults to ``None`` so that "not given" stays distinguishable from
  "set to the default value"; that is what lets an explicit flag override a
  ``--config`` file while unset flags keep the file's value.  See
  :func:`merged_config`.

  Args:
    parser: The parser (or subparser) to add the arguments to.
  """
  parser.add_argument(
      "--config",
      default=None,
      help="JSON training config to use as the base; flags override it",
  )
  parser.add_argument(
      "--source",
      choices=["tinystories", "sample"],
      default=None,
      help="training corpus: 'tinystories' downloads it, 'sample' is offline "
      "(default: tinystories)",
  )
  parser.add_argument("--split",
                      default=None,
                      help="dataset split to load (default: train)")
  parser.add_argument(
      "--max-stories",
      type=int,
      default=None,
      help="cap on the number of stories to load from the corpus (default: 200000)",
  )
  parser.add_argument("--steps",
                      type=int,
                      default=None,
                      help="number of optimizer steps to run (default: 50000)")
  parser.add_argument("--batch-size",
                      type=int,
                      default=None,
                      help="number of blocks per optimizer step (default: 64)")
  parser.add_argument("--block-size",
                      type=int,
                      default=None,
                      help="training context length in tokens (default: 256)")
  parser.add_argument("--lr",
                      type=float,
                      default=None,
                      help="peak learning rate (default: 3e-4)")
  parser.add_argument(
      "--min-lr",
      type=float,
      default=None,
      help="final learning rate of the cosine schedule (default: 3e-5)",
  )
  parser.add_argument("--warmup-steps",
                      type=int,
                      default=None,
                      help="linear warmup length (default: 500)")
  parser.add_argument("--weight-decay",
                      type=float,
                      default=None,
                      help="AdamW weight decay (default: 0.1)")
  parser.add_argument("--beta1",
                      type=float,
                      default=None,
                      help="AdamW beta1 (default: 0.9)")
  parser.add_argument("--beta2",
                      type=float,
                      default=None,
                      help="AdamW beta2 (default: 0.95)")
  parser.add_argument("--grad-clip",
                      type=float,
                      default=None,
                      help="gradient-norm clip value (default: 1.0)")
  parser.add_argument("--log-every",
                      type=int,
                      default=None,
                      help="print a loss line every N steps (default: 50)")
  parser.add_argument(
      "--eval-every",
      type=int,
      default=None,
      help="run validation every N steps, 0 disables (default: 500)",
  )
  parser.add_argument("--eval-batches",
                      type=int,
                      default=None,
                      help="batches per validation pass (default: 20)")
  parser.add_argument(
      "--out-dir",
      default=None,
      help="directory for the checkpoint and config (default: runs/default)",
  )
  parser.add_argument(
      "--device",
      default=None,
      help="torch device: 'auto' (CUDA if available), 'cpu' or 'cuda' (default: auto)",
  )
  parser.add_argument(
      "--amp",
      action=argparse.BooleanOptionalAction,
      default=None,
      help="enable mixed-precision autocast (default: enabled)",
  )
  parser.add_argument(
      "--amp-dtype",
      choices=["auto", "float16", "bfloat16", "float32"],
      default=None,
      help="autocast dtype (default: auto)",
  )
  parser.add_argument("--num-threads",
                      type=int,
                      default=None,
                      help="torch CPU threads (0 = library default)")
  parser.add_argument("--seed", type=int, default=None, help="RNG seed (default: 1337)")
  parser.add_argument("--d-model",
                      type=int,
                      default=None,
                      help="embedding / residual-stream width (default: 512)")
  parser.add_argument("--n-layers",
                      type=int,
                      default=None,
                      help="number of transformer blocks (default: 6)")
  parser.add_argument("--n-heads",
                      type=int,
                      default=None,
                      help="attention heads; must divide d_model (default: 8)")
  parser.add_argument("--d-ff",
                      type=int,
                      default=None,
                      help="feed-forward sublayer width (default: 2048)")
  parser.add_argument(
      "--max-seq-len",
      type=int,
      default=None,
      help="positional-embedding limit; must be >= block_size (default: 256)",
  )
  parser.add_argument("--dropout",
                      type=float,
                      default=None,
                      help="dropout probability (default: 0.0)")


def _lr_at(step, cfg):
  """Return the learning rate for a given step.

  The schedule is a linear warmup followed by a cosine decay to ``min_lr``.

  Args:
    step: The current optimizer step (0-indexed).
    cfg: The training configuration.

  Returns:
    The learning rate for this step.
  """
  if step < cfg.warmup_steps:
    return cfg.lr * (step + 1) / cfg.warmup_steps
  if step >= cfg.steps:
    return cfg.min_lr
  progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
  coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
  return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


def _prepare_tokens(cfg, vocab):
  """Load the corpus and turn it into a single token stream.

  Args:
    cfg: The training configuration.
    vocab: The vocabulary.

  Returns:
    A 1-D ``long`` tensor of token ids.

  Raises:
    ValueError: If ``cfg.source`` is not a known source.
  """
  if cfg.source == "sample":
    texts = sample_stories()
  elif cfg.source == "tinystories":
    texts = load_tinystories(split=cfg.split, max_stories=cfg.max_stories)
  else:
    raise ValueError(f"unknown data source {cfg.source!r}")
  return build_token_stream(texts, vocab)


@torch.no_grad()
def evaluate(model, dataset, cfg, generator, device=None, amp_dtype=None):
  """Compute the mean next-token cross-entropy over a few batches.

  Args:
    model: The model to evaluate (temporarily switched to eval mode).
    dataset: The validation block dataset.
    cfg: The training configuration.
    generator: RNG used for reproducible batch selection.
    device: The resolved device.  Defaults to resolving ``cfg.device``.
    amp_dtype: The autocast dtype.  Defaults to resolving ``cfg.amp_dtype``.

  Returns:
    The mean loss over up to ``cfg.eval_batches`` batches.
  """
  device = device or resolve_device(cfg.device)
  amp_dtype = amp_dtype or resolve_amp_dtype(cfg.amp_dtype, device)
  use_amp = cfg.amp
  model.eval()
  losses = []
  for i, (x, y) in enumerate(iter_batches(dataset, cfg.batch_size, True, generator)):
    if i >= cfg.eval_batches:
      break
    x, y = x.to(device), y.to(device)
    with autocast_context(device, amp_dtype, use_amp):
      logits = model(x)
    loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
    losses.append(loss.item())
  model.train()
  return sum(losses) / max(1, len(losses))


def train(cfg=None, vocab=None):
  """Run a full training job and return the checkpoint path.

  Args:
    cfg: The training configuration.  Defaults to :class:`TrainConfig`.
    vocab: The vocabulary.  Defaults to :data:`~llmcodec.vocab.DEFAULT_VOCAB`.

  Returns:
    The path to the saved ``model.pt`` checkpoint.
  """
  cfg = cfg or TrainConfig()
  vocab = vocab or DEFAULT_VOCAB
  cfg.validate(vocab)

  device = resolve_device(cfg.device)
  amp_dtype = resolve_amp_dtype(cfg.amp_dtype, device)
  if device.type == "cpu" and cfg.num_threads > 0:
    torch.set_num_threads(cfg.num_threads)
  torch.manual_seed(cfg.seed)

  os.makedirs(cfg.out_dir, exist_ok=True)

  tokens = _prepare_tokens(cfg, vocab)

  # Carve off the last ~5% of blocks as a held-out validation split.
  n_val = max(1, len(BlockDataset(tokens, cfg.block_size)) // 20)
  val_tokens = tokens[-(n_val + 1) * cfg.block_size:]
  train_tokens = tokens[:-(n_val + 1) * cfg.block_size]
  train_dataset = BlockDataset(train_tokens, cfg.block_size)
  val_dataset = BlockDataset(val_tokens, cfg.block_size)

  model = TinyLM(cfg.model).to(device)
  n_params = sum(p.numel() for p in model.parameters())
  logger.info("vocab=%d params=%s tokens=%s", vocab.vocab_size, f"{n_params:,}",
              f"{tokens.numel():,}")
  logger.info("train_blocks=%s val_blocks=%s", f"{len(train_dataset):,}",
              f"{len(val_dataset):,}")
  amp_note = f"amp={amp_dtype}" if cfg.amp else "amp=off"
  logger.info("device=%s %s", device, amp_note)

  optimizer = torch.optim.AdamW(
      model.parameters(),
      lr=cfg.lr,
      betas=(cfg.beta1, cfg.beta2),
      weight_decay=cfg.weight_decay,
  )
  scaler = make_grad_scaler(device, cfg.amp)

  generator = torch.Generator().manual_seed(cfg.seed)
  ckpt_path = os.path.join(cfg.out_dir, "model.pt")

  running = []
  t0 = time.time()
  step = 0
  while step < cfg.steps:
    for x, y in iter_batches(train_dataset, cfg.batch_size, True, generator):
      if step >= cfg.steps:
        break
      lr = _lr_at(step, cfg)
      for group in optimizer.param_groups:
        group["lr"] = lr

      x, y = x.to(device), y.to(device)
      with autocast_context(device, amp_dtype, cfg.amp):
        logits = model(x)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                               y.reshape(-1))

      optimizer.zero_grad(set_to_none=True)
      scaler.scale(loss).backward()
      scaler.unscale_(optimizer)
      torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
      scaler.step(optimizer)
      scaler.update()

      running.append(loss.item())
      if step % cfg.log_every == 0:
        window = running[-cfg.log_every:]
        avg = sum(window) / len(window)
        dt = time.time() - t0
        logger.info("step %6d/%d loss %.4f lr %.2e elapsed %6.1fs", step, cfg.steps,
                    avg, lr, dt)
      if cfg.eval_every and step > 0 and step % cfg.eval_every == 0:
        val = evaluate(model,
                       val_dataset,
                       cfg,
                       generator,
                       device=device,
                       amp_dtype=amp_dtype)
        logger.info("step %6d val_loss %.4f", step, val)
      step += 1

  model.save(ckpt_path)
  with open(os.path.join(cfg.out_dir, "train_config.json"), "w") as fh:
    json.dump({"train": cfg.to_dict(), "vocab": vocab.to_dict()}, fh, indent=2)

  final_val = evaluate(model,
                       val_dataset,
                       cfg,
                       generator,
                       device=device,
                       amp_dtype=amp_dtype)
  logger.info("done. final val_loss %.4f. checkpoint -> %s", final_val, ckpt_path)
  return ckpt_path


def main(argv=None):
  """Command-line entry point for ``python -m llmcodec.train``.

  Args:
    argv: Argument list (defaults to ``sys.argv[1:]``).

  Returns:
    The process exit code.
  """
  import argparse

  logging.basicConfig(level=logging.INFO, format="[llmcodec] %(message)s")

  parser = argparse.ArgumentParser(description="Train the llmcodec TinyLM.")
  add_train_arguments(parser)
  args = parser.parse_args(argv)

  cfg = config_from_args(args)
  train(cfg)
  return 0


if __name__ == "__main__":  # pragma: no cover
  raise SystemExit(main())
