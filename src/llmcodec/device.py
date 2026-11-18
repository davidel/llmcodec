"""Device selection and mixed-precision helpers.

Training and inference both want to use a GPU when one is available and to fall
back to the CPU otherwise, without the caller having to probe
``torch.cuda.is_available()`` everywhere.  These helpers centralise that logic
and the choice of the autocast dtype.

The convention used throughout the package is that a ``device`` string of
``"auto"`` (the default) resolves to CUDA if present and to the CPU otherwise;
any other value is passed straight to :func:`torch.device`.
"""

import torch

# Dtype names accepted by ``resolve_amp_dtype``.
_AMP_DTYPES = {
    "auto": None,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def resolve_device(spec="auto"):
  """Resolve a device specification to a concrete :class:`torch.device`.

  Args:
    spec: ``"auto"`` (or ``None``) to pick CUDA when available and CPU
      otherwise, or any string accepted by :class:`torch.device` (for example
      ``"cpu"``, ``"cuda"``, ``"cuda:1"``).

  Returns:
    The resolved device.
  """
  if spec is None or spec == "auto":
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
  return torch.device(spec)


def resolve_amp_dtype(spec="auto", device=None):
  """Resolve an autocast dtype name for the given device.

  ``"auto"`` selects ``float16`` on CUDA and ``bfloat16`` on CPU, both of which
  have fast tensor-core / vector paths and are safe defaults for training.

  Args:
    spec: One of ``"auto"``, ``"float16"``, ``"bfloat16"`` or ``"float32"``.
    device: The device the dtype will be used on (only used for ``"auto"``).

  Returns:
    The resolved :class:`torch.dtype`.

  Raises:
    ValueError: If ``spec`` is not a recognised dtype name.
  """
  if spec not in _AMP_DTYPES:
    raise ValueError(f"unknown amp dtype {spec!r}; choose from {sorted(_AMP_DTYPES)}")
  if spec == "auto":
    if device is not None and device.type == "cuda":
      return torch.float16
    return torch.bfloat16
  return _AMP_DTYPES[spec]


def autocast_context(device, dtype, enabled):
  """Return a context manager for mixed-precision autocast.

  A thin wrapper around :class:`torch.autocast` so callers do not have to repeat
  the device/dtype/enabled plumbing.  Autocast is only actually engaged when
  ``enabled`` is true; otherwise the returned context is a no-op.

  Args:
    device: The device being used.
    dtype: The autocast dtype.
    enabled: Whether to enable autocast.

  Returns:
    A context manager usable with ``with``.
  """
  return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def make_grad_scaler(device, enabled):
  """Return a gradient scaler for mixed-precision training.

  Gradient scaling is only needed (and only supported) for float16 on CUDA; for
  CPU or bfloat16 the returned scaler is disabled and behaves as a no-op that
  still exposes the ``scale``/``step``/``unscale_`` API.

  Args:
    device: The device being trained on.
    enabled: Whether float16 autocast with a scaler is active.

  Returns:
    A :class:`torch.amp.GradScaler`.
  """
  use_scaler = enabled and device.type == "cuda"
  return torch.amp.GradScaler(device.type, enabled=use_scaler)
