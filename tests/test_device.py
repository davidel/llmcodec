"""Tests for device selection and mixed-precision helpers."""

import pytest
import torch

from llmcodec.device import (
    autocast_context,
    make_grad_scaler,
    resolve_amp_dtype,
    resolve_device,
)


def test_resolve_device_auto_matches_cuda_availability():
  device = resolve_device("auto")
  assert device.type == ("cuda" if torch.cuda.is_available() else "cpu")


def test_resolve_device_explicit():
  assert resolve_device("cpu").type == "cpu"


def test_resolve_amp_dtype_auto_cuda_is_float16():
  assert resolve_amp_dtype("auto", torch.device("cuda")) == torch.float16


def test_resolve_amp_dtype_auto_cpu_is_bfloat16():
  assert resolve_amp_dtype("auto", torch.device("cpu")) == torch.bfloat16


def test_resolve_amp_dtype_explicit():
  assert resolve_amp_dtype("float32", torch.device("cpu")) == torch.float32
  assert resolve_amp_dtype("float16", torch.device("cpu")) == torch.float16


def test_resolve_amp_dtype_unknown_raises():
  import pytest

  with pytest.raises(ValueError):
    resolve_amp_dtype("int8", torch.device("cpu"))


def test_autocast_context_disabled_is_noop():
  with autocast_context(torch.device("cpu"), torch.bfloat16, enabled=False):
    x = torch.ones(2, dtype=torch.float32)
  assert x.dtype == torch.float32


def test_grad_scaler_disabled_on_cpu():
  scaler = make_grad_scaler(torch.device("cpu"), enabled=True)
  # On CPU (or bfloat16) the scaler is disabled but still exposes the API.
  assert scaler.is_enabled() is False


def test_grad_scaler_enabled_on_cuda():
  if not torch.cuda.is_available():
    pytest.skip("CUDA not available")
  scaler = make_grad_scaler(torch.device("cuda"), enabled=True)
  assert scaler.is_enabled()
