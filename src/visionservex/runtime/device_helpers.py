# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Shared helpers for device-aware tensor handling.

All wired engines should use these helpers to ensure:
- Float tensors are cast to the model dtype.
- Integer/bool tensors (token IDs, attention masks) are never cast.
- Models are safely moved to the target device.
"""

from __future__ import annotations

from typing import Any


def select_dtype(device: str, precision: str) -> Any:
    """Return the torch dtype for the given device + precision string."""
    try:
        import torch  # type: ignore
    except ImportError:
        return None

    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp32":
        return torch.float32

    # auto: choose fp16 on GPU (where it saves memory), fp32 on CPU/MPS
    if precision == "auto":
        base = device.split(":")[0].lower()
        if base == "cuda":
            return torch.float16
        # MPS fp16 has gaps; keep fp32 for safety
        return torch.float32

    return torch.float32


def move_inputs_to_device(
    inputs: dict[str, Any],
    device: str,
    dtype: Any,
    *,
    cast_floats_only: bool = True,
) -> dict[str, Any]:
    """Move all tensor values in ``inputs`` to ``device``, optionally casting.

    Rules:
    - Integer-dtype tensors (token IDs, attention masks, position IDs) are
      **never** cast — doing so corrupts embedding lookups.
    - Only float tensors are cast to ``dtype`` when ``cast_floats_only=True``.
    - Non-tensor values are passed through unchanged.
    """
    try:
        import torch  # type: ignore
    except ImportError:
        return inputs

    out: dict[str, Any] = {}
    for k, v in inputs.items():
        if not isinstance(v, torch.Tensor):
            out[k] = v
            continue
        v = v.to(device=device)
        if cast_floats_only and v.is_floating_point() and dtype is not None:
            v = v.to(dtype=dtype)
        out[k] = v
    return out


def safe_model_to_device(model: Any, device: str, dtype: Any) -> Any:
    """Move ``model`` to ``device`` and optionally convert to ``dtype``.

    Returns the model (allows chaining).
    """
    try:
        import torch  # type: ignore
    except ImportError:
        return model

    model = model.to(device)
    # Only convert the model's floating-point parameters; let layer-norm etc.
    # stay in fp32 if the platform needs it (MPS).
    if dtype is not None and dtype != torch.float32:
        import contextlib

        with contextlib.suppress(Exception):
            model = model.to(dtype=dtype)
    return model


def device_is_available(device_name: str) -> bool:
    """Return True if the named device is available and passes sanity."""
    from visionservex.runtime.device import available_devices

    base = device_name.split(":")[0].lower()
    for d in available_devices():
        if d.name.split(":")[0].lower() == base:
            return d.available and d.sanity_ok is not False
    return False


__all__ = [
    "device_is_available",
    "move_inputs_to_device",
    "safe_model_to_device",
    "select_dtype",
]
