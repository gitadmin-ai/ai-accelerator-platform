"""Framework dispatch: numpy arrays are handled directly (this module never
imports torch at module scope); a torch.Tensor is recognized by duck-typing
its module name and handed off to tensor_adapter.py, imported lazily so
that importing this module -- and everything that imports it -- never
requires torch to be installed.
"""
from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np

from pipeline.checkpoint import serializer


def _is_torch_tensor_module(value: Any) -> bool:
    return type(value).__module__.split(".")[0] == "torch"


def is_tensor_like(value: Any) -> bool:
    return isinstance(value, np.ndarray) or _is_torch_tensor_module(value)


def describe_tensor(value: Any, timing=None) -> Tuple[str, List[int], memoryview]:
    """Returns (dtype_str, shape, zero-copy byte buffer)."""
    if isinstance(value, np.ndarray):
        dtype_str = serializer.dtype_str_from_numpy(value.dtype)
        return dtype_str, list(value.shape), serializer.buffer_from_numpy(value)

    if _is_torch_tensor_module(value):
        from pipeline.checkpoint import tensor_adapter

        if not tensor_adapter.is_torch_tensor(value):
            raise TypeError(f"unsupported torch-namespaced type: {type(value)}")

        if value.device.type != "cpu":
            if timing is not None:
                with timing.measure("gpu_to_cpu_s"):
                    cpu_tensor = tensor_adapter.stage_to_cpu(value)
                    tensor_adapter.synchronize_staging(value.device)
            else:
                cpu_tensor = tensor_adapter.stage_to_cpu(value)
                tensor_adapter.synchronize_staging(value.device)
        else:
            cpu_tensor = value
        return tensor_adapter.describe_torch_tensor(cpu_tensor)

    raise TypeError(f"unsupported tensor type for checkpointing: {type(value)}")


def reconstruct_tensor(dtype_str: str, shape: List[int], raw: bytes):
    """Best-effort generic reconstruction. Returns a numpy array when the
    dtype has a numpy representation; otherwise returns the raw bytes
    unchanged (callers needing e.g. bfloat16 as a torch.Tensor should call
    pipeline.checkpoint.tensor_adapter.reconstruct_torch_tensor directly).
    """
    np_dtype = serializer.numpy_dtype_for(dtype_str)
    if np_dtype is not None:
        return serializer.reconstruct_numpy(dtype_str, shape, raw)
    return raw
