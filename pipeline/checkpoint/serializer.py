"""Portable tensor <-> bytes serialization.

This module never touches pickle for tensor payload (constraint #6 in the
project brief). A tensor is represented as three independent pieces that
together are enough to reconstruct it exactly:

    dtype string (e.g. "torch.bfloat16")   -- what the elements are
    shape (list[int])                      -- how they're arranged
    raw contiguous bytes                    -- the payload itself

The raw bytes are always the tensor's native little-endian in-memory layout
(after .contiguous()), so writing them is a zero-copy memoryview slice, not
a re-encoding step. This module is deliberately torch-free: it operates on
anything that produces a contiguous buffer (numpy arrays, bytes, or the
uint8 view torch_adapter.py hands it for dtypes numpy can't represent, like
bfloat16). That keeps it fully unit-testable in an environment without
torch installed (see training/tests/test_serializer.py).
"""
from __future__ import annotations

import hashlib
from typing import Optional, Sequence

import numpy as np

# torch dtype string -> numpy dtype, for the dtypes numpy can represent
# natively. bfloat16 has no numpy equivalent (numpy < 2.x has no bfloat16
# type at all) so it is intentionally absent here: tensor_adapter.py stores
# and restores bfloat16 tensors via a torch.uint8 byte view instead of
# routing them through numpy.
_TORCH_TO_NUMPY = {
    "torch.float64": np.dtype("float64"),
    "torch.float32": np.dtype("float32"),
    "torch.float16": np.dtype("float16"),
    "torch.int64": np.dtype("int64"),
    "torch.int32": np.dtype("int32"),
    "torch.int16": np.dtype("int16"),
    "torch.int8": np.dtype("int8"),
    "torch.uint8": np.dtype("uint8"),
    "torch.bool": np.dtype("bool"),
}
_NUMPY_TO_TORCH = {v.name: k for k, v in _TORCH_TO_NUMPY.items()}

# Element size in bytes for every dtype string this module knows about,
# including bfloat16 (2 bytes, same layout as float16 but different
# exponent/mantissa split — irrelevant here since we never interpret the
# bits, only move them).
ITEM_SIZE_BYTES = {
    "torch.float64": 8,
    "torch.float32": 4,
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.int64": 8,
    "torch.int32": 4,
    "torch.int16": 2,
    "torch.int8": 1,
    "torch.uint8": 1,
    "torch.bool": 1,
}


def numpy_dtype_for(dtype_str: str) -> Optional[np.dtype]:
    """None means "numpy can't represent this dtype" (bfloat16).

    Accepts both the curated "torch.xxx" strings (used for actual model/
    optimizer tensors, which are always real torch tensors in real usage)
    and a generic "numpy.xxx" fallback (used for numpy-native artifacts
    that never go through torch at all, e.g. RNG generator state arrays --
    see training/utils/seed.py) covering any numpy dtype torch has no
    equivalent for (uint16/32/64, etc).
    """
    if dtype_str in _TORCH_TO_NUMPY:
        return _TORCH_TO_NUMPY[dtype_str]
    if dtype_str.startswith("numpy."):
        try:
            return np.dtype(dtype_str[len("numpy.") :])
        except TypeError:
            return None
    return None


def dtype_str_from_numpy(np_dtype: np.dtype) -> str:
    key = np.dtype(np_dtype).name
    if key in _NUMPY_TO_TORCH:
        return _NUMPY_TO_TORCH[key]
    return f"numpy.{key}"


def numel_of(shape: Sequence[int]) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


def byte_size_of(dtype_str: str, shape: Sequence[int]) -> int:
    if dtype_str in ITEM_SIZE_BYTES:
        return numel_of(shape) * ITEM_SIZE_BYTES[dtype_str]
    np_dtype = numpy_dtype_for(dtype_str)
    if np_dtype is not None:
        return numel_of(shape) * np_dtype.itemsize
    raise ValueError(f"unknown dtype string: {dtype_str}")


def sha256_hex(data) -> str:
    """Accepts anything supporting the buffer protocol (bytes, bytearray,
    memoryview, numpy array) without copying it into a new bytes object."""
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def buffer_from_numpy(arr: np.ndarray) -> memoryview:
    """Zero-copy byte view of a numpy array's storage.

    Falls back to np.ascontiguousarray only when the array truly isn't
    contiguous (e.g. a transposed view) — that copy is unavoidable since
    there is no contiguous byte range to point at otherwise.
    """
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return memoryview(arr).cast("B")


def reconstruct_numpy(dtype_str: str, shape: Sequence[int], raw: bytes) -> np.ndarray:
    np_dtype = numpy_dtype_for(dtype_str)
    if np_dtype is None:
        raise ValueError(
            f"{dtype_str} has no numpy representation; use "
            "pipeline.checkpoint.tensor_adapter to reconstruct a torch tensor directly"
        )
    arr = np.frombuffer(raw, dtype=np_dtype)
    return arr.reshape(list(shape))
