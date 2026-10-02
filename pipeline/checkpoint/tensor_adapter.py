"""torch.Tensor <-> (dtype_str, shape, buffer) adapter.

The one module in training/checkpoint allowed to import torch unconditionally
-- everything else (format, serializer, chunker, blobstore_backend,
parallel_writer, manager, validation) stays torch-optional so it can be
unit-tested with plain numpy arrays in environments without torch/CUDA
installed. tensor_io.py imports this module lazily, only once it has
actually seen a torch.Tensor.

GPU -> CPU staging (see training/README.md's "GPU -> CPU handling"
section): a CUDA tensor is copied into a pinned CPU staging buffer with a
non-blocking D2H copy rather than a plain blocking `tensor.cpu()` per
tensor, so the copy for tensor N+1 can be issued before tensor N's copy
has necessarily retired -- the caller (manager.py) issues stage_to_cpu()
for every tensor first, then a single synchronize, then chunks/hashes the
now-resident host buffers. This is the "correct and measurable" version
called for in the brief, not a fully overlapped copy/chunk/write pipeline.
"""
from __future__ import annotations

from typing import Any, List, Tuple

import torch


def torch_dtype_str(dtype: "torch.dtype") -> str:
    return str(dtype)  # e.g. "torch.float32", "torch.bfloat16"


def dtype_from_str(dtype_str: str) -> "torch.dtype":
    if not dtype_str.startswith("torch."):
        raise ValueError(f"not a torch dtype string: {dtype_str}")
    name = dtype_str.split(".", 1)[1]
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unknown torch dtype string: {dtype_str}")
    return dtype


def stage_to_cpu(tensor: "torch.Tensor", pinned: bool = True) -> "torch.Tensor":
    """Return a contiguous CPU copy of `tensor`.

    CUDA tensors are copied via a pinned staging buffer with
    non_blocking=True; call synchronize_staging() once after issuing all
    the copies you need for one checkpoint, not after each individual one,
    so the D2H transfers can overlap.
    """
    if tensor.device.type == "cpu":
        return tensor.detach().contiguous()
    staging = torch.empty(tuple(tensor.shape), dtype=tensor.dtype, device="cpu", pin_memory=pinned)
    staging.copy_(tensor.detach(), non_blocking=pinned)
    return staging


def synchronize_staging(device: "torch.device") -> None:
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def describe_torch_tensor(tensor: "torch.Tensor") -> Tuple[str, List[int], memoryview]:
    """(dtype_str, shape, zero-copy byte view). `tensor` must already be on
    CPU and will be made contiguous if it isn't (stage_to_cpu handles the
    CUDA case upstream of this call).
    """
    if tensor.device.type != "cpu":
        raise ValueError("describe_torch_tensor requires a CPU tensor; call stage_to_cpu() first")
    tensor = tensor.contiguous()
    dtype_str = torch_dtype_str(tensor.dtype)
    shape = list(tensor.shape)
    # Flatten before reinterpreting as uint8: view() resizes the *last*
    # dimension to account for the different element size, but a 0-dim
    # tensor (e.g. the scalar `step` counter PyTorch's AdamW state has
    # stored as a tensor since 2.1+) has no last dimension to resize, so
    # view() on it raises. reshape(-1) is a free view (tensor is already
    # contiguous) and works uniformly regardless of the original rank;
    # `shape` above still records the true shape for reconstruction.
    flat = tensor.reshape(-1)
    byte_view = flat if flat.dtype == torch.uint8 else flat.view(torch.uint8)
    buf = memoryview(byte_view.numpy())
    return dtype_str, shape, buf


def reconstruct_torch_tensor(dtype_str: str, shape: List[int], raw: bytes) -> "torch.Tensor":
    dtype = dtype_from_str(dtype_str)
    byte_tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    typed = byte_tensor if dtype == torch.uint8 else byte_tensor.view(dtype)
    return typed.reshape(shape).clone()


def reconstruct_tensor(dtype_str: str, shape: List[int], raw: bytes) -> Any:
    """Rebuilds a saved tensor of either kind the checkpoint stores: a torch.Tensor for
    "torch.*" dtypes, a numpy array for "numpy.*" ones (numpy's own RNG state is saved
    as a uint32 array, which torch has no dtype for). Resume code must use this, not
    reconstruct_torch_tensor, for anything that was not guaranteed to be a torch tensor.
    """
    if dtype_str.startswith("torch."):
        return reconstruct_torch_tensor(dtype_str, shape, raw)
    from pipeline.checkpoint import serializer

    return serializer.reconstruct_numpy(dtype_str, shape, raw).copy()


def is_torch_tensor(value: Any) -> bool:
    return type(value).__module__.split(".")[0] == "torch" and isinstance(value, torch.Tensor)
