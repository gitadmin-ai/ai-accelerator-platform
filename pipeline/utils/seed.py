"""RNG state capture/restore for correct, reproducible resume.

Captures python `random`, numpy, and (if torch is importable) torch CPU +
every visible CUDA device's generator state. The numpy state tuple embeds
an ndarray, and torch's states are themselves tensors (ByteTensor) -- both
flow through checkpoint/state_flatten.py's generic tensor detection like
any other tensor leaf, so they get the same chunked-and-checksummed
treatment as model weights rather than a separate ad-hoc encoding.
"""
from __future__ import annotations

import random
from typing import Any, Dict

import numpy as np


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    try:
        import torch

        state["torch_cpu"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = [
                torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        import torch

        torch.set_rng_state(state["torch_cpu"])
        if "torch_cuda" in state and torch.cuda.is_available():
            for i, s in enumerate(state["torch_cuda"]):
                torch.cuda.set_rng_state(s, i)
