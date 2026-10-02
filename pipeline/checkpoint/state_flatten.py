"""Flatten a nested state dict (optimizer.state_dict(), scheduler.state_dict(),
RNG/grad-scaler bookkeeping) into:

    - a JSON-safe "skeleton" describing the structure, with each tensor
      leaf replaced by a {"__tensor__": "<name>"} marker
    - a flat {name: tensor} dict of the tensor leaves themselves

Tensor leaves are then serialized through the normal chunked tensor
pipeline (serializer.py / chunker.py) exactly like model weights, instead
of being pickled in place — an optimizer's Adam state ("exp_avg",
"exp_avg_sq" per parameter) or a generator's RNG state are tensors like any
other, and pickle is never touched.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

_TENSOR_KEY = "__tensor__"
_DICT_KEY = "__dict__"
_LIST_KEY = "__list__"
_TUPLE_KEY = "__tuple__"
_SCALAR_KEY = "__scalar__"

_SCALAR_TYPES = (int, float, str, bool, type(None))


def flatten_state_dict(
    obj: Any, is_tensor: Callable[[Any], bool], prefix: str = "root"
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Returns (skeleton, tensors)."""
    tensors: Dict[str, Any] = {}

    def walk(node: Any, path: str) -> Dict[str, Any]:
        if is_tensor(node):
            tensors[path] = node
            return {_TENSOR_KEY: path}
        if isinstance(node, dict):
            return {_DICT_KEY: {str(k): walk(v, f"{path}.{k}") for k, v in node.items()}}
        if isinstance(node, list):
            return {_LIST_KEY: [walk(v, f"{path}.{i}") for i, v in enumerate(node)]}
        if isinstance(node, tuple):
            return {_TUPLE_KEY: [walk(v, f"{path}.{i}") for i, v in enumerate(node)]}
        if isinstance(node, _SCALAR_TYPES):
            return {_SCALAR_KEY: node}
        raise TypeError(f"state_flatten: unsupported leaf type {type(node)} at {path}")

    return walk(obj, prefix), tensors


def unflatten_state_dict(skeleton: Dict[str, Any], tensors: Dict[str, Any]) -> Any:
    def walk(node: Dict[str, Any]) -> Any:
        if _TENSOR_KEY in node:
            return tensors[node[_TENSOR_KEY]]
        if _DICT_KEY in node:
            return {k: walk(v) for k, v in node[_DICT_KEY].items()}
        if _LIST_KEY in node:
            return [walk(v) for v in node[_LIST_KEY]]
        if _TUPLE_KEY in node:
            return tuple(walk(v) for v in node[_TUPLE_KEY])
        if _SCALAR_KEY in node:
            return node[_SCALAR_KEY]
        raise ValueError(f"state_flatten: corrupt skeleton node {node!r}")

    return walk(skeleton)


def restore_optimizer_state_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """flatten_state_dict stringifies every dict key (the skeleton is JSON), but a torch
    optimizer's state_dict() keys its per-parameter "state" by integer parameter id, and
    load_state_dict() looks those ids up as ints -- with string keys the saved state
    (Adam's exp_avg/exp_avg_sq/step) matches nothing and is silently lost. Turns the
    all-digit keys of state_dict["state"] back into ints."""
    state = state_dict.get("state")
    if not isinstance(state, dict):
        return state_dict
    fixed = {int(k) if isinstance(k, str) and k.isdigit() else k: v for k, v in state.items()}
    return {**state_dict, "state": fixed}
