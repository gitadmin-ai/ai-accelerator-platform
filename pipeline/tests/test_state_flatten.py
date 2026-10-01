import numpy as np

from pipeline.checkpoint import state_flatten
from pipeline.checkpoint.tensor_io import is_tensor_like


def test_flatten_unflatten_scalars_only():
    obj = {"a": 1, "b": [1, 2, 3], "c": (1.0, "x", None, True)}
    skeleton, tensors = state_flatten.flatten_state_dict(obj, is_tensor_like)
    assert tensors == {}
    restored = state_flatten.unflatten_state_dict(skeleton, tensors)
    assert restored == obj
    assert isinstance(restored["c"], tuple)


def test_flatten_unflatten_with_tensor_leaves():
    obj = {
        "state": {
            0: {"step": 5, "exp_avg": np.ones(4, dtype=np.float32), "exp_avg_sq": np.zeros(4)},
            1: {"step": 3, "exp_avg": np.full(4, 2.0, dtype=np.float32)},
        },
        "param_groups": [{"lr": 0.001, "weight_decay": 0.0}],
    }
    skeleton, tensors = state_flatten.flatten_state_dict(obj, is_tensor_like, prefix="optimizer")
    assert len(tensors) == 3
    for name, t in tensors.items():
        assert isinstance(t, np.ndarray)

    restored = state_flatten.unflatten_state_dict(skeleton, tensors)
    assert restored["param_groups"] == obj["param_groups"]
    assert np.array_equal(restored["state"]["0"]["exp_avg"], obj["state"][0]["exp_avg"])
    assert restored["state"]["0"]["step"] == 5
    assert np.array_equal(restored["state"]["1"]["exp_avg"], obj["state"][1]["exp_avg"])


def test_flatten_survives_json_round_trip():
    import json

    obj = {"nested": {"tensor": np.arange(3, dtype=np.float32), "flag": True}}
    skeleton, tensors = state_flatten.flatten_state_dict(obj, is_tensor_like)
    skeleton_json = json.loads(json.dumps(skeleton))
    restored = state_flatten.unflatten_state_dict(skeleton_json, tensors)
    assert np.array_equal(restored["nested"]["tensor"], obj["nested"]["tensor"])
    assert restored["nested"]["flag"] is True


def test_flatten_tuple_of_arrays_like_numpy_rng_state():
    # mirrors what np.random.get_state() returns
    rng_state = ("MT19937", np.zeros(624, dtype=np.uint32), 624, 0, 0.0)
    skeleton, tensors = state_flatten.flatten_state_dict({"numpy": rng_state}, is_tensor_like)
    assert len(tensors) == 1
    restored = state_flatten.unflatten_state_dict(skeleton, tensors)
    assert isinstance(restored["numpy"], tuple)
    assert restored["numpy"][0] == "MT19937"
    assert np.array_equal(restored["numpy"][1], rng_state[1])
    assert restored["numpy"][2:] == rng_state[2:]


def test_flatten_rejects_unsupported_leaf_type():
    import pytest

    class Weird:
        pass

    with pytest.raises(TypeError):
        state_flatten.flatten_state_dict({"x": Weird()}, is_tensor_like)
