"""What a resume has to get back from a checkpoint besides the model weights: the
optimizer's state and the RNG state, each of which broke the real resume path."""
import random

import numpy as np
import pytest

from pipeline.checkpoint import state_flatten
from pipeline.checkpoint.manager import BlobStoreCheckpointManager
from pipeline.tests.test_checkpoint_packing import _MemBackend
from pipeline.utils import seed as seed_utils


def _manager():
    return BlobStoreCheckpointManager("run", storage=_MemBackend(), num_workers=1, chunk_size_bytes=1 << 20)


def test_restore_optimizer_state_keys_turns_digit_keys_back_into_ints():
    flat = {"state": {"0": {"step": 1}, "12": {"step": 2}}, "param_groups": [{"params": [0, 12]}]}
    fixed = state_flatten.restore_optimizer_state_keys(flat)
    assert set(fixed["state"]) == {0, 12} and fixed["param_groups"] == flat["param_groups"]
    assert set(flat["state"]) == {"0", "12"}  # input untouched


def test_restore_optimizer_state_keys_leaves_other_shapes_alone():
    assert state_flatten.restore_optimizer_state_keys({"param_groups": []}) == {"param_groups": []}
    named = {"state": {"a": 1}}
    assert state_flatten.restore_optimizer_state_keys(named) == named


def test_numpy_rng_state_round_trips_without_torch():
    # The numpy generator's state is a uint32 array: the part torch cannot represent.
    from pipeline.checkpoint import tensor_io

    random.seed(5); np.random.seed(5)
    state = {"python": random.getstate(), "numpy": np.random.get_state()}
    expected = (random.random(), np.random.random())
    mgr = _manager()
    mgr.save_checkpoint("c1", {}, None, None, state, epoch=1, global_step=1, model_name_or_path="m",
                        base_model_id="b", dataset_id="d", lora_config={}, training_config={})
    loaded = mgr.load_checkpoint("c1")
    tensors = {n: tensor_io.reconstruct_tensor(*v) for n, v in loaded.training_state_tensors_raw.items()}
    back = state_flatten.unflatten_state_dict(loaded.training_state_skeleton, tensors)
    random.seed(0); np.random.seed(0)  # scramble, then restore
    seed_utils.restore_rng_state(back)
    assert (random.random(), np.random.random()) == expected
