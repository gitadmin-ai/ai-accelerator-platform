"""Resume state that needs torch (skipped where torch is not installed): torch/numpy RNG
state and AdamW state round-tripping through a checkpoint into a fresh process."""
import random

import numpy as np
import pytest

from pipeline.checkpoint import state_flatten
from pipeline.checkpoint.manager import BlobStoreCheckpointManager
from pipeline.tests.test_checkpoint_packing import _MemBackend
from pipeline.utils import seed as seed_utils

torch = pytest.importorskip("torch", reason="torch not installed")


def _manager():
    return BlobStoreCheckpointManager("run", storage=_MemBackend(), num_workers=1, chunk_size_bytes=1 << 20)


def test_reconstruct_tensor_dispatches_on_the_dtype_string():
    from pipeline.checkpoint import tensor_adapter

    t = tensor_adapter.reconstruct_tensor("torch.float32", [2], np.array([1.5, 2.5], dtype="float32").tobytes())
    assert isinstance(t, torch.Tensor) and t.tolist() == [1.5, 2.5]
    a = tensor_adapter.reconstruct_tensor("numpy.uint32", [3], np.array([1, 2, 3], dtype="uint32").tobytes())
    assert isinstance(a, np.ndarray) and a.dtype == np.uint32 and a.tolist() == [1, 2, 3]
    with pytest.raises(ValueError):
        tensor_adapter.reconstruct_tensor("float32", [1], b"\0\0\0\0")


def test_full_rng_state_with_torch_survives_a_checkpoint():
    from pipeline.checkpoint import tensor_adapter

    seed_utils.set_all_seeds(123) if hasattr(seed_utils, "set_all_seeds") else (random.seed(1), np.random.seed(1), torch.manual_seed(1))
    mgr = _manager()
    mgr.save_checkpoint("c1", {}, None, None, {"rng": seed_utils.capture_rng_state()}, epoch=1, global_step=1,
                        model_name_or_path="m", base_model_id="b", dataset_id="d", lora_config={}, training_config={})
    expected = (random.random(), float(np.random.random()), float(torch.rand(1)))
    loaded = mgr.load_checkpoint("c1")
    tensors = {n: tensor_adapter.reconstruct_tensor(*v) for n, v in loaded.training_state_tensors_raw.items()}
    back = state_flatten.unflatten_state_dict(loaded.training_state_skeleton, tensors)
    random.seed(99); np.random.seed(99); torch.manual_seed(99)
    seed_utils.restore_rng_state(back["rng"])
    assert (random.random(), float(np.random.random()), float(torch.rand(1))) == expected


def _adamw_after_steps():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(3):
        opt.zero_grad(); model(torch.randn(8, 4)).sum().backward(); opt.step()
    return model, opt


def test_adamw_state_survives_a_checkpoint_and_loads_into_a_fresh_optimizer():
    from pipeline.checkpoint import tensor_adapter, tensor_io

    model, opt = _adamw_after_steps()
    skeleton, flat = state_flatten.flatten_state_dict(opt.state_dict(), tensor_io.is_tensor_like, prefix="optimizer")
    mgr = _manager()
    mgr.save_checkpoint("c1", {}, opt.state_dict(), None, {}, epoch=1, global_step=3, model_name_or_path="m",
                        base_model_id="b", dataset_id="d", lora_config={}, training_config={})
    loaded = mgr.load_checkpoint("c1")
    tensors = {n: tensor_adapter.reconstruct_tensor(*v) for n, v in loaded.optimizer_tensors_raw.items()}
    restored = state_flatten.unflatten_state_dict(loaded.optimizer_skeleton, tensors)

    fresh = torch.optim.AdamW(model.parameters(), lr=1e-2)
    fresh.load_state_dict(state_flatten.restore_optimizer_state_keys(restored))
    assert len(fresh.state) == len(opt.state) == 2  # weight and bias
    for p in model.parameters():
        for key in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(fresh.state[p][key], opt.state[p][key]), key

    # Why restore_optimizer_state_keys exists. Without it torch does not error: it keeps the
    # saved state under the string keys "0"/"1", which match no parameter, so every parameter
    # restarts with empty Adam state and the saved state is silently orphaned.
    unfixed = torch.optim.AdamW(model.parameters(), lr=1e-2)
    unfixed.load_state_dict(restored)
    assert all(p not in unfixed.state for p in model.parameters())
