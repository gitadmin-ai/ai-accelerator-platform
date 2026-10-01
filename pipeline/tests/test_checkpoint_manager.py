import numpy as np
import pytest

pytest.importorskip("eclib_blobstore", reason="eclib_blobstore native extension not built")

from pipeline.checkpoint.exceptions import CheckpointIncompleteError, CheckpointNotFoundError
from pipeline.checkpoint.manager import BlobStoreCheckpointManager
from pipeline.utils import seed as seed_utils


def _rng():
    return np.random.RandomState(0)


def _lora_model_state(rng):
    state = {}
    for layer in range(2):
        prefix = f"base_model.model.layers.{layer}.self_attn.q_proj"
        state[f"{prefix}.lora_A.weight"] = rng.randn(8, 64).astype(np.float32)
        state[f"{prefix}.lora_B.weight"] = rng.randn(64, 8).astype(np.float32)
    return state


def _optimizer_state(model_state, rng):
    state = {}
    for i, name in enumerate(model_state):
        state[i] = {
            "step": i + 1,
            "exp_avg": rng.randn(*model_state[name].shape).astype(np.float32),
            "exp_avg_sq": np.abs(rng.randn(*model_state[name].shape)).astype(np.float32),
        }
    return {"state": state, "param_groups": [{"lr": 2e-4, "weight_decay": 0.0}]}


def _scheduler_state():
    return {"base_lrs": [2e-4], "last_epoch": 3, "_step_count": 30}


def _training_state(epoch, global_step):
    return {
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": 0.42,
        "rng": seed_utils.capture_rng_state(),
    }


def _assert_deep_equal(a, b, path=""):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        assert np.array_equal(a, b), f"array mismatch at {path}"
        return
    if isinstance(a, dict):
        assert isinstance(b, dict), f"type mismatch at {path}"
        assert set(a) == set(b), f"key mismatch at {path}: {set(a)} vs {set(b)}"
        for k in a:
            _assert_deep_equal(a[k], b[k], f"{path}.{k}")
        return
    if isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)), f"type mismatch at {path}"
        assert len(a) == len(b), f"length mismatch at {path}"
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_deep_equal(x, y, f"{path}[{i}]")
        return
    assert a == b, f"value mismatch at {path}: {a!r} != {b!r}"


@pytest.mark.parametrize("num_workers", [1, 2, 4])
def test_save_load_round_trip(num_workers):
    rng = _rng()
    model_state = _lora_model_state(rng)
    optimizer_state = _optimizer_state(model_state, rng)
    scheduler_state = _scheduler_state()
    training_state = _training_state(epoch=2, global_step=160)

    mgr = BlobStoreCheckpointManager(
        run_id=f"run-{num_workers}",
        num_workers=num_workers,
        chunk_size_bytes=1024,
        config_overrides=dict(num_data_devices=10, device_size_mb=32),
    )
    try:
        manifest, metrics = mgr.save_checkpoint(
            "epoch-2-step-160",
            model_state,
            optimizer_state,
            scheduler_state,
            training_state,
            epoch=2,
            global_step=160,
            model_name_or_path="/models/qwen-tiny",
            base_model_id="Qwen/Qwen2.5-0.5B",
            dataset_id="onboarding-tutor-v1",
            lora_config={"r": 8, "lora_alpha": 16, "target_modules": ["q_proj"]},
            training_config={"learning_rate": 2e-4, "per_device_train_batch_size": 2},
        )
        assert manifest.status == "COMPLETE"
        assert metrics.num_chunks >= metrics.num_tensors
        assert metrics.size_bytes > 0
        assert metrics.total_s >= 0
        assert metrics.num_workers == num_workers

        assert mgr.get_latest_checkpoint() == "epoch-2-step-160"
        assert any(e["checkpoint_id"] == "epoch-2-step-160" for e in mgr.list_checkpoints())

        loaded = mgr.load_checkpoint("epoch-2-step-160")
        restored = mgr.materialize_numpy_state(loaded)

        for name, arr in model_state.items():
            assert np.array_equal(restored["model"][name], arr)

        _assert_deep_equal(restored["scheduler"], scheduler_state)
        assert restored["training_state"]["epoch"] == 2
        assert restored["training_state"]["global_step"] == 160
        assert restored["training_state"]["best_metric"] == pytest.approx(0.42)
        _assert_deep_equal(restored["training_state"]["rng"]["python"], list(training_state["rng"]["python"]))

        report = mgr.verify("epoch-2-step-160")
        assert report.ok
        assert report.chunks_checked > 0
        assert report.issues == []
    finally:
        mgr.shutdown()


def test_tensor_larger_than_chunk_size_reconstructs_correctly():
    rng = _rng()
    big = rng.randn(2000).astype(np.float32)  # 8000 bytes
    model_state = {"big_tensor": big}

    mgr = BlobStoreCheckpointManager(
        run_id="run-bigtensor",
        num_workers=3,
        chunk_size_bytes=37,  # deliberately awkward, not a divisor of 8000
        config_overrides=dict(num_data_devices=10, device_size_mb=32),
    )
    try:
        manifest, _ = mgr.save_checkpoint(
            "ckpt-big",
            model_state,
            None,
            None,
            {"epoch": 0, "global_step": 0},
            epoch=0,
            global_step=0,
            model_name_or_path="m",
            base_model_id="m",
            dataset_id="d",
            lora_config={},
            training_config={},
        )
        assert len(manifest.model_tensors[0].chunks) > 1

        loaded = mgr.load_checkpoint("ckpt-big")
        restored = mgr.materialize_numpy_state(loaded)
        assert np.array_equal(restored["model"]["big_tensor"], big)
    finally:
        mgr.shutdown()


def test_load_missing_checkpoint_raises():
    mgr = BlobStoreCheckpointManager(
        run_id="run-missing", num_workers=1, config_overrides=dict(num_data_devices=10, device_size_mb=32)
    )
    try:
        with pytest.raises(CheckpointNotFoundError):
            mgr.load_checkpoint("does-not-exist")
    finally:
        mgr.shutdown()


def test_get_latest_checkpoint_none_when_empty():
    mgr = BlobStoreCheckpointManager(
        run_id="run-empty", num_workers=1, config_overrides=dict(num_data_devices=10, device_size_mb=32)
    )
    try:
        assert mgr.get_latest_checkpoint() is None
        assert mgr.list_checkpoints() == []
    finally:
        mgr.shutdown()


def test_get_latest_checkpoint_picks_highest_epoch_step():
    mgr = BlobStoreCheckpointManager(
        run_id="run-latest", num_workers=2, config_overrides=dict(num_data_devices=10, device_size_mb=32)
    )
    try:
        for epoch, step in [(0, 10), (0, 20), (1, 5)]:
            mgr.save_checkpoint(
                f"epoch-{epoch}-step-{step}",
                {"w": np.ones(4, dtype=np.float32)},
                None,
                None,
                {"epoch": epoch, "global_step": step},
                epoch=epoch,
                global_step=step,
                model_name_or_path="m",
                base_model_id="m",
                dataset_id="d",
                lora_config={},
                training_config={},
            )
        assert mgr.get_latest_checkpoint() == "epoch-1-step-5"
        assert len(mgr.list_checkpoints()) == 3
    finally:
        mgr.shutdown()
