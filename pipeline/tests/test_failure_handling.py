import json

import numpy as np
import pytest

pytest.importorskip("eclib_blobstore", reason="eclib_blobstore native extension not built")

from pipeline.checkpoint.blobstore_backend import BlobStoreError
from pipeline.checkpoint.exceptions import CheckpointCorruptError, CheckpointIncompleteError, CheckpointWriteError
from pipeline.checkpoint.format import STATUS_INCOMPLETE, manifest_blob_id
from pipeline.checkpoint.manager import BlobStoreCheckpointManager


def _make_manager(run_id, num_workers=2):
    return BlobStoreCheckpointManager(
        run_id=run_id,
        num_workers=num_workers,
        chunk_size_bytes=256,
        config_overrides=dict(num_data_devices=10, device_size_mb=32),
    )


def _save_simple(mgr, checkpoint_id, epoch=0, step=0):
    return mgr.save_checkpoint(
        checkpoint_id,
        {"w": np.arange(200, dtype=np.float32)},
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


def test_verify_detects_corrupted_chunk():
    mgr = _make_manager("run-corrupt")
    try:
        manifest, _ = _save_simple(mgr, "ckpt-1")
        chunk = manifest.model_tensors[0].chunks[0]
        shard = mgr.shards[chunk.shard % mgr.num_workers]
        shard.put(chunk.blob_id, b"\xff" * chunk.length)  # corrupt in place

        report = mgr.verify("ckpt-1")
        assert not report.ok
        assert any(i.kind == "checksum_mismatch" for i in report.issues)
    finally:
        mgr.shutdown()


def test_verify_detects_missing_chunk():
    mgr = _make_manager("run-missing-chunk")
    try:
        manifest, _ = _save_simple(mgr, "ckpt-1")
        chunk = manifest.model_tensors[0].chunks[0]
        shard = mgr.shards[chunk.shard % mgr.num_workers]
        shard.delete(chunk.blob_id)

        report = mgr.verify("ckpt-1")
        assert not report.ok
        assert any(i.kind == "missing_chunk" for i in report.issues)
    finally:
        mgr.shutdown()


def test_load_raises_on_corrupted_chunk():
    mgr = _make_manager("run-load-corrupt")
    try:
        manifest, _ = _save_simple(mgr, "ckpt-1")
        chunk = manifest.model_tensors[0].chunks[0]
        shard = mgr.shards[chunk.shard % mgr.num_workers]
        shard.put(chunk.blob_id, b"\xff" * chunk.length)

        with pytest.raises(CheckpointCorruptError):
            mgr.load_checkpoint("ckpt-1")
    finally:
        mgr.shutdown()


def test_load_raises_on_missing_chunk():
    mgr = _make_manager("run-load-missing")
    try:
        manifest, _ = _save_simple(mgr, "ckpt-1")
        chunk = manifest.model_tensors[0].chunks[0]
        shard = mgr.shards[chunk.shard % mgr.num_workers]
        shard.delete(chunk.blob_id)

        with pytest.raises(BlobStoreError):
            mgr.load_checkpoint("ckpt-1")
    finally:
        mgr.shutdown()


def test_write_failure_prevents_commit(monkeypatch):
    mgr = _make_manager("run-write-fail")
    try:
        real_put = mgr.shards[1].put
        calls = {"n": 0}

        def flaky_put(blob_id, buf, policy="auto"):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("simulated BlobStore write failure")
            return real_put(blob_id, buf, policy)

        monkeypatch.setattr(mgr.shards[1], "put", flaky_put)

        with pytest.raises(CheckpointWriteError):
            _save_simple(mgr, "ckpt-should-not-exist")

        # the failed checkpoint must never be discoverable
        assert mgr.get_latest_checkpoint() is None
        assert mgr.list_checkpoints() == []
        assert mgr.list_checkpoints(include_incomplete=True) == []
    finally:
        mgr.shutdown()


def test_incomplete_checkpoint_hidden_from_latest_but_loadable_explicitly():
    mgr = _make_manager("run-incomplete")
    try:
        manifest, _ = _save_simple(mgr, "ckpt-real", epoch=1, step=1)

        # Hand-craft an INCOMPLETE manifest+catalog entry to simulate a
        # process interruption between writing chunks and committing --
        # this must not be returned by get_latest_checkpoint()/
        # list_checkpoints() unless explicitly requested.
        incomplete = manifest
        incomplete.checkpoint_id = "ckpt-crashed"
        incomplete.status = STATUS_INCOMPLETE
        mgr.shards[0].put(manifest_blob_id("ckpt-crashed"), incomplete.to_json())
        mgr._update_catalog("ckpt-crashed", STATUS_INCOMPLETE, 2, 2, 9999.0)

        assert mgr.get_latest_checkpoint() == "ckpt-real"
        assert all(e["checkpoint_id"] != "ckpt-crashed" for e in mgr.list_checkpoints())
        assert any(e["checkpoint_id"] == "ckpt-crashed" for e in mgr.list_checkpoints(include_incomplete=True))

        with pytest.raises(CheckpointIncompleteError):
            mgr.load_checkpoint("ckpt-crashed")

        # explicit override still works
        loaded = mgr.load_checkpoint("ckpt-crashed", allow_incomplete=True)
        assert loaded.manifest.status == STATUS_INCOMPLETE
    finally:
        mgr.shutdown()


def test_invalid_manifest_json_raises():
    mgr = _make_manager("run-invalid-manifest")
    try:
        mgr.shards[0].put(manifest_blob_id("garbage"), b"{not valid json")
        with pytest.raises(json.JSONDecodeError):
            mgr.load_checkpoint("garbage")
    finally:
        mgr.shutdown()
