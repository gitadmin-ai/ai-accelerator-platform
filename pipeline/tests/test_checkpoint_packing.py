"""Small chunks are packed together into shared stored objects.

Runs on an in-memory backend (no native extension), which also lets the tests
count stored objects and tamper with their bytes.
"""
import json

import numpy as np
import pytest

from pipeline.checkpoint import validation
from pipeline.checkpoint.exceptions import CheckpointCorruptError
from pipeline.checkpoint.format import ChunkRecord, Manifest, TensorRecord
from pipeline.checkpoint.manager import BlobStoreCheckpointManager
from pipeline.checkpoint.storage_backend import StorageBackend, StorageError, StorageShard

MiB = 1 << 20


class _MemShard(StorageShard):
    def __init__(self, store):
        self.store = store

    def put(self, blob_id, buf, policy="auto"):
        self.store[blob_id] = bytes(buf)
        return {"size": len(self.store[blob_id])}

    def get(self, blob_id, length):
        if blob_id not in self.store:
            raise StorageError(f"no such blob {blob_id!r}")
        return self.store[blob_id][:length]

    def exists(self, blob_id):
        return blob_id in self.store

    def head(self, blob_id):
        if blob_id not in self.store:
            raise StorageError(f"no such blob {blob_id!r}")
        return {"size": len(self.store[blob_id])}

    def delete(self, blob_id):
        del self.store[blob_id]


class _MemBackend(StorageBackend):
    def __init__(self):
        self.store = {}

    def create_shards(self, num_shards):
        return [_MemShard(self.store) for _ in range(num_shards)]


def _state(rng, n_small=60, big_floats=300_000):
    model = {f"layer{i}.lora_A.weight": rng.standard_normal((8, 96)).astype("float32") for i in range(n_small)}
    model["big.weight"] = rng.standard_normal(big_floats).astype("float32")  # ~1.2 MB: several chunks at 512 KiB
    model["empty"] = np.zeros((0,), dtype="float32")
    model["tiny.bias"] = np.arange(5, dtype="float32")
    return model


def _save(mgr, model, cid="c1"):
    return mgr.save_checkpoint(
        cid, model, {"m": model["tiny.bias"]}, None, {}, epoch=0, global_step=1,
        model_name_or_path="m", base_model_id="b", dataset_id="d", lora_config={}, training_config={},
    )


def _mgr(backend, **kw):
    kw.setdefault("chunk_size_bytes", 512 * 1024)
    kw.setdefault("num_workers", 2)
    return BlobStoreCheckpointManager("run", storage=backend, **kw)


def _blob_count(backend):
    return sum(1 for k in backend.store if "/pack_" in k or "/chunk_" in k)


def test_small_tensors_share_a_few_objects_and_round_trip():
    rng = np.random.default_rng(0)
    model = _state(rng)
    backend = _MemBackend()
    mgr = _mgr(backend)
    manifest, metrics = _save(mgr, model)

    chunk_records = sum(1 for _ in manifest.all_chunks())
    assert metrics.num_chunks == chunk_records
    assert metrics.num_blobs == _blob_count(backend)
    # 60 small tensors + tiny + empty (+ optimizer) collapse into a handful of packs
    # instead of one object each; only the big tensor's full chunks stay separate.
    assert metrics.num_blobs < chunk_records / 4
    assert any(c.packed for c in manifest.all_chunks())

    loaded = mgr.load_checkpoint("c1")
    state = mgr.materialize_numpy_state(loaded)
    for name, original in model.items():
        assert np.array_equal(np.asarray(state["model"][name]), original), name
    assert mgr.verify("c1").ok
    mgr.shutdown()


def test_packs_never_exceed_the_pack_size_and_cover_their_members_exactly():
    rng = np.random.default_rng(1)
    backend = _MemBackend()
    mgr = _mgr(backend, pack_size_bytes=64 * 1024)
    manifest, _ = _save(mgr, _state(rng))
    by_blob = {}
    for c in manifest.all_chunks():
        if c.packed:
            by_blob.setdefault(c.blob_id, []).append(c)
    assert by_blob
    for blob_id, members in by_blob.items():
        assert len(backend.store[blob_id]) == members[0].blob_length <= 64 * 1024
        spans = sorted((m.blob_offset, m.blob_offset + m.length) for m in members)
        assert spans[0][0] == 0 and spans[-1][1] == members[0].blob_length  # no gaps at the ends
        assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))  # contiguous, no overlap
    mgr.shutdown()


def test_large_chunks_are_still_stored_one_per_object():
    rng = np.random.default_rng(2)
    backend = _MemBackend()
    mgr = _mgr(backend)
    manifest, _ = _save(mgr, _state(rng))
    big = next(t for t in manifest.model_tensors if t.name == "big.weight")
    full = [c for c in big.chunks if c.length == mgr.chunk_size_bytes]
    assert full and all(not c.packed and "/chunk_" in c.blob_id for c in full)
    mgr.shutdown()


def test_pack_size_zero_disables_packing():
    rng = np.random.default_rng(3)
    backend = _MemBackend()
    mgr = _mgr(backend, pack_size_bytes=0)
    manifest, metrics = _save(mgr, _state(rng))
    assert not any(c.packed for c in manifest.all_chunks())
    assert metrics.num_blobs == metrics.num_chunks == _blob_count(backend)
    assert mgr.verify("c1").ok
    state = mgr.materialize_numpy_state(mgr.load_checkpoint("c1"))
    assert np.array_equal(np.asarray(state["model"]["tiny.bias"]), np.arange(5, dtype="float32"))
    mgr.shutdown()


def test_pack_size_is_capped_at_chunk_size():
    mgr = _mgr(_MemBackend(), chunk_size_bytes=100_000, pack_size_bytes=50 * MiB)
    assert mgr.pack_size_bytes == 100_000
    mgr.shutdown()


@pytest.mark.parametrize("verify_mode", ["checksum", "head", "none"])
def test_every_verify_mode_accepts_packed_checkpoints(verify_mode):
    rng = np.random.default_rng(4)
    mgr = _mgr(_MemBackend(), verify_mode=verify_mode)
    _save(mgr, _state(rng))
    assert mgr.verify("c1").ok
    mgr.shutdown()


def _first_pack(backend):
    return next(k for k in sorted(backend.store) if "/pack_" in k)


def test_corruption_inside_a_pack_is_detected_on_load_and_verify():
    rng = np.random.default_rng(5)
    backend = _MemBackend()
    mgr = _mgr(backend)
    _save(mgr, _state(rng))
    blob_id = _first_pack(backend)
    data = bytearray(backend.store[blob_id])
    data[len(data) // 2] ^= 0xFF
    backend.store[blob_id] = bytes(data)

    report = mgr.verify("c1")
    assert not report.ok
    assert any(i.kind == "checksum_mismatch" and i.name == blob_id for i in report.issues)
    with pytest.raises(CheckpointCorruptError):
        mgr.load_checkpoint("c1")
    mgr.shutdown()


def test_truncated_pack_is_reported_once_as_a_size_mismatch():
    rng = np.random.default_rng(6)
    backend = _MemBackend()
    mgr = _mgr(backend)
    _save(mgr, _state(rng))
    blob_id = _first_pack(backend)
    backend.store[blob_id] = backend.store[blob_id][:-10]
    report = mgr.verify("c1")
    issues = [i for i in report.issues if i.name == blob_id]
    assert len(issues) == 1 and issues[0].kind == "size_mismatch"
    mgr.shutdown()


def test_missing_pack_is_reported_once():
    rng = np.random.default_rng(7)
    backend = _MemBackend()
    mgr = _mgr(backend)
    _save(mgr, _state(rng))
    blob_id = _first_pack(backend)
    del backend.store[blob_id]
    issues = [i for i in mgr.verify("c1").issues if i.name == blob_id]
    assert len(issues) == 1 and issues[0].kind == "missing_chunk"
    mgr.shutdown()


def test_version_1_manifest_without_pack_fields_still_loads():
    legacy_chunk = {"index": 0, "blob_id": "b", "shard": 0, "offset": 0, "length": 4, "sha256": "x"}
    chunk = ChunkRecord.from_dict(legacy_chunk)
    assert not chunk.packed and chunk.blob_offset == 0 and chunk.stored_length == 4
    rec = TensorRecord.from_dict(
        {"name": "t", "dtype": "torch.float32", "shape": [1], "numel": 1, "byte_size": 4,
         "chunk_size": 4, "chunks": [legacy_chunk]}
    )
    assert not rec.chunks[0].packed


def test_unknown_future_format_version_is_refused():
    rng = np.random.default_rng(8)
    backend = _MemBackend()
    mgr = _mgr(backend)
    _save(mgr, _state(rng))
    key = "ckpt/c1/manifest.json"
    doc = json.loads(backend.store[key])
    doc["format_version"] = 99
    backend.store[key] = json.dumps(doc).encode()
    with pytest.raises(CheckpointCorruptError, match="format version 99"):
        mgr.load_checkpoint("c1")
    mgr.shutdown()
