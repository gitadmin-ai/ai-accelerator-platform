import numpy as np
import pytest

from pipeline.checkpoint.localfs_backend import LocalBlobNotFoundError, LocalFsBackend, LocalFsShard
from pipeline.checkpoint.storage_backend import StorageError


def _shard(tmp_path):
    return LocalFsShard(tmp_path / "shard-0")


def test_put_get_bytes_roundtrip(tmp_path):
    shard = _shard(tmp_path)
    shard.put("k1", b"hello world")
    assert bytes(shard.get("k1", 11)) == b"hello world"


def test_put_get_numpy_array_roundtrip(tmp_path):
    shard = _shard(tmp_path)
    arr = np.arange(50000, dtype=np.float32)
    shard.put("tensor", arr)
    out = np.frombuffer(shard.get("tensor", arr.nbytes), dtype=np.float32)
    assert np.array_equal(out, arr)


def test_put_get_memoryview_slice_roundtrip(tmp_path):
    shard = _shard(tmp_path)
    arr = np.arange(1000, dtype=np.uint8)
    mv = memoryview(arr)[100:200]
    shard.put("slice", mv)
    assert bytes(shard.get("slice", 100)) == bytes(arr[100:200])


def test_exists_and_delete(tmp_path):
    shard = _shard(tmp_path)
    assert not shard.exists("missing")
    shard.put("present", b"x" * 10)
    assert shard.exists("present")
    shard.delete("present")
    assert not shard.exists("present")


def test_get_missing_blob_raises(tmp_path):
    shard = _shard(tmp_path)
    with pytest.raises(StorageError):
        shard.get("does-not-exist", 10)
    with pytest.raises(LocalBlobNotFoundError):
        shard.get("does-not-exist", 10)


def test_head_blob_reports_size(tmp_path):
    shard = _shard(tmp_path)
    payload = b"y" * 5000
    shard.put("headme", payload)
    info = shard.head("headme")
    assert info["size"] == len(payload)


def test_shards_are_independent_dirs(tmp_path):
    shards = LocalFsBackend(tmp_path).create_shards(3)
    shards[0].put("only-on-0", b"abc")
    assert shards[0].exists("only-on-0")
    assert not shards[1].exists("only-on-0")
    assert not shards[2].exists("only-on-0")


def test_nested_blob_id_maps_to_nested_path(tmp_path):
    shard = _shard(tmp_path)
    shard.put("ckpt/run-1/model/layer.0.weight/chunk_000000", b"payload")
    assert bytes(shard.get("ckpt/run-1/model/layer.0.weight/chunk_000000", 7)) == b"payload"


def test_rejects_path_traversal(tmp_path):
    shard = _shard(tmp_path)
    with pytest.raises(ValueError):
        shard.put("../escape", b"x")
    with pytest.raises(ValueError):
        shard.get("a/../../escape", 1)


def test_put_overwrites_existing_blob(tmp_path):
    shard = _shard(tmp_path)
    shard.put("k", b"first")
    shard.put("k", b"second-value")
    assert bytes(shard.get("k", len(b"second-value"))) == b"second-value"
