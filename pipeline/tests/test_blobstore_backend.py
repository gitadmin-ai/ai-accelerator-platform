import numpy as np
import pytest

pytest.importorskip("eclib_blobstore", reason="eclib_blobstore native extension not built")

from pipeline.checkpoint.blobstore_backend import BlobStoreError, BlobStoreShard, create_shards


def _shard(**overrides):
    # small device pool / stripe size so tests run fast and deterministically
    return BlobStoreShard(
        num_data_devices=10,
        device_size_mb=32,
        small_io_replica_device_ids=[0, 1],
        index_device_ids=[8, 9],
        **overrides,
    )


def test_put_get_bytes_roundtrip():
    shard = _shard()
    shard.put("k1", b"hello world")
    assert bytes(shard.get("k1", 11)) == b"hello world"


def test_put_get_numpy_array_roundtrip():
    shard = _shard()
    arr = np.arange(50000, dtype=np.float32)
    shard.put("tensor", arr)
    out = np.frombuffer(shard.get("tensor", arr.nbytes), dtype=np.float32)
    assert np.array_equal(out, arr)


def test_put_get_memoryview_slice_roundtrip():
    shard = _shard()
    arr = np.arange(1000, dtype=np.uint8)
    mv = memoryview(arr)[100:200]
    shard.put("slice", mv)
    assert bytes(shard.get("slice", 100)) == bytes(arr[100:200])


def test_exists_and_delete():
    shard = _shard()
    assert not shard.exists("missing")
    shard.put("present", b"x" * 10)
    assert shard.exists("present")
    shard.delete("present")
    assert not shard.exists("present")


def test_get_missing_blob_raises():
    shard = _shard()
    with pytest.raises(BlobStoreError):
        shard.get("does-not-exist", 10)


def test_head_blob_reports_size_and_flags():
    shard = _shard()
    payload = b"y" * 5000  # above inline threshold
    shard.put("headme", payload)
    info = shard.head("headme")
    assert info["size"] == len(payload)
    assert info["blob_id"] == "headme"


def test_shards_are_independent_stores():
    shards = create_shards(3, num_data_devices=10, device_size_mb=32)
    shards[0].put("only-on-0", b"abc")
    assert shards[0].exists("only-on-0")
    assert not shards[1].exists("only-on-0")
    assert not shards[2].exists("only-on-0")


def test_device_failure_and_recovery_still_serves_reads():
    shard = _shard()
    payload = b"z" * 5000
    shard.put("resilient", payload)
    shard.fail_device(0)
    # replica/EC protection should still allow a correct read
    assert bytes(shard.get("resilient", len(payload))) == payload
    shard.recover_device(0)
    assert bytes(shard.get("resilient", len(payload))) == payload


def test_stats_reflect_operations():
    shard = _shard()
    shard.put("s1", b"a" * 10)
    shard.get("s1", 10)
    stats = shard.stats()
    assert stats["total_puts"] >= 1
    assert stats["total_gets"] >= 1
