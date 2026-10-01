"""BlobStoreCheckpointManager.shutdown() must close every shard.

Backends that hold a connection (DDL3) rely on this: an unclosed connection
keeps its connection_id claimed on the target, so a second manager in the same
process reusing that id never completes its handshake. Uses a fake in-memory
backend, so no native extension is needed.
"""
import pytest

from pipeline.checkpoint.manager import BlobStoreCheckpointManager
from pipeline.checkpoint.storage_backend import StorageBackend, StorageShard


class _FakeShard(StorageShard):
    def __init__(self, fail_close=False):
        self.closed = 0
        self._fail_close = fail_close
        self._blobs = {}

    def put(self, blob_id, buf, policy="auto"):
        self._blobs[blob_id] = bytes(buf)
        return {"size": len(self._blobs[blob_id])}

    def get(self, blob_id, length):
        return self._blobs[blob_id][:length]

    def exists(self, blob_id):
        return blob_id in self._blobs

    def head(self, blob_id):
        return {"size": len(self._blobs[blob_id])}

    def delete(self, blob_id):
        del self._blobs[blob_id]

    def close(self):
        self.closed += 1
        if self._fail_close:
            raise OSError("close failed")


class _FakeBackend(StorageBackend):
    def __init__(self, fail_close_at=()):
        self.shards = []
        self._fail_close_at = fail_close_at

    def create_shards(self, num_shards):
        self.shards = [_FakeShard(fail_close=i in self._fail_close_at) for i in range(num_shards)]
        return self.shards


def test_shutdown_closes_every_shard():
    backend = _FakeBackend()
    mgr = BlobStoreCheckpointManager("run", storage=backend, num_workers=3)
    mgr.shutdown()
    assert [s.closed for s in backend.shards] == [1, 1, 1]


def test_shutdown_is_idempotent():
    backend = _FakeBackend()
    mgr = BlobStoreCheckpointManager("run", storage=backend, num_workers=2)
    mgr.shutdown()
    mgr.shutdown()
    assert [s.closed for s in backend.shards] == [1, 1]


def test_shutdown_closes_remaining_shards_when_one_fails_then_raises():
    backend = _FakeBackend(fail_close_at=(0,))
    mgr = BlobStoreCheckpointManager("run", storage=backend, num_workers=3)
    with pytest.raises(OSError, match="close failed"):
        mgr.shutdown()
    assert [s.closed for s in backend.shards] == [1, 1, 1]
    mgr.shutdown()  # already shut down: no second close, no second raise
    assert [s.closed for s in backend.shards] == [1, 1, 1]


def test_default_close_is_a_noop_for_backends_without_connections():
    class _Bare(_FakeShard):
        close = StorageShard.close

    assert _Bare().close() is None
