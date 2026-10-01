"""Exercises DdlResourceStore's put_directory()/get_directory() round trip
against an in-memory fake shard instead of a real DDL3 target -- the native
ddl_client extension isn't available in this environment, but the
chunking/manifest/reassembly logic underneath it is pure Python and fully
testable with a fake that implements the same call(method, **kwargs) shape
DdlConnection exposes.
"""
import threading

import pytest

from nebula_ddl_storage.resource_store import DdlResourceNotFoundError, DdlResourceStore, _iter_chunks


class _FakeDdlError(Exception):
    """Mirrors ddl_client.DdlError: message plus a structured `.status`."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class _FakeShard:
    """In-memory stand-in for a DdlConnection: an (object_id_hi, object_id_lo)
    -> bytes dict guarded by a lock, since put_directory/get_directory
    dispatch concurrently across shards via a ThreadPoolExecutor.
    """

    def __init__(self, store: dict, lock: threading.Lock):
        self._store = store
        self._lock = lock

    def call(self, method, *, object_id_hi, object_id_lo, **kwargs):
        key = (object_id_hi, object_id_lo)
        if method == "put":
            with self._lock:
                self._store[key] = kwargs["data"]
            return None
        if method == "get":
            with self._lock:
                data = self._store.get(key)
            if data is None:
                raise _FakeDdlError("GET failed status=1 result_bytes=0", status=1)
            offset = kwargs.get("offset", 0)
            length = kwargs["length"]
            return bytes(data[offset:offset + length])
        raise AssertionError(f"unexpected method {method!r}")

    def close(self):
        pass


def _make_store(namespace="models", num_shards=3, max_chunk_bytes=1024 + 8):
    backing: dict = {}
    lock = threading.Lock()
    shards = [_FakeShard(backing, lock) for _ in range(num_shards)]
    return DdlResourceStore(namespace=namespace, num_shards=num_shards, max_chunk_bytes=max_chunk_bytes, shards=shards)


def test_iter_chunks_splits_and_handles_empty():
    chunks = _iter_chunks(b"a" * 2500, 1000)
    assert [c.length for c in chunks] == [1000, 1000, 500]
    assert b"".join(c.data for c in chunks) == b"a" * 2500

    empty = _iter_chunks(b"", 1000)
    assert len(empty) == 1
    assert empty[0].length == 0


def test_put_directory_then_get_directory_round_trips(tmp_path):
    store = _make_store()
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "config.json").write_text('{"hello": "world"}')
    (src / "sub" / "weights.bin").write_bytes(bytes(range(256)) * 20)  # 5120 bytes, spans multiple chunks

    manifest = store.put_directory("model-1", src)
    assert manifest["resource_id"] == "model-1"
    assert {f["rel_path"] for f in manifest["files"]} == {"config.json", "sub/weights.bin"}

    dest = tmp_path / "dest"
    result_dir = store.get_directory("model-1", dest)
    assert result_dir == dest
    assert (dest / "config.json").read_text() == '{"hello": "world"}'
    assert (dest / "sub" / "weights.bin").read_bytes() == (src / "sub" / "weights.bin").read_bytes()


def test_get_directory_missing_resource_raises(tmp_path):
    store = _make_store()
    with pytest.raises(DdlResourceNotFoundError):
        store.get_directory("does-not-exist", tmp_path / "dest")


def test_different_namespaces_do_not_collide(tmp_path):
    backing: dict = {}
    lock = threading.Lock()
    models_shards = [_FakeShard(backing, lock) for _ in range(2)]
    datasets_shards = [_FakeShard(backing, lock) for _ in range(2)]
    models_store = DdlResourceStore(namespace="models", num_shards=2, shards=models_shards)
    datasets_store = DdlResourceStore(namespace="datasets", num_shards=2, shards=datasets_shards)

    src = tmp_path / "src"
    src.mkdir()
    (src / "data.txt").write_text("same-id-different-namespace")
    models_store.put_directory("shared-id", src)

    with pytest.raises(DdlResourceNotFoundError):
        datasets_store.get_directory("shared-id", tmp_path / "dest")
