"""DdlBackend/DdlShard connection lifecycle, with the native client faked out."""
import pytest

import pipeline.checkpoint.ddl_backend as ddl_backend


class _FakeConnection:
    instances = []
    fail_on_instance = None  # index of the DdlConnection construction that should fail

    def __init__(self, max_chunk_bytes, **kwargs):
        if _FakeConnection.fail_on_instance == len(_FakeConnection.instances):
            raise RuntimeError("DDL MiniFS target connection timed out")
        self.kwargs = kwargs
        self.closed = 0
        _FakeConnection.instances.append(self)

    def close(self):
        self.closed += 1


@pytest.fixture(autouse=True)
def _fake_native(monkeypatch):
    _FakeConnection.instances = []
    _FakeConnection.fail_on_instance = None
    monkeypatch.setattr(ddl_backend, "DdlConnection", _FakeConnection)
    monkeypatch.setattr(ddl_backend, "require_native", lambda: None)


def _backend(**kw):
    return ddl_backend.DdlBackend(server="10.0.0.1", run_id="r", ddl_cpu_base=4, **kw)


def test_shards_get_distinct_core_and_connection_ids():
    shards = _backend(connection_id_base=100).create_shards(3)
    assert [c.kwargs["core_id"] for c in _FakeConnection.instances] == [0, 1, 2]
    assert [c.kwargs["connection_id"] for c in _FakeConnection.instances] == [100, 101, 102]
    assert [c.kwargs["ddl_cpu"] for c in _FakeConnection.instances] == [4, 5, 6]
    assert len(shards) == 3


def test_shard_close_closes_its_connection():
    (shard,) = _backend().create_shards(1)
    shard.close()
    assert _FakeConnection.instances[0].closed == 1


def test_failed_shard_creation_closes_the_shards_already_opened():
    _FakeConnection.fail_on_instance = 2  # third connection times out
    with pytest.raises(RuntimeError, match="timed out"):
        _backend().create_shards(3)
    assert [c.closed for c in _FakeConnection.instances] == [1, 1]


def test_separate_backends_do_not_share_connection_ids():
    # Two processes (a job and an inspection tool, or two jobs) against one target must
    # not present the same connection_id, or the target rejects the second handshake.
    ids = set()
    for _ in range(50):
        _FakeConnection.instances = []
        _backend().create_shards(4)
        ids.update(c.kwargs["connection_id"] for c in _FakeConnection.instances)
    assert len(ids) == 200  # 50 backends x 4 shards, no repeats
    assert all(i >= 1 for i in ids)  # 0 is reserved by the target


def test_repeated_create_shards_on_one_backend_keeps_ids_distinct():
    backend = _backend(connection_id_base=7)
    backend.create_shards(2)
    backend.create_shards(2)
    assert [c.kwargs["connection_id"] for c in _FakeConnection.instances] == [7, 8, 9, 10]
    assert [c.kwargs["core_id"] for c in _FakeConnection.instances] == [0, 1, 2, 3]
