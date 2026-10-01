"""Adapter over the native `eclib_blobstore` pybind11 extension
(ddl/python/blobstore_bindings.cpp), which binds eclib::BlobStore
(ddl/data/chunk_store.h) exactly as-is -- no extra API is invented here.

Import is lazy/guarded: the rest of training/checkpoint can be imported,
and its buffer-protocol-only unit tests can run, even before the native
extension has been built. Only code paths that actually talk to BlobStore
raise if it's missing, with a build hint.
"""
from __future__ import annotations

from typing import Any, Dict, List

from pipeline.checkpoint.storage_backend import StorageBackend, StorageShard

try:
    import eclib_blobstore as _native

    _IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - exercised only when unbuilt
    _native = None
    _IMPORT_ERROR = exc

_BUILD_HINT = (
    "eclib_blobstore native extension is not built/importable.\n"
    "Build it with:\n"
    "  cmake -S ddl -B ddl/build -DECLIB_ENABLE_PYTHON_BINDINGS=ON "
    "-Dpybind11_DIR=$(python -m pybind11 --cmakedir)\n"
    "  cmake --build ddl/build --target eclib_blobstore -j\n"
    "and put ddl/build on PYTHONPATH (or copy the built .so next to this file)."
)


def _require_native() -> None:
    if _native is None:
        raise RuntimeError(_BUILD_HINT) from _IMPORT_ERROR


if _native is not None:
    BlobStoreError = _native.BlobStoreError
else:

    class BlobStoreError(RuntimeError):
        """Placeholder type used only while the native module is unbuilt."""


def make_config(**overrides: Any):
    """Start from eclib::default_config() and override individual fields
    by name (data_fragments, num_data_devices, chunk_size_mb, ...) -- see
    ddl/core/config.h for the full field list.

    Guards one real footgun in the C++ config: default_config() hardcodes
    replica/index device ids (small_io_replica_device_ids=[0,1],
    index_device_ids=[8,9]) sized for its own num_data_devices=10. Passing
    a smaller num_data_devices without also overriding those two lists
    leaves them pointing past the end of the (now smaller) device pool,
    which DeviceManager doesn't validate -- it throws a raw
    std::vector::_M_range_check error deep inside a put/get call instead
    of failing fast at config time. We re-derive safe defaults for them
    here unless the caller supplies its own.
    """
    _require_native()
    cfg = _native.default_config()
    num_devices = overrides.get("num_data_devices")
    if num_devices is not None and num_devices != cfg.num_data_devices:
        if num_devices < cfg.data_fragments + cfg.parity_fragments:
            raise ValueError(
                f"num_data_devices={num_devices} is below data_fragments+parity_fragments="
                f"{cfg.data_fragments + cfg.parity_fragments}; erasure-coded placement needs "
                "at least that many distinct devices"
            )
        overrides.setdefault("small_io_replica_device_ids", [0, min(1, num_devices - 1)])
        overrides.setdefault("index_device_ids", [num_devices - 2, num_devices - 1])
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"eclib_blobstore.Config has no field {key!r}")
        setattr(cfg, key, value)
    return cfg


class BlobStoreShard(StorageShard):
    """One eclib::BlobStore instance.

    BlobStore is documented single-threaded (ddl/data/chunk_store.h): a
    single instance must only ever be called from one thread. A "shard" is
    the unit of concurrency here -- the parallel checkpoint writer gives
    each of its worker threads exactly one shard and never lets two threads
    touch the same shard, which is what makes concurrent writes safe
    without adding any locking inside BlobStore itself.

    Because each BlobStore owns its own in-process simulated DeviceManager
    (ddl/sim/mocks/device_sim.h has no real file/device backing), shards do
    not share storage with each other or across process restarts -- see
    training/README.md's "Known limitations" section.
    """

    def __init__(self, config: Any = None, **config_overrides: Any):
        _require_native()
        self.config = config if config is not None else make_config(**config_overrides)
        self._store = _native.BlobStore(self.config)

    def put(self, blob_id: str, buf: Any, policy: str = "auto") -> Dict[str, Any]:
        return self._store.put_blob(blob_id, buf, policy)

    def get(self, blob_id: str, length: int) -> bytearray:
        return self._store.get_blob(blob_id, length)

    def delete(self, blob_id: str) -> None:
        self._store.delete_blob(blob_id)

    def head(self, blob_id: str) -> Dict[str, Any]:
        return self._store.head_blob(blob_id)

    def exists(self, blob_id: str) -> bool:
        return self._store.exists(blob_id)

    def flush(self) -> None:
        self._store.flush()

    def fail_device(self, index: int) -> None:
        self._store.fail_device(index)

    def recover_device(self, index: int) -> None:
        self._store.recover_device(index)

    def stats(self) -> Dict[str, Any]:
        return self._store.stats()


def create_shards(num_shards: int, **config_overrides: Any) -> List[BlobStoreShard]:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    return [BlobStoreShard(**config_overrides) for _ in range(num_shards)]


class BlobStoreBackend(StorageBackend):
    """StorageBackend adapter over the DDL eclib::BlobStore path -- the
    default BlobStoreCheckpointManager has always used. `config_overrides`
    are forwarded to make_config() for every shard, exactly as
    BlobStoreCheckpointManager's own `config_overrides` constructor arg did
    before `storage=` existed.
    """

    def __init__(self, **config_overrides: Any):
        self._config_overrides = config_overrides

    def create_shards(self, num_shards: int) -> List[BlobStoreShard]:
        return create_shards(num_shards, **self._config_overrides)
