"""Storage adapter that BlobStoreCheckpointManager, ParallelWriter, and
validation.py all talk to -- deliberately just the handful of blob-store
primitives the checkpoint pipeline actually uses (put/get/exists/head),
so any backend implementing this interface is a drop-in replacement for
the others. Chunking, the manifest/catalog format, and verification are
all backend-agnostic already; only object construction (manager.py's
`storage=` argument) needs to know which one is in play.

Two implementations exist today:
  - blobstore_backend.BlobStoreBackend / BlobStoreShard: DDL's
    eclib::BlobStore (erasure-coded, chunk_store.h).
  - localfs_backend.LocalFsBackend / LocalFsShard: plain files on a local
    (e.g. ext4) filesystem -- no native extension required.
"""
from __future__ import annotations

import abc
from typing import Any, Dict, List


class StorageError(RuntimeError):
    """Common base for "no such blob" errors raised by a StorageShard.

    BlobStoreShard raises the native eclib_blobstore.BlobStoreError instead
    (it isn't a subclass of this -- it's a pybind11 type from a C++
    extension), so code that needs to catch "missing blob" across backends
    generically, such as validation.py, catches
    `(BlobStoreError, StorageError)`.
    """


class StorageShard(abc.ABC):
    """One independent unit of concurrency.

    Like eclib::BlobStore itself (see blobstore_backend.py), a shard is not
    required to be thread-safe: ParallelWriter gives each of its worker
    threads exactly one shard for that thread's entire lifetime and never
    lets two threads touch the same shard, so implementations don't need
    to add their own locking to satisfy that contract.
    """

    @abc.abstractmethod
    def put(self, blob_id: str, buf: Any, policy: str = "auto") -> Dict[str, Any]:
        """Writes `buf` (anything supporting the buffer protocol) under
        `blob_id`, overwriting any existing blob with that id. Returns a
        backend-specific info dict; callers only rely on it being present,
        never on specific keys (manifest/chunk bookkeeping comes from the
        caller's own records, not this return value).
        """

    @abc.abstractmethod
    def get(self, blob_id: str, length: int) -> Any:
        """Returns exactly `length` bytes previously written to `blob_id`.
        Raises a StorageError (or, for BlobStoreShard, BlobStoreError) if
        `blob_id` doesn't exist.
        """

    @abc.abstractmethod
    def exists(self, blob_id: str) -> bool: ...

    @abc.abstractmethod
    def head(self, blob_id: str) -> Dict[str, Any]:
        """Returns at least {"size": <int>} for `blob_id` without reading
        its payload. Raises a StorageError (or BlobStoreError) if missing.
        """

    @abc.abstractmethod
    def delete(self, blob_id: str) -> None:
        """Raises a StorageError (or BlobStoreError) if `blob_id` doesn't
        exist.
        """

    def flush(self) -> None:
        """Optional: default no-op for backends with nothing to buffer."""
        return None

    def close(self) -> None:
        """Optional: release whatever the shard holds open (a network
        connection, a native handle). Default no-op. Called once per shard by
        BlobStoreCheckpointManager.shutdown() after the writer threads have
        stopped; must be safe to call from any thread. Backends that hold a
        connection MUST override it -- an unclosed DDL3 connection keeps its
        connection_id claimed on the target, so a later client in the same
        process reusing that id never completes its handshake.
        """
        return None


class StorageBackend(abc.ABC):
    """Factory for the N StorageShards BlobStoreCheckpointManager fans its
    parallel writer out across.

    Pass a concrete instance (BlobStoreBackend, LocalFsBackend, ...) to
    `BlobStoreCheckpointManager(storage=...)` to choose where checkpoint
    chunks are persisted before the pipeline starts -- nothing else in the
    pipeline (chunking, manifest, catalog, validation) needs to know which
    one is in use.
    """

    @abc.abstractmethod
    def create_shards(self, num_shards: int) -> List[StorageShard]: ...
