"""Local-filesystem-backed implementation of the StorageShard/StorageBackend
adapter (see storage_backend.py). Lets BlobStoreCheckpointManager persist
checkpoint chunks as plain files under a directory tree (e.g. an ext4 mount)
instead of going through eclib::BlobStore -- no native extension required,
so this also works in environments where eclib_blobstore hasn't been built.

Pass it in at construction time:

    from pipeline.checkpoint.localfs_backend import LocalFsBackend
    mgr = BlobStoreCheckpointManager(
        run_id="run-1", storage=LocalFsBackend("/mnt/ckpts/run-1")
    )
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Union

from pipeline.checkpoint.storage_backend import StorageBackend, StorageError, StorageShard


class LocalBlobNotFoundError(StorageError):
    """Raised by LocalFsShard for get/head/delete of a missing blob_id."""


class LocalFsShard(StorageShard):
    """One shard = one subdirectory under the backend's root.

    `blob_id` is a '/'-separated logical key (e.g.
    "ckpt/<id>/model/<name>/chunk_000001", produced by manager.py) and maps
    1:1 onto a real relative file path, so chunks land in nested
    directories the same way BlobStore's own key space is nested.
    """

    def __init__(self, root: Union[str, os.PathLike]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, blob_id: str) -> Path:
        parts = blob_id.split("/")
        # blob_id is always built by manager.py/format.py from sanitized
        # components and never contains '..' -- guarded here anyway since
        # this turns straight into a filesystem path.
        if any(p in ("", ".", "..") for p in parts):
            raise ValueError(f"unsafe blob_id: {blob_id!r}")
        return self.root.joinpath(*parts)

    def put(self, blob_id: str, buf: Any, policy: str = "auto") -> Dict[str, Any]:
        path = self._path(blob_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = bytes(buf)
        # Write-to-temp-then-rename within the same directory (same
        # filesystem) so a reader never observes a partially-written file,
        # mirroring the atomicity a single BlobStore put_blob() call gives.
        tmp_path = path.with_name(f".tmp-{os.getpid()}-{id(data)}-{path.name}")
        try:
            tmp_path.write_bytes(data)
            os.replace(tmp_path, path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
        return {"size": len(data), "path": str(path)}

    def get(self, blob_id: str, length: int) -> bytes:
        path = self._path(blob_id)
        try:
            data = path.read_bytes()
        except FileNotFoundError as exc:
            raise LocalBlobNotFoundError(f"no such blob: {blob_id!r}") from exc
        if len(data) != length:
            raise LocalBlobNotFoundError(
                f"blob {blob_id!r} is {len(data)} bytes on disk, expected {length}"
            )
        return data

    def exists(self, blob_id: str) -> bool:
        return self._path(blob_id).is_file()

    def head(self, blob_id: str) -> Dict[str, Any]:
        path = self._path(blob_id)
        try:
            size = path.stat().st_size
        except FileNotFoundError as exc:
            raise LocalBlobNotFoundError(f"no such blob: {blob_id!r}") from exc
        return {"size": size, "blob_id": blob_id}

    def delete(self, blob_id: str) -> None:
        path = self._path(blob_id)
        try:
            path.unlink()
        except FileNotFoundError as exc:
            raise LocalBlobNotFoundError(f"no such blob: {blob_id!r}") from exc


class LocalFsBackend(StorageBackend):
    """Lays out N shard subdirectories (shard-0, shard-1, ...) under `root`,
    one independent LocalFsShard per subdirectory -- mirroring how
    BlobStoreBackend hands each parallel-writer thread its own independent
    BlobStore instance.
    """

    def __init__(self, root: Union[str, os.PathLike]):
        self.root = Path(root)

    def create_shards(self, num_shards: int) -> List[LocalFsShard]:
        if num_shards < 1:
            raise ValueError("num_shards must be >= 1")
        return [LocalFsShard(self.root / f"shard-{i}") for i in range(num_shards)]
