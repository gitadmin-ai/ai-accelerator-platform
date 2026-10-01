"""Whole-resource (directory of files) storage on DDL3, for model snapshots
and dataset directories -- distinct from pipeline/checkpoint/'s chunked
*tensor* storage, but built on the same DdlConnection primitive (see
connection.py) so both get the same parallel-shard, thread-safe behavior.

DDL3 only exposes PUT/GET opcodes -- no stat/exists/delete (see
connection.py's DdlConnection docstring and ddl_backend.py) -- so every
object this module writes is framed with an 8-byte big-endian length
header, same convention pipeline/checkpoint/ddl_backend.py uses and for the
same reasons: a size-only head() needs somewhere to read from without a
native stat call, and it keeps every write >= 8 bytes on the wire.

A resource (one `resource_id`, e.g. a model or dataset's row id in
backend's ModelStore/DatasetStore) is stored as: every file under its local
directory, chunked and fanned out across `num_shards` DdlConnections, plus
one small JSON manifest blob (file list, chunk map, sha256 per file) that
get_directory() reads first to know what to reconstruct.
"""
from __future__ import annotations

import concurrent.futures
import dataclasses
import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional

from nebula_ddl_storage.connection import DdlConnection, is_not_found_error, key_to_object_id

_HEADER = struct.Struct(">Q")  # 8-byte big-endian payload length


class DdlResourceNotFoundError(KeyError):
    pass


def _sanitize(rel_path: str) -> str:
    return rel_path.replace("/", "__")


@dataclasses.dataclass(frozen=True)
class _Chunk:
    index: int
    offset: int
    length: int
    data: bytes


def _iter_chunks(data: bytes, chunk_size_bytes: int) -> "List[_Chunk]":
    """Splits `data` into <=chunk_size_bytes pieces. A zero-length file
    still yields exactly one (empty) chunk, so it gets a manifest entry
    like any other file -- same convention as
    pipeline/checkpoint/chunker.py's iter_chunks().
    """
    total = len(data)
    if total == 0:
        return [_Chunk(index=0, offset=0, length=0, data=b"")]
    chunks = []
    index = 0
    offset = 0
    while offset < total:
        length = min(chunk_size_bytes, total - offset)
        chunks.append(_Chunk(index=index, offset=offset, length=length, data=data[offset:offset + length]))
        offset += length
        index += 1
    return chunks


class DdlResourceStore:
    """`namespace` scopes keys so resource blobs never collide with
    checkpoint blobs (which namespace by run_id) or resources of a
    different kind -- pass e.g. "models" or "datasets".
    """

    def __init__(
        self,
        namespace: str,
        server: str = "",
        port: int = 58000,
        tenant: int = 1,
        num_shards: int = 4,
        max_chunk_bytes: int = 32 * 1024 * 1024,
        ddl_cpu_base: int = 0,
        numa: int = 0,
        timeout_seconds: int = 30,
        shards: "Optional[List[Any]]" = None,
    ):
        """`shards` lets tests inject fake connections (anything with a
        `.call(method, **kwargs)` matching DdlConnection's shape) instead of
        talking to a real DDL3 target -- production code never sets it.
        """
        if num_shards < 1:
            raise ValueError("num_shards must be >= 1")
        self.namespace = namespace
        self._data_chunk_bytes = max_chunk_bytes - _HEADER.size
        self._shards = shards if shards is not None else [
            DdlConnection(
                server=server, port=port, tenant=tenant, ddl_cpu=ddl_cpu_base + i, numa=numa,
                max_chunk_bytes=max_chunk_bytes, timeout_seconds=timeout_seconds,
                core_id=i, connection_id=i + 1,
            )
            for i in range(num_shards)
        ]
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(self._shards))

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        for shard in self._shards:
            shard.close()

    # ------------------------------------------------------------------
    # low-level framed put/get/head, mirroring ddl_backend.py's DdlShard
    # ------------------------------------------------------------------

    def _object_id(self, blob_id: str):
        return key_to_object_id(f"{self.namespace}/{blob_id}")

    def _put(self, shard_index: int, blob_id: str, data: bytes) -> None:
        payload = _HEADER.pack(len(data)) + data
        hi, lo = self._object_id(blob_id)
        self._shards[shard_index % len(self._shards)].call("put", object_id_hi=hi, object_id_lo=lo, data=payload)

    def _head(self, shard_index: int, blob_id: str) -> int:
        hi, lo = self._object_id(blob_id)
        try:
            header = self._shards[shard_index % len(self._shards)].call(
                "get", object_id_hi=hi, object_id_lo=lo, length=_HEADER.size, offset=0,
            )
        except Exception as exc:  # noqa: BLE001 - re-raised as a typed not-found below
            if is_not_found_error(exc):
                raise DdlResourceNotFoundError(blob_id) from exc
            raise
        (size,) = _HEADER.unpack(header)
        return size

    def _get(self, shard_index: int, blob_id: str, length: int) -> bytes:
        if length == 0:
            return b""
        hi, lo = self._object_id(blob_id)
        return bytes(self._shards[shard_index % len(self._shards)].call(
            "get", object_id_hi=hi, object_id_lo=lo, length=length, offset=_HEADER.size,
        ))

    # ------------------------------------------------------------------
    # whole-resource put/get
    # ------------------------------------------------------------------

    def put_directory(self, resource_id: str, local_dir: Path) -> Dict[str, Any]:
        files = sorted(p for p in local_dir.rglob("*") if p.is_file())
        file_records = []
        futures = []

        for file_index, path in enumerate(files):
            rel_path = str(path.relative_to(local_dir))
            data = path.read_bytes()
            chunks = _iter_chunks(data, self._data_chunk_bytes)
            chunk_records = []
            for chunk in chunks:
                shard = (file_index + chunk.index) % len(self._shards)
                blob_id = f"resource/{resource_id}/{_sanitize(rel_path)}/chunk_{chunk.index:06d}"
                futures.append(self._executor.submit(self._put, shard, blob_id, chunk.data))
                chunk_records.append({
                    "index": chunk.index, "blob_id": blob_id, "shard": shard,
                    "offset": chunk.offset, "length": chunk.length,
                })
            file_records.append({
                "rel_path": rel_path, "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(), "chunks": chunk_records,
            })

        for future in concurrent.futures.as_completed(futures):
            future.result()  # re-raises on failure -- nothing is committed to the manifest below if any write failed

        manifest = {"resource_id": resource_id, "namespace": self.namespace, "files": file_records}
        manifest_bytes = json.dumps(manifest).encode("utf-8")
        self._put(0, f"resource/{resource_id}/manifest.json", manifest_bytes)
        return manifest

    def get_directory(self, resource_id: str, target_dir: Path) -> Path:
        manifest_blob_id = f"resource/{resource_id}/manifest.json"
        try:
            size = self._head(0, manifest_blob_id)
            manifest_bytes = self._get(0, manifest_blob_id, size)
        except DdlResourceNotFoundError:
            raise DdlResourceNotFoundError(resource_id) from None
        manifest = json.loads(manifest_bytes)

        target_dir.mkdir(parents=True, exist_ok=True)
        buffers: Dict[str, bytearray] = {}
        futures = {}
        for file_record in manifest["files"]:
            buf = bytearray(file_record["size"])
            buffers[file_record["rel_path"]] = buf
            (target_dir / file_record["rel_path"]).parent.mkdir(parents=True, exist_ok=True)
            for chunk in file_record["chunks"]:
                fut = self._executor.submit(self._get, chunk["shard"], chunk["blob_id"], chunk["length"])
                futures[fut] = (file_record["rel_path"], chunk["offset"], chunk["length"])

        for fut in concurrent.futures.as_completed(futures):
            rel_path, offset, length = futures[fut]
            data = fut.result()
            buffers[rel_path][offset:offset + length] = data

        for file_record in manifest["files"]:
            buf = buffers[file_record["rel_path"]]
            if hashlib.sha256(bytes(buf)).hexdigest() != file_record["sha256"]:
                raise DdlResourceNotFoundError(
                    f"checksum mismatch reconstructing {file_record['rel_path']} for resource {resource_id!r}"
                )
            (target_dir / file_record["rel_path"]).write_bytes(bytes(buf))
        return target_dir
