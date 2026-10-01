"""Checkpoint integrity validation.

BlobStore's own putBlob() has no failure signal in its return type, and a
head_blob() size check alone is not actually enough to catch a silent
write failure in the simulated backend: BlobMeta.data_length is recorded
from the caller-supplied length unconditionally, independent of whether
the bytes actually made it to a (possibly-failed) device (see
ddl/data/chunk_store.cpp's put_small(), which skips failed devices in its
replica loop without surfacing that fact). So this file gives
BlobStoreCheckpointManager two check strengths:

  - check_chunk_head() / check_object_head(): existence + size only (one
    head_blob call). Fast, but only reliably catches a chunk that never
    got an index entry at all.
  - check_chunk_checksum() / check_object_checksum(): a full read-back +
    SHA-256 compare against the manifest. This is the one that actually
    catches the "index says OK but the bytes are wrong/missing" class of
    failure, at the cost of doubling read/write I/O -- BlobStoreCheckpoint-
    Manager therefore uses this as its *default* inline post-write check
    (verify_mode="checksum"); verify_mode="head" trades that guarantee for
    speed, and "none" skips inline validation entirely (verify() can
    still be run later).
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional

from pipeline.checkpoint.blobstore_backend import BlobStoreError
from pipeline.checkpoint.format import Manifest, STATUS_COMPLETE
from pipeline.checkpoint.serializer import sha256_hex
from pipeline.checkpoint.storage_backend import StorageError, StorageShard

# Any backend's "no such blob" exception -- BlobStoreShard raises the native
# BlobStoreError (a pybind11 type, not a StorageError subclass); other
# StorageShard implementations (e.g. LocalFsShard) raise a StorageError.
_NOT_FOUND_ERRORS = (BlobStoreError, StorageError)


@dataclasses.dataclass
class VerifyIssue:
    kind: str
    name: str
    detail: str


@dataclasses.dataclass
class VerifyReport:
    checkpoint_id: str
    ok: bool
    chunks_checked: int
    objects_checked: int
    issues: List[VerifyIssue]


def _check_head(shards, shard_idx, blob_id, expected_len, kind_prefix) -> Optional[VerifyIssue]:
    shard = shards[shard_idx % len(shards)]
    try:
        info = shard.head(blob_id)
    except _NOT_FOUND_ERRORS as exc:
        return VerifyIssue(f"missing_{kind_prefix}", blob_id, str(exc))
    if info["size"] != expected_len:
        return VerifyIssue("size_mismatch", blob_id, f"expected {expected_len}, got {info['size']}")
    return None


def _check_checksum(
    shards, shard_idx, blob_id, expected_len, expected_sha, kind_prefix
) -> Optional[VerifyIssue]:
    shard = shards[shard_idx % len(shards)]
    try:
        data = shard.get(blob_id, expected_len)
    except _NOT_FOUND_ERRORS as exc:
        return VerifyIssue(f"missing_{kind_prefix}", blob_id, str(exc))
    if len(data) != expected_len:
        return VerifyIssue("size_mismatch", blob_id, f"expected {expected_len}, got {len(data)}")
    digest = sha256_hex(data)
    if digest != expected_sha:
        return VerifyIssue("checksum_mismatch", blob_id, f"expected {expected_sha}, got {digest}")
    return None


def check_chunk_head(shards: List[StorageShard], chunk) -> Optional[VerifyIssue]:
    return _check_head(shards, chunk.shard, chunk.blob_id, chunk.length, "chunk")


def check_object_head(shards: List[StorageShard], obj) -> Optional[VerifyIssue]:
    return _check_head(shards, obj.shard, obj.blob_id, obj.length, "object")


def check_chunk_checksum(shards: List[StorageShard], chunk) -> Optional[VerifyIssue]:
    return _check_checksum(shards, chunk.shard, chunk.blob_id, chunk.length, chunk.sha256, "chunk")


def check_object_checksum(shards: List[StorageShard], obj) -> Optional[VerifyIssue]:
    return _check_checksum(shards, obj.shard, obj.blob_id, obj.length, obj.sha256, "object")


def verify_deep(shards: List[StorageShard], manifest: Manifest) -> VerifyReport:
    issues: List[VerifyIssue] = []
    chunks_checked = 0
    objects_checked = 0

    for chunk in manifest.all_chunks():
        chunks_checked += 1
        issue = check_chunk_checksum(shards, chunk)
        if issue:
            issues.append(issue)

    for obj in manifest.all_objects():
        objects_checked += 1
        issue = check_object_checksum(shards, obj)
        if issue:
            issues.append(issue)

    return VerifyReport(
        checkpoint_id=manifest.checkpoint_id,
        ok=(len(issues) == 0 and manifest.status == STATUS_COMPLETE),
        chunks_checked=chunks_checked,
        objects_checked=objects_checked,
        issues=issues,
    )
