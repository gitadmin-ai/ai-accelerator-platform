"""DDL3 MiniFS storage backend -- StorageShard/StorageBackend adapter (see
storage_backend.py) over NebulaR's ddl_client (client/python/ddl_client_py.cc,
a pybind11 binding of CoreClient). Lets BlobStoreCheckpointManager persist
checkpoint chunks to a real DDL3 MiniFS target instead of eclib::BlobStore or
plain files -- chunking, the manifest/catalog format, and verification are
all backend-agnostic already (see manager.py); only object construction
(manager.py's `storage=` argument) needs to know this backend exists.

Import is lazy/guarded, same pattern as blobstore_backend.py: the rest of
training/checkpoint stays importable, and non-DDL backends' tests can run,
even before ddl_client has been built.

Framing
-------
DDL3 only exposes PUT/GET opcodes -- no stat/exists/delete (see NebulaR's
target/fscore/objstore/ddl_minifs_target.cc, which registers only those two
handlers) -- and rejects a zero-byte PUT outright (status kInvalid, see
common/ddl/client_target/core_protocol.h). But the StorageShard contract
requires a size-only head()/exists(), and chunker.py's contract requires
support for zero-length chunks (a 0-byte tensor still yields one empty
ChunkSlice, "so tiny/empty tensors get a manifest entry like any other
tensor"). So every object DdlShard writes is framed with an 8-byte
big-endian length header:

    [ 8-byte length (big-endian) ][ payload bytes ]

head()/exists() read only those first 8 bytes -- DDL3 supports partial
offset/length GETs, verified against a live target while building
ddl_client. get() skips the header and reads exactly `length` payload
bytes, short-circuiting to b"" without a wire round trip when length == 0.
This keeps every write >= 8 bytes on the wire regardless of the caller's
buffer size (dodging the zero-byte-PUT rejection), and gives head()
something to read without a native stat call.

Key derivation and run isolation
---------------------------------
blob_id strings (e.g. "ckpt/epoch-1-step-500/model/base_model..._2Eweight/
chunk_000000", built by manager.py/format.py) are hashed into DDL3's
128-bit (object_id_hi, object_id_lo) via ddl_checkpoint_keys.key_to_object_id
(NebulaR's client/python/ddl_checkpoint_keys.py). manager.py's blob_id
scheme does NOT include run_id -- harmless for LocalFsBackend (one
directory per run) and BlobStoreBackend (one ephemeral in-process store per
run), but DDL3 is a real shared, persistent, networked target: two runs
checkpointing at the same epoch/step would otherwise silently overwrite
each other's blobs. DdlShard namespaces every key with `namespace`
(DdlBackend passes run_id) before hashing to prevent that; `tenant` gives a
second, protocol-level isolation knob (a distinct on-disk path per tenant)
for operators who want to separate runs/jobs more strongly than a hash
namespace.

Thread/connection ownership
----------------------------
CoreClient's underlying ucp_worker is created with UCS_THREAD_MODE_SINGLE
(NebulaR's common/ddl/client_target/ucx_core.cpp) and asserts that every
subsequent call comes from the exact OS thread that created it -- verified
directly: a second thread touching the same connection aborts the whole
process with "Assertion `ucs_async_check_owner_thread(...)' failed", not a
catchable Python exception. ParallelWriter's "each worker thread owns
exactly one shard for its entire lifetime" (see its docstring) covers jobs
submitted through it, but manager.py's coordinator thread *also* calls
shard[_COORDINATOR_SHARD].put()/get()/exists()/head() directly and
synchronously for the manifest/catalog/scheduler/skeleton JSON objects
(see _put_json_object/_get_json_object/_read_catalog/_read_manifest) --
never routed through ParallelWriter. So shard 0 is genuinely touched by two
different OS threads across its lifetime (its own worker thread for
regular tensor chunks, then the coordinator thread for JSON objects) by
manager.py's own design, backend-independent. That's harmless for
LocalFsShard/BlobStoreShard (no thread-affinity requirement) but fatal for
a UCX-backed shard.

DdlShard therefore does not touch DdlClient/ucp_worker from whatever thread
calls its public methods -- it delegates to `nebula_ddl_storage.connection.
DdlConnection`, a shared owner-thread-per-connection wrapper (extracted
here so the new resource store used for model/dataset storage gets the
exact same thread-safety guarantee without duplicating this reasoning).
Each DdlConnection owns one dedicated background thread for its entire
lifetime, constructs its DdlClient there (so construction and every
subsequent use share one OS thread by construction, not by caller
discipline), and marshals every put()/get()/head() call onto it through a
queue, blocking the caller for the result. This makes DdlShard itself safe
to call from any external thread -- exactly what the StorageShard contract
already promises manager.py/ParallelWriter -- without changing either of
those files. The same reasoning is why pin_cpu=False is passed to the
underlying DdlClient: OS-thread CPU affinity is meaningless to pin from any
thread other than the one about to make the actual calls, which here is
the DdlConnection's own owner thread, not any caller's.
"""
from __future__ import annotations

import struct
from typing import Any, Dict, List

from nebula_ddl_storage.connection import DdlConnection, is_not_found_error, key_to_object_id, require_native
from pipeline.checkpoint.storage_backend import StorageBackend, StorageError, StorageShard

_HEADER = struct.Struct(">Q")  # 8-byte big-endian payload length


class DdlBlobNotFoundError(StorageError):
    """Raised by DdlShard for get()/head() of a missing blob_id."""


class DdlShard(StorageShard):
    def __init__(self, namespace: str, max_chunk_bytes: int, **client_kwargs: Any):
        require_native()
        self.namespace = namespace
        self._conn = DdlConnection(max_chunk_bytes=max_chunk_bytes + _HEADER.size, **client_kwargs)

    def _object_id(self, blob_id: str):
        return key_to_object_id(f"{self.namespace}/{blob_id}")

    def put(self, blob_id: str, buf: Any, policy: str = "auto") -> Dict[str, Any]:
        data = bytes(buf)
        payload = _HEADER.pack(len(data)) + data
        object_id_hi, object_id_lo = self._object_id(blob_id)
        self._conn.call("put", object_id_hi=object_id_hi, object_id_lo=object_id_lo, data=payload)
        return {"size": len(data)}

    def get(self, blob_id: str, length: int) -> bytes:
        if length == 0:
            return b""
        object_id_hi, object_id_lo = self._object_id(blob_id)
        try:
            return self._conn.call(
                "get", object_id_hi=object_id_hi, object_id_lo=object_id_lo,
                length=length, offset=_HEADER.size,
            )
        except Exception as exc:  # noqa: BLE001 - re-raised below unless it's a not-found
            if is_not_found_error(exc):
                raise DdlBlobNotFoundError(f"no such blob: {blob_id!r}") from exc
            raise

    def exists(self, blob_id: str) -> bool:
        try:
            self.head(blob_id)
            return True
        except DdlBlobNotFoundError:
            return False

    def head(self, blob_id: str) -> Dict[str, Any]:
        object_id_hi, object_id_lo = self._object_id(blob_id)
        try:
            header = self._conn.call(
                "get", object_id_hi=object_id_hi, object_id_lo=object_id_lo,
                length=_HEADER.size, offset=0,
            )
        except Exception as exc:  # noqa: BLE001 - re-raised below unless it's a not-found
            if is_not_found_error(exc):
                raise DdlBlobNotFoundError(f"no such blob: {blob_id!r}") from exc
            raise
        (size,) = _HEADER.unpack(header)
        return {"size": size, "blob_id": blob_id}

    def delete(self, blob_id: str) -> None:
        raise NotImplementedError(
            "DDL3 MiniFS has no delete opcode (only PUT/GET are registered -- "
            "see NebulaR's target/fscore/objstore/ddl_minifs_target.cc). Not "
            "called anywhere in the save/load/resume path today, only "
            "required by the StorageShard interface."
        )

    def close(self) -> None:
        self._conn.close()


class DdlBackend(StorageBackend):
    """One DdlShard (DdlClient connection) per ParallelWriter worker thread,
    each pinned to its own DDL3 owner core (`ddl_cpu_base + shard index`) so
    concurrent shards never share a CoreClient connection.
    """

    def __init__(
        self,
        server: str,
        run_id: str,
        port: int = 58000,
        tenant: int = 1,
        max_chunk_bytes: int = 32 * 1024 * 1024,
        ddl_cpu_base: int = 0,
        numa: int = 0,
        timeout_seconds: int = 30,
    ):
        require_native()
        self.server = server
        self.run_id = run_id
        self.port = port
        self.tenant = tenant
        self.max_chunk_bytes = max_chunk_bytes
        self.ddl_cpu_base = ddl_cpu_base
        self.numa = numa
        self.timeout_seconds = timeout_seconds

    def create_shards(self, num_shards: int) -> List[DdlShard]:
        # core_id/connection_id must be distinct per concurrent DdlClient
        # against the same target (see ddl_client_py.cc's constructor
        # comment) -- shard_index (0-based) satisfies core_id directly;
        # connection_id starts at 1 since 0 has special "empty slot"
        # meaning in UcxCoreTransport::bind_connection.
        shards: List[DdlShard] = []
        try:
            for shard_index in range(num_shards):
                shards.append(
                    DdlShard(
                        namespace=self.run_id,
                        max_chunk_bytes=self.max_chunk_bytes,
                        server=self.server,
                        port=self.port,
                        tenant=self.tenant,
                        ddl_cpu=self.ddl_cpu_base + shard_index,
                        numa=self.numa,
                        timeout_seconds=self.timeout_seconds,
                        core_id=shard_index,
                        connection_id=shard_index + 1,
                    )
                )
        except Exception:
            # A later shard failing to connect must not leave the earlier
            # ones open: their connection_ids would stay claimed on the
            # target and a retry from this process would hang on handshake.
            for shard in shards:
                try:
                    shard.close()
                except Exception:  # noqa: BLE001 - original error matters more
                    pass
            raise
        return shards
