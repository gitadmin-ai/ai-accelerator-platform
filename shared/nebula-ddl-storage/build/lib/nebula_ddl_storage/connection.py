"""Low-level DDL3 MiniFS connection wrapper, extracted from
pipeline/checkpoint/ddl_backend.py's original DdlShard so both the
checkpoint pipeline and the resource store (models/datasets) share one
implementation of the thread-affinity workaround described below, rather
than each hand-rolling it.

CoreClient's underlying ucp_worker is created with UCS_THREAD_MODE_SINGLE
(NebulaR's common/ddl/client_target/ucx_core.cpp) and asserts that every
subsequent call comes from the exact OS thread that created it -- a second
thread touching the same connection aborts the whole process with
"Assertion `ucs_async_check_owner_thread(...)' failed", not a catchable
Python exception (see ddl_backend.py's module docstring for the full
reasoning, including why the checkpoint pipeline's coordinator thread
specifically needs this rather than just caller discipline).

DdlConnection therefore never touches ddl_client.DdlClient from whatever
thread calls its public methods. It owns one dedicated background thread
for its entire lifetime, constructs its DdlClient there, and marshals every
call onto it through a queue, blocking the caller for the result -- making
a DdlConnection instance safe to call from any external thread.

Import of the native `ddl_client` extension is lazy/guarded: this module
(and anything built on top of it) stays importable, and buffer-protocol-
only unit tests can run, even before the native extension has been built.
Only code paths that actually talk to DDL3 raise, with a build hint.
"""
from __future__ import annotations

import hashlib
import queue
import threading
from typing import Any, Optional

try:
    import ddl_client as _native

    _IMPORT_ERROR: Optional[Exception] = None
except ImportError as exc:  # pragma: no cover - exercised only when unbuilt
    _native = None
    _IMPORT_ERROR = exc

_BUILD_HINT = (
    "ddl_client native extension is not built/importable.\n"
    "Build it from the NebulaR repo's client/ directory (pybind11 module "
    "`ddl_client`, see client/python/ddl_client_py.cc) and put the built "
    ".so on PYTHONPATH."
)


def require_native() -> None:
    if _native is None:
        raise RuntimeError(_BUILD_HINT) from _IMPORT_ERROR


def key_to_object_id(key: str) -> "tuple[int, int]":
    """Derive (object_id_hi, object_id_lo) from a string key: SHA-256 of the
    UTF-8 key, first 16 bytes split big-endian into two uint64s.
    Deterministic, so a GET can recompute the id from the key alone. This is
    the single definition of the key scheme: DDL3 itself only knows the
    128-bit id (the ddl_client wheel is a thin PUT/GET binding with no
    hashing), so anything that must find these objects by name -- the
    checkpoint backend, the resource store, any future reader -- has to go
    through this function. Changing it orphans everything already stored.
    """
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    object_id_hi = int.from_bytes(digest[0:8], "big")
    object_id_lo = int.from_bytes(digest[8:16], "big")
    return object_id_hi, object_id_lo


# Mirrors ddl::runtime::CoreStatus::kNotFound (NebulaR
# common/ddl/client_target/core_protocol.h). ddl_client.DdlError carries the
# target's status in its `.status` attribute (None when the failure never
# produced a completion: startup, timeout, closed client, oversize payload).
# Only status == NOT_FOUND_STATUS is ever treated as "missing"; every other
# DdlError is re-raised as-is rather than risked being swallowed.
NOT_FOUND_STATUS = 1


def is_not_found_error(exc: Exception) -> bool:
    return getattr(exc, "status", None) == NOT_FOUND_STATUS


_SHUTDOWN = object()


class _Call:
    __slots__ = ("method", "args", "kwargs", "result")

    def __init__(self, method: str, args: tuple, kwargs: dict):
        self.method = method
        self.args = args
        self.kwargs = kwargs
        self.result: "queue.Queue" = queue.Queue(maxsize=1)


class DdlConnection:
    """One ddl_client.DdlClient, owned by one dedicated thread for its
    entire lifetime. `server`/`port`/`tenant`/`ddl_cpu`/`numa`/
    `max_chunk_bytes`/`timeout_seconds`/`core_id`/`connection_id` are
    forwarded to DdlClient's constructor exactly as named (verified against
    the real binding, client/python/ddl_client_py.cc) -- core_id/
    connection_id must each be distinct across concurrent DdlConnections
    against the same target from the same process.
    """

    def __init__(
        self,
        server: str,
        port: int = 58000,
        tenant: int = 1,
        ddl_cpu: int = 0,
        numa: int = 0,
        max_chunk_bytes: int = 32 * 1024 * 1024,
        timeout_seconds: int = 30,
        core_id: int = 0,
        connection_id: int = 1,
    ):
        require_native()
        self._calls: "queue.Queue" = queue.Queue()
        ready: "queue.Queue" = queue.Queue(maxsize=1)
        self._owner = threading.Thread(
            target=self._owner_loop,
            args=(ready, server, port, tenant, ddl_cpu, numa, max_chunk_bytes, timeout_seconds, core_id, connection_id),
            daemon=True,
            name=f"ddl-connection-{connection_id}",
        )
        self._owner.start()
        ok, err = ready.get()
        if not ok:
            raise err

    def _owner_loop(
        self, ready: "queue.Queue", server, port, tenant, ddl_cpu, numa,
        max_chunk_bytes, timeout_seconds, core_id, connection_id,
    ) -> None:
        try:
            client = _native.DdlClient(
                server=server, port=port, tenant=tenant, ddl_cpu=ddl_cpu, numa=numa,
                max_chunk_bytes=max_chunk_bytes, timeout_seconds=timeout_seconds,
                pin_cpu=False,  # this thread is the only one that will ever call it
                core_id=core_id, connection_id=connection_id,
            )
        except Exception as exc:  # noqa: BLE001 - reported back, not swallowed
            ready.put((False, exc))
            return
        ready.put((True, None))

        while True:
            call = self._calls.get()
            if call is _SHUTDOWN:
                client.close()
                return
            try:
                value = getattr(client, call.method)(*call.args, **call.kwargs)
                call.result.put((True, value))
            except Exception as exc:  # noqa: BLE001 - reported back, not swallowed
                call.result.put((False, exc))

    def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        c = _Call(method, args, kwargs)
        self._calls.put(c)
        ok, value = c.result.get()
        if not ok:
            raise value
        return value

    def close(self) -> None:
        self._calls.put(_SHUTDOWN)
        self._owner.join()
