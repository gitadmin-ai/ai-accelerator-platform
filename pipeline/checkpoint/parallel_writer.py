"""Bounded, shard-based parallel checkpoint writer.

Design (see training/README.md for the full rationale):

    Producer (manager.py builds chunk jobs)
        |
        v
    N bounded per-shard queues  <-- backpressure: a full queue blocks the
        |    |    |                 producer instead of buffering an
        v    v    v                 unbounded number of pinned chunks
    Worker Worker Worker  (exactly N OS threads, never one per tensor/chunk)
        |    |    |
        v    v    v
    Shard0 Shard1 Shard2  (independent eclib::BlobStore instances)

Each worker thread owns exactly one BlobStore shard for its entire
lifetime and no other thread ever touches that shard, so this achieves
real concurrent execution (the GIL is released for the duration of each
native put_blob/get_blob call -- see blobstore_bindings.cpp) while
respecting BlobStore's documented single-threaded-per-instance contract.
"""
from __future__ import annotations

import dataclasses
import queue
import threading
import time
from typing import Any, List, Optional, Union

from pipeline.checkpoint.serializer import sha256_hex
from pipeline.checkpoint.storage_backend import StorageShard


@dataclasses.dataclass
class WriteJob:
    shard: int
    blob_id: str
    data: Any  # anything supporting the buffer protocol (memoryview, bytes, ndarray)
    policy: str = "auto"
    tag: Any = None  # opaque, echoed back on the result for the caller to match up


@dataclasses.dataclass
class WriteResult:
    job: WriteJob
    ok: bool
    error: Optional[str]
    put_info: Optional[dict]
    bytes_written: int
    duration_s: float


@dataclasses.dataclass
class VerifyJob:
    """A checksum re-read, run on the same per-shard worker pool as writes
    so post-write validation doesn't become a serial bottleneck that
    swamps the parallel-write speedup (it did, before this existed --
    see training/README.md's performance section).
    """

    shard: int
    blob_id: str
    length: int
    expected_sha256: str
    tag: Any = None


@dataclasses.dataclass
class VerifyResult:
    job: VerifyJob
    ok: bool
    error: Optional[str]
    duration_s: float


Job = Union[WriteJob, VerifyJob]
JobResult = Union[WriteResult, VerifyResult]


_SHUTDOWN = object()


class ParallelWriter:
    def __init__(self, shards: List[StorageShard], queue_maxsize_per_shard: int = 8):
        if not shards:
            raise ValueError("ParallelWriter needs at least one shard")
        self._shards = shards
        self._num_shards = len(shards)
        self._queues: List["queue.Queue"] = [
            queue.Queue(maxsize=queue_maxsize_per_shard) for _ in shards
        ]
        self._results: "queue.Queue[WriteResult]" = queue.Queue()
        self._submitted = 0
        self._threads: List[threading.Thread] = []
        for idx, shard in enumerate(shards):
            t = threading.Thread(
                target=self._worker_loop, args=(idx, shard), daemon=True, name=f"ckpt-writer-{idx}"
            )
            t.start()
            self._threads.append(t)

    @property
    def num_workers(self) -> int:
        return self._num_shards

    def _worker_loop(self, shard_idx: int, shard: StorageShard) -> None:
        q = self._queues[shard_idx]
        while True:
            job = q.get()
            if job is _SHUTDOWN:
                q.task_done()
                return
            start = time.perf_counter()
            try:
                if isinstance(job, VerifyJob):
                    data = shard.get(job.blob_id, job.length)
                    ok = len(data) == job.length and sha256_hex(data) == job.expected_sha256
                    error = None if ok else "size or checksum mismatch on re-read"
                    self._results.put(
                        VerifyResult(job=job, ok=ok, error=error, duration_s=time.perf_counter() - start)
                    )
                else:
                    info = shard.put(job.blob_id, job.data, job.policy)
                    self._results.put(
                        WriteResult(
                            job=job,
                            ok=True,
                            error=None,
                            put_info=info,
                            bytes_written=len(job.data),
                            duration_s=time.perf_counter() - start,
                        )
                    )
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                result_cls = VerifyResult if isinstance(job, VerifyJob) else WriteResult
                extra = {} if result_cls is VerifyResult else {"put_info": None, "bytes_written": 0}
                self._results.put(
                    result_cls(
                        job=job, ok=False, error=str(exc), duration_s=time.perf_counter() - start, **extra
                    )
                )
            finally:
                q.task_done()

    def submit(self, job: Job) -> None:
        """Blocks (backpressure) if the target shard's queue is full."""
        self._queues[job.shard % self._num_shards].put(job)
        self._submitted += 1

    def collect(self, expected: int) -> List[JobResult]:
        """Block until `expected` results have arrived, in completion order
        (not submission order -- callers match results back to their jobs
        via WriteResult.job / WriteResult.job.tag).
        """
        return [self._results.get() for _ in range(expected)]

    def shutdown(self) -> None:
        for q in self._queues:
            q.put(_SHUTDOWN)
        for t in self._threads:
            t.join()
