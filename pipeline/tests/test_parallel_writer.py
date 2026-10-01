import pytest

pytest.importorskip("eclib_blobstore", reason="eclib_blobstore native extension not built")

from pipeline.checkpoint.blobstore_backend import create_shards
from pipeline.checkpoint.parallel_writer import ParallelWriter, WriteJob


def _shards(n):
    return create_shards(n, num_data_devices=10, device_size_mb=32)


def test_parallel_writer_round_trip_across_shards():
    shards = _shards(4)
    writer = ParallelWriter(shards, queue_maxsize_per_shard=4)
    try:
        jobs = [
            WriteJob(shard=i % 4, blob_id=f"blob-{i}", data=bytes([i]) * 100, tag=i)
            for i in range(20)
        ]
        for job in jobs:
            writer.submit(job)
        results = writer.collect(len(jobs))
        assert len(results) == len(jobs)
        assert all(r.ok for r in results)
        for r in results:
            shard = shards[r.job.shard % 4]
            assert bytes(shard.get(r.job.blob_id, len(r.job.data))) == r.job.data
    finally:
        writer.shutdown()


def test_parallel_writer_num_workers_matches_shard_count():
    shards = _shards(3)
    writer = ParallelWriter(shards)
    try:
        assert writer.num_workers == 3
    finally:
        writer.shutdown()


def test_parallel_writer_reports_failure_without_raising():
    shards = _shards(2)
    writer = ParallelWriter(shards)
    try:
        # policy="not-a-real-policy" -> native put_blob raises inside the worker;
        # the writer must report it as a failed WriteResult, not crash the pool.
        writer.submit(WriteJob(shard=0, blob_id="bad", data=b"x", policy="bogus", tag="bad"))
        results = writer.collect(1)
        assert len(results) == 1
        assert results[0].ok is False
        assert results[0].error
    finally:
        writer.shutdown()


def test_parallel_writer_backpressure_bounds_queue():
    shards = _shards(1)
    writer = ParallelWriter(shards, queue_maxsize_per_shard=2)
    try:
        # submit more than the queue can hold; must not raise or grow unbounded,
        # it should simply block until workers drain (verified by getting all results back)
        n = 10
        for i in range(n):
            writer.submit(WriteJob(shard=0, blob_id=f"bp-{i}", data=b"x" * 10, tag=i))
        results = writer.collect(n)
        assert len(results) == n
        assert all(r.ok for r in results)
    finally:
        writer.shutdown()
