#!/usr/bin/env python3
"""Benchmark BlobStoreCheckpointManager's parallel-writer throughput across
different worker counts, using identical checkpoint data for every run so
the numbers are comparable.

Note on scope: the task brief's example CLI is
`--checkpoint <checkpoint-id> --workers 1,2,4,8,16`, i.e. re-writing an
*existing* checkpoint at different worker counts. That isn't meaningful
here: eclib::BlobStore's current backend (ddl/sim/mocks/device_sim.h) is a
pure in-process memory simulation with no real file/device persistence, so
a checkpoint saved by an earlier process run doesn't exist for this one to
load (see training/README.md's "Known limitations"). Instead this
generates one synthetic tensor payload up front -- sized and shaped like a
real LoRA-training checkpoint (model + optimizer tensors) -- and saves that
*same in-memory payload* through a fresh BlobStoreCheckpointManager (fresh
shards) at each worker count, which is what actually varies the thing
under test (parallel writer count) while holding everything else fixed.

Usage:
    python -m training.benchmark_checkpoint --workers 1,2,4,8,16 \\
        --size-mb 512 --chunk-size-mb 16 --num-tensors 64
"""
from __future__ import annotations

import argparse
import math
from typing import Dict, List

import numpy as np

from pipeline.checkpoint.manager import BlobStoreCheckpointManager
from pipeline.utils.metrics import CheckpointMetrics


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", default="1,2,4,8,16", help="Comma-separated worker counts to compare")
    p.add_argument("--size-mb", type=float, default=128.0, help="Approximate total payload size")
    p.add_argument("--chunk-size-mb", type=float, default=8.0)
    p.add_argument("--num-tensors", type=int, default=32, help="Number of tensors the payload is split into")
    p.add_argument(
        "--num-data-devices",
        type=int,
        default=10,
        help="Simulated BlobStore devices per shard (must be >= data_fragments+parity_fragments, 10 by default)",
    )
    p.add_argument(
        "--device-size-mb",
        type=int,
        default=0,
        help="Simulated per-device capacity per shard (0 = auto-size from --size-mb "
        "with headroom for replica+EC overhead)",
    )
    p.add_argument("--verify-mode", default="checksum", choices=["checksum", "head", "none"])
    return p.parse_args(argv)


def make_synthetic_payload(total_mb: float, num_tensors: int, seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    bytes_per_tensor = int(total_mb * 1024 * 1024 / num_tensors)
    elems_per_tensor = max(1, bytes_per_tensor // 4)  # float32
    return {
        f"layer.{i}.lora_A.weight": rng.randn(elems_per_tensor).astype(np.float32)
        for i in range(num_tensors)
    }


def _auto_device_size_mb(size_mb: float, num_data_devices: int, num_workers: int) -> int:
    # Each worker owns an independent, full device pool (own DeviceManager
    # -- see blobstore_backend.py), but only handles ~1/num_workers of the
    # payload (chunks round-robin across shards), and BlobStore both
    # replicates small IO and erasure-codes it -- so size each shard for
    # its own share of the payload, not the whole thing, with 4x headroom
    # for replica (2x) + 8+2 EC (1.25x) overhead.
    per_shard_mb = size_mb / num_workers
    return max(16, math.ceil(per_shard_mb * 4 / num_data_devices))


def run_one(workers: int, args: argparse.Namespace, payload: Dict[str, np.ndarray]) -> CheckpointMetrics:
    device_size_mb = args.device_size_mb or _auto_device_size_mb(
        args.size_mb, args.num_data_devices, workers
    )
    manager = BlobStoreCheckpointManager(
        run_id=f"bench-{workers}",
        num_workers=workers,
        chunk_size_bytes=int(args.chunk_size_mb * 1024 * 1024),
        verify_mode=args.verify_mode,
        config_overrides=dict(
            num_data_devices=args.num_data_devices,
            device_size_mb=device_size_mb,
        ),
    )
    try:
        _, metrics = manager.save_checkpoint(
            "bench-checkpoint",
            payload,
            None,
            None,
            {"epoch": 0, "global_step": 0},
            epoch=0,
            global_step=0,
            model_name_or_path="benchmark",
            base_model_id="benchmark",
            dataset_id="benchmark",
            lora_config={},
            training_config={},
        )
        return metrics
    finally:
        manager.shutdown()


def format_table(rows: List[CheckpointMetrics]) -> str:
    lines = [
        f"{'workers':>7} | {'checkpoint_time':>15} | {'throughput':>14}",
        f"{'-' * 7}-|-{'-' * 15}-|-{'-' * 14}",
    ]
    for m in rows:
        lines.append(f"{m.num_workers:>7} | {m.total_s:>13.3f}s | {m.throughput_mb_s:>11.2f} MB/s")
    return "\n".join(lines)


def main(argv=None) -> None:
    args = parse_args(argv)
    worker_counts = [int(w) for w in args.workers.split(",")]

    payload = make_synthetic_payload(args.size_mb, args.num_tensors)
    actual_mb = sum(a.nbytes for a in payload.values()) / (1024 * 1024)
    print(f"Synthetic payload: {len(payload)} tensors, {actual_mb:.1f} MB total\n")

    results: List[CheckpointMetrics] = []
    for w in worker_counts:
        print(f"--- {w} worker(s) ---")
        metrics = run_one(w, args, payload)
        print(metrics.format_report())
        print()
        results.append(metrics)

    print(format_table(results))


if __name__ == "__main__":
    main()
