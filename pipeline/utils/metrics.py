"""Checkpoint performance measurement -- real numbers only, never fabricated.

Every field on CheckpointMetrics is filled in from an actual
time.perf_counter() delta or an actual byte count measured during
save_checkpoint(); nothing here is estimated or hardcoded.
"""
from __future__ import annotations

import dataclasses
import time
from contextlib import contextmanager


@dataclasses.dataclass
class Timing:
    """Mutable phase-time accumulator threaded through save_checkpoint()."""

    gpu_to_cpu_s: float = 0.0
    chunking_s: float = 0.0
    write_s: float = 0.0
    commit_s: float = 0.0

    @contextmanager
    def measure(self, field: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            setattr(self, field, getattr(self, field) + (time.perf_counter() - start))


@dataclasses.dataclass
class CheckpointMetrics:
    checkpoint_id: str
    size_bytes: int
    num_tensors: int
    num_chunks: int
    chunk_size_bytes: int
    num_workers: int

    gpu_to_cpu_s: float
    chunking_s: float
    write_s: float
    commit_s: float
    total_s: float

    @property
    def throughput_mb_s(self) -> float:
        if self.total_s <= 0:
            return 0.0
        return (self.size_bytes / (1024 * 1024)) / self.total_s

    def format_report(self) -> str:
        mb = self.size_bytes / (1024 * 1024)
        lines = [
            "=" * 60,
            "CHECKPOINT",
            "=" * 60,
            f"Checkpoint: {self.checkpoint_id}",
            f"Size:       {mb:.2f} MB",
            f"Tensors:    {self.num_tensors}",
            f"Chunks:     {self.num_chunks}",
            f"Chunk size: {self.chunk_size_bytes // (1024 * 1024)} MB",
            f"Workers:    {self.num_workers}",
            "",
            f"GPU->CPU:   {self.gpu_to_cpu_s:.3f} sec",
            f"Chunking:   {self.chunking_s:.3f} sec",
            f"Writes:     {self.write_s:.3f} sec",
            f"Commit:     {self.commit_s:.3f} sec",
            "",
            f"Total:      {self.total_s:.3f} sec",
            f"Throughput: {self.throughput_mb_s:.2f} MB/s",
            "=" * 60,
        ]
        return "\n".join(lines)
