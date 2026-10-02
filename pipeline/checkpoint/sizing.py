"""Default chunk size and writer count per checkpoint storage backend.

Kept torch-free (and separate from the training script) so the rules can be
unit-tested, and so there is one place that explains them.

Why DDL differs: every DDL3 connection registers memory up front, sized from
the chunk size -- the ddl_client binding gives each connection (16 + 8) slabs of
`max_chunk_bytes` plus one application buffer, all pre-touched. Measured against
a real target that is ~121 MiB per connection at 4 MiB chunks and ~821 MiB at
32 MiB chunks (so ~3.3 GiB for the old 4-writer default, enough to push a 7.8 GB
machine into swap while training). Checkpoint tensors are small (LoRA adapters
are KiB-sized); chunk size only matters for tensors above it, which just become
more objects. So DDL defaults to small chunks and fewer writers. The target's
--max-object-mb must be at least chunk size + 8 bytes (the length frame).
"""
from __future__ import annotations

from typing import Optional, Tuple

DEFAULT_CHUNK_SIZE_MB = 256
DEFAULT_NUM_WORKERS = 4

DDL_DEFAULT_CHUNK_SIZE_MB = 4
DDL_DEFAULT_NUM_WORKERS = 2


def resolve_checkpoint_sizing(
    storage: str, chunk_size_mb: Optional[int], num_workers: Optional[int]
) -> Tuple[int, int]:
    """Returns (chunk_size_mb, num_workers). An explicit value always wins; None
    selects the default for `storage`."""
    ddl = storage == "ddl"
    if chunk_size_mb is None:
        chunk_size_mb = DDL_DEFAULT_CHUNK_SIZE_MB if ddl else DEFAULT_CHUNK_SIZE_MB
    if num_workers is None:
        num_workers = DDL_DEFAULT_NUM_WORKERS if ddl else DEFAULT_NUM_WORKERS
    return chunk_size_mb, num_workers
