"""Split a tensor's raw byte buffer into fixed-size, independently
addressable chunks.

Chunking is what makes parallel writers possible: each chunk becomes one
BlobStore object, so N chunks can be handed to N workers and written
concurrently (see parallel_writer.py). Chunk size is configurable
(--checkpoint-chunk-size-mb) rather than fixed, per constraint #10.
"""
from __future__ import annotations

import dataclasses
from typing import Iterator


@dataclasses.dataclass(frozen=True)
class ChunkSlice:
    index: int
    offset: int
    length: int
    data: memoryview


def chunk_count(byte_size: int, chunk_size_bytes: int) -> int:
    if chunk_size_bytes <= 0:
        raise ValueError("chunk_size_bytes must be positive")
    if byte_size == 0:
        return 1
    return (byte_size + chunk_size_bytes - 1) // chunk_size_bytes


def iter_chunks(buf, chunk_size_bytes: int) -> Iterator[ChunkSlice]:
    """Yield zero-copy ChunkSlice views over `buf` (anything supporting the
    buffer protocol). A zero-length buffer still yields exactly one
    (empty) chunk so tiny/empty tensors get a manifest entry like any
    other tensor.
    """
    if chunk_size_bytes <= 0:
        raise ValueError("chunk_size_bytes must be positive")
    mv = memoryview(buf).cast("B")
    total = len(mv)

    if total == 0:
        yield ChunkSlice(index=0, offset=0, length=0, data=mv[0:0])
        return

    index = 0
    offset = 0
    while offset < total:
        length = min(chunk_size_bytes, total - offset)
        yield ChunkSlice(index=index, offset=offset, length=length, data=mv[offset : offset + length])
        offset += length
        index += 1
