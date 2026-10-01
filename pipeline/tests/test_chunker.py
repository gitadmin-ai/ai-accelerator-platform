import pytest

from pipeline.checkpoint.chunker import chunk_count, iter_chunks


def test_chunk_count_smaller_than_chunk_size():
    assert chunk_count(100, 1024) == 1


def test_chunk_count_exact_multiple():
    assert chunk_count(2048, 1024) == 2


def test_chunk_count_larger_with_remainder():
    assert chunk_count(2049, 1024) == 3


def test_chunk_count_empty():
    assert chunk_count(0, 1024) == 1


def test_chunk_count_rejects_nonpositive_chunk_size():
    with pytest.raises(ValueError):
        chunk_count(10, 0)


def test_iter_chunks_smaller_than_chunk_size():
    data = bytes(range(10))
    chunks = list(iter_chunks(data, 1024))
    assert len(chunks) == 1
    assert chunks[0].offset == 0
    assert chunks[0].length == 10
    assert bytes(chunks[0].data) == data


def test_iter_chunks_exact_multiple_boundary():
    data = bytes(range(20))
    chunks = list(iter_chunks(data, 5))
    assert [c.length for c in chunks] == [5, 5, 5, 5]
    assert [c.offset for c in chunks] == [0, 5, 10, 15]
    assert b"".join(bytes(c.data) for c in chunks) == data


def test_iter_chunks_with_remainder():
    data = bytes(range(23))
    chunks = list(iter_chunks(data, 5))
    assert [c.length for c in chunks] == [5, 5, 5, 5, 3]
    assert b"".join(bytes(c.data) for c in chunks) == data


def test_iter_chunks_empty_buffer_yields_one_empty_chunk():
    chunks = list(iter_chunks(b"", 1024))
    assert len(chunks) == 1
    assert chunks[0].length == 0
    assert chunks[0].offset == 0


def test_iter_chunks_indices_are_sequential():
    data = bytes(range(100))
    chunks = list(iter_chunks(data, 7))
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_iter_chunks_zero_copy_view_reflects_source_mutation():
    import numpy as np

    arr = np.arange(100, dtype=np.uint8)
    chunks = list(iter_chunks(arr, 10))
    arr[0] = 255
    assert chunks[0].data[0] == 255
