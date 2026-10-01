import numpy as np
import pytest

from pipeline.checkpoint import serializer


@pytest.mark.parametrize(
    "np_dtype",
    [np.float32, np.float64, np.float16, np.int64, np.int32, np.int16, np.int8, np.uint8, np.bool_],
)
def test_dtype_roundtrip(np_dtype):
    dtype_str = serializer.dtype_str_from_numpy(np.dtype(np_dtype))
    assert dtype_str.startswith("torch.")
    assert serializer.numpy_dtype_for(dtype_str) == np.dtype(np_dtype)


def test_bfloat16_has_no_numpy_dtype():
    assert serializer.numpy_dtype_for("torch.bfloat16") is None
    assert "torch.bfloat16" in serializer.ITEM_SIZE_BYTES


@pytest.mark.parametrize("shape", [(1,), (10,), (3, 4), (2, 3, 4), (0,), ()])
def test_numel_and_byte_size(shape):
    arr = np.zeros(shape, dtype=np.float32)
    assert serializer.numel_of(shape) == arr.size
    assert serializer.byte_size_of("torch.float32", shape) == arr.nbytes


def test_buffer_from_numpy_is_zero_copy_view():
    arr = np.arange(1000, dtype=np.float32)
    buf = serializer.buffer_from_numpy(arr)
    assert len(buf) == arr.nbytes
    # mutate original -> view reflects it (same underlying memory)
    arr[0] = 42.0
    assert bytes(buf[:4]) == np.float32(42.0).tobytes()


def test_buffer_from_numpy_noncontiguous_copies_safely():
    arr = np.arange(20, dtype=np.float32).reshape(4, 5)
    transposed = arr.T  # not C-contiguous
    buf = serializer.buffer_from_numpy(transposed)
    assert len(buf) == transposed.nbytes
    restored = np.frombuffer(buf, dtype=np.float32).reshape(transposed.shape)
    assert np.array_equal(restored, transposed)


def test_reconstruct_numpy_roundtrip():
    arr = np.random.RandomState(0).randn(7, 3).astype(np.float32)
    buf = serializer.buffer_from_numpy(arr)
    restored = serializer.reconstruct_numpy("torch.float32", list(arr.shape), bytes(buf))
    assert np.array_equal(restored, arr)


def test_reconstruct_numpy_rejects_bfloat16():
    with pytest.raises(ValueError):
        serializer.reconstruct_numpy("torch.bfloat16", [4], b"\x00" * 8)


def test_sha256_hex_deterministic_and_sensitive_to_content():
    a = b"hello world"
    b = b"hello worle"
    assert serializer.sha256_hex(a) == serializer.sha256_hex(a)
    assert serializer.sha256_hex(a) != serializer.sha256_hex(b)
    assert len(serializer.sha256_hex(a)) == 64
