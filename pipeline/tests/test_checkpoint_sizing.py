import pytest

from pipeline.checkpoint.sizing import (
    DDL_DEFAULT_CHUNK_SIZE_MB,
    DDL_DEFAULT_NUM_WORKERS,
    DEFAULT_CHUNK_SIZE_MB,
    DEFAULT_NUM_WORKERS,
    resolve_checkpoint_sizing,
)


def test_ddl_defaults_are_small():
    assert resolve_checkpoint_sizing("ddl", None, None) == (DDL_DEFAULT_CHUNK_SIZE_MB, DDL_DEFAULT_NUM_WORKERS)
    # Guard the reason for the split: DDL connections pre-allocate ~30x the chunk
    # size each, so the DDL defaults must stay far below the other backends'.
    assert DDL_DEFAULT_CHUNK_SIZE_MB * DDL_DEFAULT_NUM_WORKERS * 30 < 1024  # MiB


@pytest.mark.parametrize("storage", ["local", "blobstore"])
def test_other_backends_keep_the_large_defaults(storage):
    assert resolve_checkpoint_sizing(storage, None, None) == (DEFAULT_CHUNK_SIZE_MB, DEFAULT_NUM_WORKERS)


@pytest.mark.parametrize("storage", ["ddl", "local"])
def test_explicit_values_win(storage):
    assert resolve_checkpoint_sizing(storage, 8, 7) == (8, 7)


def test_each_value_defaults_independently():
    assert resolve_checkpoint_sizing("ddl", 16, None) == (16, DDL_DEFAULT_NUM_WORKERS)
    assert resolve_checkpoint_sizing("ddl", None, 3) == (DDL_DEFAULT_CHUNK_SIZE_MB, 3)
