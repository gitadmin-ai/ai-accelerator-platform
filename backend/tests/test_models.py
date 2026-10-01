import time

import pytest

from backend.models import (
    ModelDeleteConflictError,
    ModelNameConflictError,
    ModelNotFoundError,
    ModelStore,
    ModelValidationError,
)
from backend.resource_storage import LocalResourceStorage
from backend.settings_store import SettingsStore


@pytest.fixture
def store(tmp_path):
    return ModelStore(db_path=":memory:", storage=LocalResourceStorage(tmp_path / "models"), settings_store=SettingsStore(":memory:"))


def _wait_for_terminal(store, model_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = store.get(model_id)
        if record["status"] in ("ready", "error"):
            return record
        time.sleep(0.02)
    raise AssertionError(f"model {model_id} did not reach a terminal state within {timeout}s")


def test_create_download_starts_pending_then_moves_off_pending(store, monkeypatch):
    def fake_snapshot_download(repo_id, local_dir):
        (__import__("pathlib").Path(local_dir) / "config.json").write_text(
            '{"architectures": ["QwenForCausalLM"]}'
        )
        (__import__("pathlib").Path(local_dir) / "weights.bin").write_bytes(b"0" * 1024)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    record = store.create_download("my-model", "org/repo")
    assert record["status"] in ("pending", "downloading")

    final = _wait_for_terminal(store, record["id"])
    assert final["status"] == "ready"
    assert final["architecture"] == "QwenForCausalLM"
    assert final["size_bytes"] == 1024 + len('{"architectures": ["QwenForCausalLM"]}')
    assert store.abs_path(final).is_dir()


def test_failed_download_sets_error_status(store, monkeypatch):
    def fake_snapshot_download(repo_id, local_dir):
        raise RuntimeError("repository not found")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    record = store.create_download("bad-model", "org/does-not-exist")
    final = _wait_for_terminal(store, record["id"])
    assert final["status"] == "error"
    assert "repository not found" in final["error"]


def test_create_download_rejects_duplicate_name(store, monkeypatch):
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda repo_id, local_dir: None)
    store.create_download("dup", "org/a")
    with pytest.raises(ModelNameConflictError):
        store.create_download("dup", "org/b")


def test_get_missing_raises(store):
    with pytest.raises(ModelNotFoundError):
        store.get("does-not-exist")


def test_create_download_requires_name_and_repo(store):
    with pytest.raises(ModelValidationError):
        store.create_download("", "org/repo")
    with pytest.raises(ModelValidationError):
        store.create_download("name", "")


def test_delete_removes_record_and_files(store, monkeypatch):
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda repo_id, local_dir: None)
    record = store.create_download("to-delete", "org/repo")
    final = _wait_for_terminal(store, record["id"])
    abs_path = store.abs_path(final)
    assert abs_path.exists()

    store.delete(final["id"])

    with pytest.raises(ModelNotFoundError):
        store.get(final["id"])
    assert not abs_path.exists()


def test_delete_blocked_while_download_in_progress(store, monkeypatch):
    def slow_snapshot_download(repo_id, local_dir):
        time.sleep(1.0)

    monkeypatch.setattr("huggingface_hub.snapshot_download", slow_snapshot_download)
    record = store.create_download("in-progress", "org/repo")
    assert record["status"] in ("pending", "downloading")
    with pytest.raises(ModelDeleteConflictError):
        store.delete(record["id"])
    _wait_for_terminal(store, record["id"])


def test_delete_missing_raises(store):
    with pytest.raises(ModelNotFoundError):
        store.delete("does-not-exist")
