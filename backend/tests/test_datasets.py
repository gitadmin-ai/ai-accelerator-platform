import pytest

from backend import workspace
from backend.datasets import (
    DatasetNameConflictError,
    DatasetNotFoundError,
    DatasetStore,
    DatasetValidationError,
)
from backend.resource_storage import LocalResourceStorage
from backend.settings_store import SettingsStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    # No legacy demo dataset in this tmp workspace -- isolated, clean slate.
    monkeypatch.setattr(workspace, "LEGACY_DEMO_DATASET", tmp_path / "no-such-file.jsonl")
    return DatasetStore(db_path=":memory:", storage=LocalResourceStorage(tmp_path / "datasets"), settings_store=SettingsStore(":memory:"))


def _jsonl(*texts: str) -> bytes:
    import json
    return "\n".join(json.dumps({"text": t}) for t in texts).encode("utf-8")


def test_create_local_persists_file_and_metadata(store, tmp_path):
    record = store.create_local("my-dataset", "data.jsonl", _jsonl("hello", "world"))
    assert record["status"] == "ready"
    assert record["num_examples"] == 2
    assert record["source"] == "local"

    abs_path = store.abs_path(record)
    assert abs_path.exists()
    assert abs_path.name == "dataset.jsonl"
    assert (abs_path.parent / "metadata.json").exists()


def test_create_local_rejects_non_jsonl(store):
    with pytest.raises(DatasetValidationError):
        store.create_local("bad-ext", "data.csv", b"a,b,c")


def test_create_local_rejects_malformed_json(store):
    with pytest.raises(DatasetValidationError):
        store.create_local("bad-json", "data.jsonl", b"{not valid json}\n")


def test_create_local_rejects_duplicate_name(store):
    store.create_local("dup", "a.jsonl", _jsonl("x"))
    with pytest.raises(DatasetNameConflictError):
        store.create_local("dup", "b.jsonl", _jsonl("y"))


def test_list_and_get(store):
    record = store.create_local("listed", "a.jsonl", _jsonl("x", "y", "z"))
    assert any(d["id"] == record["id"] for d in store.list())
    assert store.get(record["id"])["num_examples"] == 3


def test_get_missing_raises(store):
    with pytest.raises(DatasetNotFoundError):
        store.get("does-not-exist")


def test_preview_splits_instruction_response_format(store):
    text = "### Instruction:\nWhat is X?\n\n### Response:\nX is Y."
    record = store.create_local("preview-me", "a.jsonl", _jsonl(text))
    preview = store.preview(record["id"], limit=5)
    assert preview[0]["user"] == "What is X?"
    assert preview[0]["assistant"] == "X is Y."


def test_preview_respects_limit_without_reading_whole_file(store):
    record = store.create_local("many", "a.jsonl", _jsonl(*[f"row-{i}" for i in range(50)]))
    preview = store.preview(record["id"], limit=3)
    assert len(preview) == 3


def test_seed_registers_legacy_demo_dataset_in_place(tmp_path, monkeypatch):
    demo_path = tmp_path / "demo_onboarding_v1.jsonl"
    demo_path.write_bytes(_jsonl("a", "b", "c"))
    monkeypatch.setattr(workspace, "LEGACY_DEMO_DATASET", demo_path)
    monkeypatch.setattr(workspace, "REPO_ROOT", tmp_path)

    store = DatasetStore(db_path=":memory:", storage=LocalResourceStorage(tmp_path / "datasets"), settings_store=SettingsStore(":memory:"))
    names = [d["name"] for d in store.list()]
    assert "demo_onboarding_v1" in names

    # Idempotent: constructing a second store against the same file doesn't duplicate it.
    store2 = DatasetStore(db_path=":memory:", storage=LocalResourceStorage(tmp_path / "datasets2"), settings_store=SettingsStore(":memory:"))
    assert len([d for d in store2.list() if d["name"] == "demo_onboarding_v1"]) == 1


def test_delete_removes_record_and_files(store):
    record = store.create_local("to-delete", "a.jsonl", _jsonl("x"))
    abs_path = store.abs_path(record)
    assert abs_path.exists()

    store.delete(record["id"])

    with pytest.raises(DatasetNotFoundError):
        store.get(record["id"])
    assert not abs_path.exists()
    assert not abs_path.parent.exists()


def test_delete_missing_raises(store):
    with pytest.raises(DatasetNotFoundError):
        store.delete("does-not-exist")


def test_delete_legacy_dataset_removes_record_but_not_real_file(tmp_path, monkeypatch):
    demo_path = tmp_path / "demo_onboarding_v1.jsonl"
    demo_path.write_bytes(_jsonl("a", "b"))
    monkeypatch.setattr(workspace, "LEGACY_DEMO_DATASET", demo_path)
    monkeypatch.setattr(workspace, "REPO_ROOT", tmp_path)

    store = DatasetStore(db_path=":memory:", storage=LocalResourceStorage(tmp_path / "datasets"), settings_store=SettingsStore(":memory:"))
    record = next(d for d in store.list() if d["name"] == "demo_onboarding_v1")

    store.delete(record["id"])

    with pytest.raises(DatasetNotFoundError):
        store.get(record["id"])
    # storage.delete() only ever removes root/<id> -- the legacy file lives
    # elsewhere, so deleting the resource record must not touch it.
    assert demo_path.exists()


class _FakeHfDataset:
    column_names = ["text"]

    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


def test_create_from_huggingface_passes_config_name_through(store, monkeypatch):
    captured = {}

    def fake_load_dataset(repo_id, config_name, split):
        captured.update(repo_id=repo_id, config_name=config_name, split=split)
        return _FakeHfDataset([{"text": "a"}, {"text": "b"}])

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    record = store.create_from_huggingface("glue-mrpc", "nyu-mll/glue", config_name="mrpc")
    assert record["num_examples"] == 2
    assert captured == {"repo_id": "nyu-mll/glue", "config_name": "mrpc", "split": "train"}


def test_create_from_huggingface_passes_split_through(store, monkeypatch):
    captured = {}

    def fake_load_dataset(repo_id, config_name, split):
        captured["split"] = split
        return _FakeHfDataset([{"text": "a"}])

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    # e.g. glue's "ax" diagnostic config only has a "test" split, not "train".
    store.create_from_huggingface("glue-ax", "nyu-mll/glue", config_name="ax", split="test")
    assert captured["split"] == "test"


def test_create_from_huggingface_defaults_config_name_to_none(store, monkeypatch):
    captured = {}

    def fake_load_dataset(repo_id, config_name, split):
        captured["config_name"] = config_name
        return _FakeHfDataset([{"text": "a"}])

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    store.create_from_huggingface("no-config-ds", "org/repo")
    assert captured["config_name"] is None
