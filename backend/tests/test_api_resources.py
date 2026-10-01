import sys
import time

import pytest
from fastapi.testclient import TestClient

from backend import workspace
from backend.api.main import create_app
from backend.datasets import DatasetStore
from backend.job_manager import JobManager
from backend.job_store import JobStore
from backend.models import ModelStore
from backend.resource_storage import LocalResourceStorage
from backend.settings_store import SettingsStore


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(workspace, "LEGACY_DEMO_DATASET", tmp_path / "no-such-file.jsonl")
    manager = JobManager(
        store=JobStore(":memory:"),
        python_executable=sys.executable,
        entrypoint_module="backend.tests.fixtures.stub_trainer",
    )
    settings = SettingsStore(":memory:")
    datasets = DatasetStore(db_path=":memory:", storage=LocalResourceStorage(tmp_path / "datasets"), settings_store=settings)
    models = ModelStore(db_path=":memory:", storage=LocalResourceStorage(tmp_path / "models"), settings_store=settings)
    app = create_app(job_manager=manager, dataset_store=datasets, model_store=models, settings_store=settings)
    return TestClient(app)


def test_dataset_upload_then_list_and_get(client):
    files = {"file": ("d.jsonl", b'{"text": "a"}\n{"text": "b"}\n', "application/octet-stream")}
    resp = client.post("/datasets", data={"name": "uploaded-ds"}, files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["num_examples"] == 2
    assert body["status"] == "ready"

    listing = client.get("/datasets").json()
    assert any(d["id"] == body["id"] for d in listing)

    fetched = client.get(f"/datasets/{body['id']}").json()
    assert fetched["name"] == "uploaded-ds"


def test_dataset_upload_rejects_bad_extension(client):
    files = {"file": ("d.csv", b"a,b,c", "text/csv")}
    resp = client.post("/datasets", data={"name": "bad"}, files=files)
    assert resp.status_code == 400


def test_dataset_upload_duplicate_name_409(client):
    files = {"file": ("d.jsonl", b'{"text": "a"}\n', "application/octet-stream")}
    client.post("/datasets", data={"name": "dup"}, files=files)
    resp = client.post("/datasets", data={"name": "dup"}, files=files)
    assert resp.status_code == 409


def test_dataset_preview(client):
    files = {"file": ("d.jsonl", b'{"text": "hello world"}\n', "application/octet-stream")}
    created = client.post("/datasets", data={"name": "preview-ds"}, files=files).json()
    preview = client.get(f"/datasets/{created['id']}/preview").json()
    assert preview[0]["text"] == "hello world"


def test_get_missing_dataset_404(client):
    assert client.get("/datasets/does-not-exist").status_code == 404


def test_model_download_via_api_and_polling(client, monkeypatch):
    def fake_snapshot_download(repo_id, local_dir):
        from pathlib import Path
        Path(local_dir, "config.json").write_text('{"model_type": "qwen2"}')

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    resp = client.post("/models", json={"name": "my-model", "repo_id": "org/repo"})
    assert resp.status_code == 200
    model_id = resp.json()["id"]

    deadline = time.time() + 5
    final = None
    while time.time() < deadline:
        final = client.get(f"/models/{model_id}").json()
        if final["status"] in ("ready", "error"):
            break
        time.sleep(0.02)
    assert final["status"] == "ready"
    assert final["architecture"] == "qwen2"

    listing = client.get("/models").json()
    assert any(m["id"] == model_id for m in listing)


def test_get_missing_model_404(client):
    assert client.get("/models/does-not-exist").status_code == 404


def test_activity_reflects_dataset_creation(client):
    files = {"file": ("d.jsonl", b'{"text": "a"}\n', "application/octet-stream")}
    client.post("/datasets", data={"name": "activity-ds"}, files=files)
    activity = client.get("/activity").json()
    assert any(a["type"] == "dataset_created" and "activity-ds" in a["message"] for a in activity)


def test_delete_dataset_via_api(client):
    files = {"file": ("d.jsonl", b'{"text": "a"}\n', "application/octet-stream")}
    created = client.post("/datasets", data={"name": "to-delete"}, files=files).json()

    resp = client.delete(f"/datasets/{created['id']}")
    assert resp.status_code == 204
    assert client.get(f"/datasets/{created['id']}").status_code == 404


def test_delete_missing_dataset_404(client):
    assert client.delete("/datasets/does-not-exist").status_code == 404


def test_delete_model_via_api(client, monkeypatch):
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda repo_id, local_dir: None)
    created = client.post("/models", json={"name": "to-delete", "repo_id": "org/repo"}).json()
    model_id = created["id"]

    deadline = time.time() + 5
    while time.time() < deadline:
        if client.get(f"/models/{model_id}").json()["status"] in ("ready", "error"):
            break
        time.sleep(0.02)

    resp = client.delete(f"/models/{model_id}")
    assert resp.status_code == 204
    assert client.get(f"/models/{model_id}").status_code == 404


def test_delete_model_in_progress_returns_409(client, monkeypatch):
    def slow_snapshot_download(repo_id, local_dir):
        time.sleep(1.0)

    monkeypatch.setattr("huggingface_hub.snapshot_download", slow_snapshot_download)
    created = client.post("/models", json={"name": "in-progress", "repo_id": "org/repo"}).json()
    model_id = created["id"]

    resp = client.delete(f"/models/{model_id}")
    assert resp.status_code == 409
    assert client.get(f"/models/{model_id}").status_code == 200

    deadline = time.time() + 5
    while time.time() < deadline:
        if client.get(f"/models/{model_id}").json()["status"] in ("ready", "error"):
            break
        time.sleep(0.02)


def test_delete_missing_model_404(client):
    assert client.delete("/models/does-not-exist").status_code == 404
