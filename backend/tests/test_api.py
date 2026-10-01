import sys
import time

import pytest
from fastapi.testclient import TestClient

from backend import workspace
from backend.api.main import create_app
from backend.job_manager import JobManager
from backend.job_store import JobStore
from backend.settings_store import SettingsStore
from backend.state_machine import JobStatus, TERMINAL_STATES


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "RUNS_ROOT", tmp_path / "runs")
    manager = JobManager(
        store=JobStore(":memory:"),
        python_executable=sys.executable,
        entrypoint_module="backend.tests.fixtures.stub_trainer",
    )
    app = create_app(job_manager=manager, settings_store=SettingsStore(":memory:"))
    return TestClient(app)


def _wait_for_terminal(client: TestClient, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = client.get(f"/jobs/{job_id}").json()
        if record["status"] in {s.value for s in TERMINAL_STATES}:
            return record
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not reach a terminal state within {timeout}s")


def test_create_job_returns_submitted(client):
    resp = client.post("/jobs", json={"name": "api-test-job"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "SUBMITTED"
    assert body["name"] == "api-test-job"
    assert "job_id" in body


def test_create_job_rejects_bad_storage_backend(client):
    resp = client.post("/jobs", json={"storage": {"backend": "nebula"}})
    assert resp.status_code == 422


def test_list_and_get_job(client):
    created = client.post("/jobs", json={"name": "list-me"}).json()
    listing = client.get("/jobs").json()
    assert any(j["job_id"] == created["job_id"] for j in listing)

    fetched = client.get(f"/jobs/{created['job_id']}").json()
    assert fetched["job_id"] == created["job_id"]
    assert fetched["config"]["name"] == "list-me"


def test_get_missing_job_404(client):
    resp = client.get("/jobs/does-not-exist")
    assert resp.status_code == 404


def test_start_missing_job_404(client):
    resp = client.post("/jobs/does-not-exist/start")
    assert resp.status_code == 404


def test_start_twice_returns_409(client):
    created = client.post("/jobs", json={}).json()
    job_id = created["job_id"]
    assert client.post(f"/jobs/{job_id}/start").status_code == 200
    assert client.post(f"/jobs/{job_id}/start").status_code == 409


def test_full_job_lifecycle_via_api(client):
    created = client.post("/jobs", json={"name": "full-lifecycle"}).json()
    job_id = created["job_id"]

    start_resp = client.post(f"/jobs/{job_id}/start")
    assert start_resp.status_code == 200
    assert start_resp.json()["status"] in ("QUEUED", "RUNNING")

    final = _wait_for_terminal(client, job_id)
    assert final["status"] == "COMPLETED"

    checkpoints = client.get(f"/jobs/{job_id}/checkpoints").json()
    assert len(checkpoints) == 1
    assert checkpoints[0]["checkpoint_id"] == "epoch-0-step-1"


def test_failed_job_via_api(client):
    created = client.post("/jobs", json={"dataset": {"name": "STUB_FAIL"}}).json()
    job_id = created["job_id"]
    client.post(f"/jobs/{job_id}/start")

    final = _wait_for_terminal(client, job_id)
    assert final["status"] == "FAILED"
    assert final["error"]


def test_event_stream_replays_full_backlog_for_completed_job(client):
    created = client.post("/jobs", json={"name": "sse-test"}).json()
    job_id = created["job_id"]
    client.post(f"/jobs/{job_id}/start")
    _wait_for_terminal(client, job_id)

    with client.stream("GET", f"/jobs/{job_id}/events") as resp:
        assert resp.status_code == 200
        lines = [line for line in resp.iter_lines() if line.startswith("data: ")]
    assert any('"job_complete"' in line for line in lines)


def test_checkpoints_for_missing_job_404(client):
    resp = client.get("/jobs/does-not-exist/checkpoints")
    assert resp.status_code == 404


def test_config_endpoint_returns_catalog(client):
    resp = client.get("/config")
    assert resp.status_code == 200
    body = resp.json()
    assert any(m["id"] == "Qwen/Qwen2.5-0.5B" for m in body["models"])
    assert {b["id"] for b in body["storage_backends"]} == {"local", "ddl"}


def test_download_artifact_for_completed_job_returns_zip(client):
    created = client.post("/jobs", json={"name": "artifact-test"}).json()
    job_id = created["job_id"]
    client.post(f"/jobs/{job_id}/start")
    _wait_for_terminal(client, job_id)

    resp = client.get(f"/jobs/{job_id}/artifact")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    assert f'{job_id}-model.zip' in resp.headers["content-disposition"]

    import io
    import zipfile

    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    names = set(zf.namelist())
    assert "adapter_model.bin" in names
    assert "adapter_config.json" in names
    # never leak the on-disk workspace path into the response
    assert str(workspace.RUNS_ROOT) not in resp.headers["content-disposition"]


def test_download_artifact_for_missing_job_404(client):
    resp = client.get("/jobs/does-not-exist/artifact")
    assert resp.status_code == 404


def test_download_artifact_for_incomplete_job_409(client):
    created = client.post("/jobs", json={"name": "not-done-yet"}).json()
    resp = client.get(f"/jobs/{created['job_id']}/artifact")
    assert resp.status_code == 409


def test_download_artifact_missing_on_disk_404(client):
    created = client.post("/jobs", json={"name": "missing-artifact"}).json()
    job_id = created["job_id"]
    client.post(f"/jobs/{job_id}/start")
    _wait_for_terminal(client, job_id)

    import shutil

    shutil.rmtree(workspace.final_adapter_dir(job_id))
    resp = client.get(f"/jobs/{job_id}/artifact")
    assert resp.status_code == 404


def test_event_stream_after_silent_crash_replays_terminal_event(client):
    # A client connecting *after* a silent-crash FAILED (no client running
    # while it happens) must still see the terminal event in its backlog
    # replay and the stream must close, not hang (Phase 2 fix).
    created = client.post("/jobs", json={"dataset": {"name": "STUB_CRASH_SILENT"}}).json()
    job_id = created["job_id"]
    client.post(f"/jobs/{job_id}/start")
    _wait_for_terminal(client, job_id)

    with client.stream("GET", f"/jobs/{job_id}/events") as resp:
        assert resp.status_code == 200
        lines = [line for line in resp.iter_lines() if line.startswith("data: ")]
    assert any('"job_failed"' in line for line in lines)


def test_delete_missing_job_404(client):
    resp = client.delete("/jobs/does-not-exist")
    assert resp.status_code == 404


def test_delete_submitted_job_returns_204(client):
    created = client.post("/jobs", json={"name": "never-started"}).json()
    job_id = created["job_id"]

    resp = client.delete(f"/jobs/{job_id}")
    assert resp.status_code == 204
    assert client.get(f"/jobs/{job_id}").status_code == 404


def test_delete_completed_job_returns_204(client):
    created = client.post("/jobs", json={"name": "delete-me"}).json()
    job_id = created["job_id"]
    client.post(f"/jobs/{job_id}/start")
    _wait_for_terminal(client, job_id)

    resp = client.delete(f"/jobs/{job_id}")
    assert resp.status_code == 204
    assert client.get(f"/jobs/{job_id}").status_code == 404
    assert not workspace.job_dir(job_id).exists()


def test_delete_running_job_returns_409(client):
    created = client.post("/jobs", json={"name": "still-running"}).json()
    job_id = created["job_id"]
    client.post(f"/jobs/{job_id}/start")

    resp = client.delete(f"/jobs/{job_id}")
    assert resp.status_code == 409
    assert client.get(f"/jobs/{job_id}").status_code == 200

    _wait_for_terminal(client, job_id)
