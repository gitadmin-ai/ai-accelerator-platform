import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from backend import job_manager, workspace
from backend.config import JobConfig
from backend.job_manager import JobManager, _RemoteTrainingHandle
from backend.job_store import JobStore
from backend.settings_store import SettingsStore
from backend.state_machine import JobStatus, TERMINAL_STATES


class _FakeTrainingService:
    """Stands in for pipeline/service/app.py's /jobs/{id}/status: answers
    `code` (404 = unknown job) or, for 200, {"exit_code": exit_code}."""

    def __init__(self):
        self.code = 404
        self.exit_code = None
        service = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps({"exit_code": service.exit_code} if service.code == 200 else {"detail": "unknown"})
                self.send_response(service.code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body.encode())

            def log_message(self, *args):
                pass

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_port}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self):
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def service():
    svc = _FakeTrainingService()
    yield svc
    svc.close()


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(job_manager.time, "monotonic", lambda: now[0])
    return now


def test_unknown_job_within_grace_keeps_polling(service, clock):
    handle = _RemoteTrainingHandle(service.url, "j1", unknown_job_grace_s=30)
    assert handle.poll() is None
    clock[0] += 29
    assert handle.poll() is None
    assert handle.returncode is None


def test_unknown_job_past_grace_is_reported_as_failed_worker(service, clock):
    handle = _RemoteTrainingHandle(service.url, "j1", unknown_job_grace_s=30)
    assert handle.poll() is None
    clock[0] += 31
    assert handle.poll() == job_manager._WORKER_LOST_EXIT_CODE
    assert handle.returncode == job_manager._WORKER_LOST_EXIT_CODE
    assert "no longer knows this job" in handle.failure_reason
    # sticky: a later 200 must not resurrect the job
    service.code, service.exit_code = 200, None
    assert handle.poll() == job_manager._WORKER_LOST_EXIT_CODE


def test_a_successful_status_resets_the_unknown_job_timer(service, clock):
    handle = _RemoteTrainingHandle(service.url, "j1", unknown_job_grace_s=30)
    assert handle.poll() is None            # 404 starts the timer
    clock[0] += 20
    service.code = 200                      # service answers: still running
    assert handle.poll() is None
    clock[0] += 20
    service.code = 404                      # a fresh 404 streak starts from zero
    assert handle.poll() is None
    clock[0] += 29
    assert handle.poll() is None


def test_exit_code_from_service_is_passed_through(service, clock):
    service.code, service.exit_code = 200, 0
    handle = _RemoteTrainingHandle(service.url, "j1")
    assert handle.poll() == 0


def test_unreachable_service_never_fails_the_handle(clock):
    # Nothing listens here: connection refused is transient (service restarting),
    # unlike a 404 -- it must not be turned into a failure however long it lasts.
    handle = _RemoteTrainingHandle("http://127.0.0.1:1", "j1", unknown_job_grace_s=1)
    assert handle.poll() is None
    clock[0] += 3600
    assert handle.poll() is None


def test_job_manager_marks_job_failed_when_service_forgets_it(tmp_path, monkeypatch, service):
    monkeypatch.setattr(workspace, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(job_manager, "_POLL_INTERVAL_S", 0.02)

    class Client:
        def start(self, job_id, flags, log_file, *, python_executable, entrypoint_module):
            return _RemoteTrainingHandle(service.url, job_id, unknown_job_grace_s=0.3)

    manager = JobManager(
        store=JobStore(":memory:"),
        python_executable=sys.executable,
        training_client=Client(),
        settings_store=SettingsStore(":memory:"),
    )
    job_id = manager.submit(JobConfig())["job_id"]
    manager.start(job_id)

    deadline = time.time() + 10
    while time.time() < deadline:
        record = manager.store.get(job_id)
        if JobStatus(record["status"]) in TERMINAL_STATES:
            break
        time.sleep(0.05)
    assert JobStatus(record["status"]) == JobStatus.FAILED
    assert "no longer knows this job" in record["error"]
    backlog = manager.event_backlog(job_id)
    assert backlog[-1]["event"] == "job_failed"
