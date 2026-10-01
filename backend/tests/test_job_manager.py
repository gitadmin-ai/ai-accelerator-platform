import sys
import time

import pytest

from backend import workspace
from backend.config import JobConfig
from backend.job_manager import JobDeleteConflictError, JobManager, JobNotFoundError
from backend.job_store import JobStore
from backend.settings_store import SettingsStore
from backend.state_machine import IllegalTransitionError, JobStatus, TERMINAL_STATES


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "RUNS_ROOT", tmp_path / "runs")
    return JobManager(
        store=JobStore(":memory:"),
        python_executable=sys.executable,
        entrypoint_module="backend.tests.fixtures.stub_trainer",
        settings_store=SettingsStore(":memory:"),
    )


def _wait_for_terminal(manager: JobManager, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = manager.store.get(job_id)
        if JobStatus(record["status"]) in TERMINAL_STATES:
            return record
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not reach a terminal state within {timeout}s")


def test_submit_creates_job_in_submitted_state(manager):
    record = manager.submit(JobConfig(name="test-job"))
    assert record["status"] == JobStatus.SUBMITTED.value
    assert workspace.config_path(record["job_id"]).exists()


def test_build_flags_passes_lora_hyperparameters(manager):
    config = JobConfig(training={"lora_r": 32, "lora_alpha": 64, "lora_dropout": 0.1})
    flags = manager._build_flags("job-1", config)
    assert "--lora-r" in flags and flags[flags.index("--lora-r") + 1] == "32"
    assert "--lora-alpha" in flags and flags[flags.index("--lora-alpha") + 1] == "64"
    assert "--lora-dropout" in flags and flags[flags.index("--lora-dropout") + 1] == "0.1"


def test_build_flags_defaults_to_local_checkpoint_storage(manager):
    flags = manager._build_flags("job-1", JobConfig())
    assert flags[flags.index("--checkpoint-storage") + 1] == "local"
    assert "--checkpoint-local-dir" in flags
    assert "--ddl-server" not in flags
    assert flags[flags.index("--model-source") + 1] == "local"
    assert flags[flags.index("--dataset-source") + 1] == "local"


def test_build_flags_emits_ddl_checkpoint_storage_when_configured(manager):
    manager.settings_store.update(ddl_server="nebula.internal", ddl_port=59000, ddl_tenant=2, ddl_cpu_base=4)
    config = JobConfig(storage={"backend": "ddl"})
    flags = manager._build_flags("job-1", config)
    assert flags[flags.index("--checkpoint-storage") + 1] == "ddl"
    assert "--checkpoint-local-dir" not in flags
    assert flags[flags.index("--ddl-server") + 1] == "nebula.internal"
    assert flags[flags.index("--ddl-port") + 1] == "59000"
    assert flags[flags.index("--ddl-tenant") + 1] == "2"
    assert flags[flags.index("--ddl-cpu-base") + 1] == "4"


def test_build_flags_emits_ddl_model_and_dataset_source(manager):
    manager.settings_store.update(ddl_server="nebula.internal")
    config = JobConfig(model={"name": "model-id-123", "source": "ddl"}, dataset={"name": "dataset-id-456", "source": "ddl"})
    flags = manager._build_flags("job-1", config)
    assert flags[flags.index("--model-source") + 1] == "ddl"
    assert flags[flags.index("--model-path") + 1] == "model-id-123"
    assert flags[flags.index("--dataset-source") + 1] == "ddl"
    assert flags[flags.index("--dataset") + 1] == "dataset-id-456"
    # checkpoint storage is independent and still defaults to local
    assert flags[flags.index("--checkpoint-storage") + 1] == "local"


def test_build_flags_raises_when_ddl_requested_without_server(manager):
    config = JobConfig(storage={"backend": "ddl"})
    with pytest.raises(ValueError):
        manager._build_flags("job-1", config)


def test_start_missing_job_raises(manager):
    with pytest.raises(JobNotFoundError):
        manager.start("does-not-exist")


def test_start_twice_raises_illegal_transition(manager):
    record = manager.submit(JobConfig())
    manager.start(record["job_id"])
    with pytest.raises(IllegalTransitionError):
        manager.start(record["job_id"])


def test_full_lifecycle_reaches_completed(manager):
    record = manager.submit(JobConfig(name="stub-success"))
    job_id = record["job_id"]
    manager.start(job_id)

    final = _wait_for_terminal(manager, job_id)
    assert final["status"] == JobStatus.COMPLETED.value
    assert final["started_at"] is not None
    assert final["completed_at"] is not None
    assert final["last_event"]["event"] == "job_complete"

    checkpoints = manager.get_checkpoints(job_id)
    assert len(checkpoints) == 1
    assert checkpoints[0]["checkpoint_id"] == "epoch-0-step-1"
    assert checkpoints[0]["size_bytes"] == 1024


def test_failed_job_transitions_to_failed(manager):
    record = manager.submit(JobConfig(dataset={"name": "STUB_FAIL"}))
    job_id = record["job_id"]
    manager.start(job_id)

    final = _wait_for_terminal(manager, job_id)
    assert final["status"] == JobStatus.FAILED.value
    assert final["error"] is not None
    assert "simulated failure" in final["error"]


def test_silent_crash_produces_a_persisted_failed_event(manager):
    # The stub exits nonzero without emitting anything -- JobManager's own
    # synthetic-FAILED fallback must fire, and (Phase 2 fix) must persist
    # that event to events.jsonl, not just publish it live.
    record = manager.submit(JobConfig(dataset={"name": "STUB_CRASH_SILENT"}))
    job_id = record["job_id"]
    manager.start(job_id)

    final = _wait_for_terminal(manager, job_id)
    assert final["status"] == JobStatus.FAILED.value
    assert "code 7" in final["error"]

    backlog = manager.event_backlog(job_id)
    assert backlog, "synthetic FAILED event must be written to events.jsonl"
    assert backlog[-1]["event"] == "job_failed"
    assert backlog[-1]["stage"] == JobStatus.FAILED.value


def test_events_are_tailed_in_order(manager):
    record = manager.submit(JobConfig())
    job_id = record["job_id"]
    q = manager.subscribe(job_id)
    manager.start(job_id)
    _wait_for_terminal(manager, job_id)

    stages_seen = []
    while not q.empty():
        event = q.get_nowait()
        if "stage" in event and event["stage"]:
            stages_seen.append(event["stage"])

    assert "RUNNING" in stages_seen
    assert "CHECKPOINTING" in stages_seen
    assert "EVALUATING" in stages_seen
    assert stages_seen[-1] == "COMPLETED"


def test_delete_missing_job_raises(manager):
    with pytest.raises(JobNotFoundError):
        manager.delete("does-not-exist")


def test_delete_submitted_job_removes_record_and_workspace(manager):
    record = manager.submit(JobConfig(name="never-started"))
    job_id = record["job_id"]
    assert workspace.job_dir(job_id).exists()

    manager.delete(job_id)

    assert manager.store.get(job_id) is None
    assert not workspace.job_dir(job_id).exists()


def test_delete_completed_job_removes_record_and_workspace(manager):
    record = manager.submit(JobConfig(name="stub-success"))
    job_id = record["job_id"]
    manager.start(job_id)
    _wait_for_terminal(manager, job_id)

    manager.delete(job_id)

    assert manager.store.get(job_id) is None
    assert not workspace.job_dir(job_id).exists()


def test_delete_running_job_is_rejected(manager):
    record = manager.submit(JobConfig(name="still-running"))
    job_id = record["job_id"]
    manager.start(job_id)  # moves to QUEUED, then the stub runs on a background thread

    with pytest.raises(JobDeleteConflictError):
        manager.delete(job_id)

    # Rejected deletes must not touch the record or the workspace.
    assert manager.store.get(job_id) is not None
    assert workspace.job_dir(job_id).exists()

    _wait_for_terminal(manager, job_id)
