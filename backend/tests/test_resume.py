"""Resuming a job from an earlier job's checkpoint: config, validation at submit
time, the launch flags, and the API's error mapping."""
import json
import sys

import pytest
from fastapi.testclient import TestClient

from backend import workspace
from backend.api.main import create_app
from backend.config import JobConfig
from backend.job_manager import InvalidResumeError, JobManager
from backend.job_store import JobStore
from backend.settings_store import SettingsStore
from backend.state_machine import JobStatus


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "RUNS_ROOT", tmp_path / "runs")
    return JobManager(
        store=JobStore(":memory:"),
        python_executable=sys.executable,
        entrypoint_module="backend.tests.fixtures.stub_trainer",
        settings_store=SettingsStore(":memory:"),
    )


def _finished_job(manager, config=None, status=JobStatus.COMPLETED, checkpoints=(("epoch-1-step-18", 1, 18),)):
    """A source job as a real run leaves it: config.json, events with
    checkpoint_saved entries, and a terminal status."""
    record = manager.submit(config or JobConfig(name="source"))
    job_id = record["job_id"]
    with open(workspace.events_path(job_id), "w") as fh:
        for cid, epoch, step in checkpoints:
            fh.write(json.dumps({"event": "checkpoint_saved", "checkpoint_id": cid, "epoch": epoch, "global_step": step}) + "\n")
    manager.store.update_status(job_id, status)
    return job_id


def _resume(source_id, checkpoint_id="latest", **overrides):
    cfg = {"checkpoint": {"resume_from": {"job_id": source_id, "checkpoint_id": checkpoint_id}}, "training": {"epochs": 3}}
    for key, value in overrides.items():
        cfg.setdefault(key, {}).update(value) if isinstance(value, dict) else cfg.__setitem__(key, value)
    return JobConfig(**cfg)


def test_resume_is_off_by_default_and_checkpoint_defaults_to_latest():
    assert JobConfig().checkpoint.resume_from is None
    cfg = JobConfig(checkpoint={"resume_from": {"job_id": "abc"}})
    assert cfg.checkpoint.resume_from.checkpoint_id == "latest"


def test_valid_resume_is_accepted_and_kept_in_the_stored_config(manager):
    source = _finished_job(manager)
    record = manager.submit(_resume(source))
    assert record["status"] == JobStatus.SUBMITTED.value
    stored = json.loads(workspace.config_path(record["job_id"]).read_text())
    assert stored["checkpoint"]["resume_from"] == {"job_id": source, "checkpoint_id": "latest"}


def test_a_failed_source_job_can_still_be_resumed(manager):
    source = _finished_job(manager, status=JobStatus.FAILED)
    manager.submit(_resume(source))


@pytest.mark.parametrize("status", [JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.CHECKPOINTING])
def test_rejects_a_source_job_that_is_still_running(manager, status):
    source = _finished_job(manager, status=status)
    with pytest.raises(InvalidResumeError, match="wait for it to finish"):
        manager.submit(_resume(source))


def test_rejects_an_unknown_source_job(manager):
    with pytest.raises(InvalidResumeError, match="no such job"):
        manager.submit(_resume("doesnotexist"))


def test_rejects_different_checkpoint_storage(manager):
    source = _finished_job(manager, JobConfig(storage={"backend": "local"}))
    with pytest.raises(InvalidResumeError, match="same checkpoint storage"):
        manager.submit(_resume(source, storage={"backend": "ddl"}))


def test_rejects_a_different_base_model(manager):
    source = _finished_job(manager, JobConfig(model={"name": "model-a"}))
    with pytest.raises(InvalidResumeError, match="same base model"):
        manager.submit(_resume(source, model={"name": "model-b"}))


def test_rejects_a_different_lora_rank(manager):
    source = _finished_job(manager, JobConfig(training={"lora_r": 8}))
    with pytest.raises(InvalidResumeError, match="LoRA rank 8"):
        manager.submit(_resume(source, training={"epochs": 3, "lora_r": 16}))


def test_other_hyperparameters_and_the_dataset_may_change(manager):
    source = _finished_job(manager, JobConfig(dataset={"name": "a.jsonl"}, training={"learning_rate": 1e-4, "batch_size": 2}))
    manager.submit(_resume(source, dataset={"name": "b.jsonl"}, training={"epochs": 3, "learning_rate": 5e-5, "batch_size": 4}))


def test_rejects_a_source_with_no_checkpoint(manager):
    source = _finished_job(manager, checkpoints=())
    with pytest.raises(InvalidResumeError, match="did not save any checkpoint"):
        manager.submit(_resume(source))


def test_rejects_an_unknown_checkpoint_id_and_lists_the_real_ones(manager):
    source = _finished_job(manager, checkpoints=(("epoch-1-step-18", 1, 18), ("epoch-2-step-36", 2, 36)))
    with pytest.raises(InvalidResumeError, match="epoch-1-step-18, epoch-2-step-36"):
        manager.submit(_resume(source, checkpoint_id="epoch-9-step-99"))


def test_epochs_must_exceed_what_the_checkpoint_already_completed(manager):
    source = _finished_job(manager, checkpoints=(("epoch-2-step-36", 2, 36),))
    with pytest.raises(InvalidResumeError, match="set epochs above 2"):
        manager.submit(_resume(source, training={"epochs": 2}))
    manager.submit(_resume(source, training={"epochs": 3}))  # one more epoch is fine


def test_latest_means_the_highest_epoch_and_step(manager):
    source = _finished_job(manager, checkpoints=(("epoch-1-step-18", 1, 18), ("epoch-3-step-54", 3, 54), ("epoch-2-step-36", 2, 36)))
    with pytest.raises(InvalidResumeError, match="set epochs above 3"):
        manager.submit(_resume(source, training={"epochs": 3}))


def test_an_older_checkpoint_can_be_chosen_explicitly(manager):
    source = _finished_job(manager, checkpoints=(("epoch-1-step-18", 1, 18), ("epoch-3-step-54", 3, 54)))
    manager.submit(_resume(source, checkpoint_id="epoch-1-step-18", training={"epochs": 2}))


# --- launch flags -----------------------------------------------------------------


def _flag(flags, name):
    return flags[flags.index(name) + 1]


def test_flags_without_resume_are_unchanged(manager):
    flags = manager._build_flags("job-1", JobConfig())
    assert "--resume" not in flags
    assert _flag(flags, "--run-id") == "job-1" and _flag(flags, "--job-id") == "job-1"
    assert _flag(flags, "--checkpoint-local-dir") == str(workspace.checkpoints_dir("job-1"))


def test_local_resume_reads_and_continues_the_source_jobs_checkpoint_run(manager):
    flags = manager._build_flags("job-2", _resume("job-1", "epoch-1-step-18"))
    assert _flag(flags, "--resume") == "epoch-1-step-18"
    assert _flag(flags, "--run-id") == "job-1"  # where the checkpoints live
    assert _flag(flags, "--job-id") == "job-2"  # this job's own events
    assert _flag(flags, "--checkpoint-local-dir") == str(workspace.checkpoints_dir("job-1"))
    assert _flag(flags, "--output-dir") == str(workspace.job_dir("job-2"))
    assert _flag(flags, "--events-file") == str(workspace.events_path("job-2"))
    assert _flag(flags, "--epochs") == "3"


def test_ddl_resume_uses_the_source_jobs_run_id_namespace(manager):
    manager.settings_store.update(ddl_server="nebula.internal", ddl_tenant=7)
    flags = manager._build_flags("job-2", _resume("job-1", storage={"backend": "ddl"}))
    assert _flag(flags, "--checkpoint-storage") == "ddl"
    assert "--checkpoint-local-dir" not in flags
    assert _flag(flags, "--run-id") == "job-1" and _flag(flags, "--resume") == "latest"
    assert _flag(flags, "--ddl-tenant") == "7"


# --- API --------------------------------------------------------------------------


@pytest.fixture
def client(manager):
    return TestClient(create_app(job_manager=manager, settings_store=manager.settings_store))


def test_api_rejects_an_unusable_resume_with_422_and_a_reason(client, manager):
    resp = client.post("/jobs", json={"checkpoint": {"resume_from": {"job_id": "nope"}}})
    assert resp.status_code == 422 and "no such job" in resp.json()["detail"]
    assert manager.store.list() == []  # nothing was created


def test_api_accepts_a_valid_resume(client, manager):
    source = _finished_job(manager)
    resp = client.post("/jobs", json={
        "training": {"epochs": 3}, "checkpoint": {"resume_from": {"job_id": source, "checkpoint_id": "latest"}},
    })
    assert resp.status_code == 200
    assert resp.json()["config"]["checkpoint"]["resume_from"]["job_id"] == source
