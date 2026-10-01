"""Real, end-to-end Phase 1 acceptance test.

Submits an actual fine-tuning job (Qwen2.5-0.5B + the synthetic demo
dataset) through JobManager exactly as the API would -- a real model
download, real LoRA training, a real chunked local-filesystem checkpoint,
and a real held-out-loss evaluation -- and asserts every Phase 1
acceptance-criteria item from the approved plan.

Opt-in only (downloads a real model, trains for real -- a few minutes on
CPU), matching this repo's existing convention of skipping tests that need
resources not present by default (see
training/tests/test_blobstore_backend.py's native-extension skip). Run
explicitly with:

    RUN_REAL_E2E_TEST=1 python3 -m pytest app/tests/test_e2e_real_training.py -v -s
"""
from __future__ import annotations

import os
import sys
import time

import pytest

if not os.environ.get("RUN_REAL_E2E_TEST"):
    pytest.skip(
        "set RUN_REAL_E2E_TEST=1 to run the real (slow, model-downloading) e2e test",
        allow_module_level=True,
    )

from backend import workspace
from backend.config import JobConfig
from backend.job_manager import JobManager
from backend.job_store import JobStore
from backend.settings_store import SettingsStore
from backend.state_machine import JobStatus, TERMINAL_STATES
from nebula_events import read_events


def _wait_for_terminal(manager: JobManager, job_id: str, timeout: float = 900.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = manager.store.get(job_id)
        if JobStatus(record["status"]) in TERMINAL_STATES:
            return record
        time.sleep(1.0)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def test_real_qwen_lora_job_completes_full_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "RUNS_ROOT", tmp_path / "runs")
    manager = JobManager(
        store=JobStore(":memory:"), python_executable=sys.executable, settings_store=SettingsStore(":memory:")
    )

    config = JobConfig(
        name="e2e-real-test",
        model={"name": "Qwen/Qwen2.5-0.5B"},
        dataset={"name": str(workspace.REPO_ROOT / "data" / "demo_onboarding_v1.jsonl")},
        training={"epochs": 1, "batch_size": 2, "gradient_accumulation_steps": 1, "max_seq_length": 64},
        evaluation={"enabled": True, "split_ratio": 0.2},
    )
    record = manager.submit(config)
    job_id = record["job_id"]
    manager.start(job_id)

    final = _wait_for_terminal(manager, job_id)
    assert final["status"] == JobStatus.COMPLETED.value, final.get("error")

    events = read_events(workspace.events_path(job_id))
    event_names = [e["event"] for e in events]
    for expected in [
        "worker_started", "model_loaded", "dataset_loaded", "epoch_start",
        "checkpoint_start", "checkpoint_saved", "eval_start", "eval_complete",
        "job_complete",
    ]:
        assert expected in event_names, f"missing event {expected!r} in {event_names}"

    checkpoints = manager.get_checkpoints(job_id)
    assert len(checkpoints) == 1
    assert checkpoints[0]["size_bytes"] > 0
    assert checkpoints[0]["throughput_mb_s"] > 0

    eval_events = [e for e in events if e["event"] == "eval_complete"]
    assert eval_events and isinstance(eval_events[0]["eval_loss"], float)

    artifact_dir = workspace.artifacts_dir(job_id) / "final_adapter"
    assert artifact_dir.exists()
    assert any(artifact_dir.glob("adapter_model.*"))
