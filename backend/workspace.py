"""Per-job workspace layout on local disk -- the "Local Storage" box in the
target architecture diagram. `checkpoints_dir(job_id)` is handed straight to
pipeline.checkpoint.localfs_backend.LocalFsBackend as its root; nothing
else in this module touches the storage-provider abstraction.

    runs/<job_id>/
        config.json          the JobConfig this job was submitted with
        events.jsonl          JsonlEventEmitter output (training/utils/events.py)
        train.log              subprocess stdout/stderr
        checkpoints/            LocalFsBackend root
        artifacts/final_adapter/   final LoRA adapter (peft save_pretrained)

RUNS_ROOT and DATA_ROOT default to repo-relative paths but are overridable
via NEBULA_RUNS_ROOT / NEBULA_DATA_ROOT -- in production the backend and the
pipeline's training-service run on different nodes and point both env vars
at the same shared network mount (e.g. NFS) so events.jsonl, checkpoints,
and downloaded models/datasets are visible to whichever node wrote them.
Tests can still monkeypatch `backend.workspace.RUNS_ROOT` to a tmp_path
directly, same as before.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = Path(os.environ["NEBULA_RUNS_ROOT"]) if os.environ.get("NEBULA_RUNS_ROOT") else REPO_ROOT / "runs"
DATA_ROOT = Path(os.environ["NEBULA_DATA_ROOT"]) if os.environ.get("NEBULA_DATA_ROOT") else REPO_ROOT / "data"
DATASETS_ROOT = DATA_ROOT / "datasets"
MODELS_ROOT = DATA_ROOT / "models"
# The one dataset that predates the Datasets resource (Phase 1's synthetic
# demo data) -- registered in place rather than copied, see app/datasets.py.
LEGACY_DEMO_DATASET = DATA_ROOT / "demo_onboarding_v1.jsonl"
RESOURCE_DB_PATH = DATA_ROOT / "nebula.db"


def job_dir(job_id: str) -> Path:
    return RUNS_ROOT / job_id


def config_path(job_id: str) -> Path:
    return job_dir(job_id) / "config.json"


def events_path(job_id: str) -> Path:
    return job_dir(job_id) / "events.jsonl"


def log_path(job_id: str) -> Path:
    return job_dir(job_id) / "train.log"


def checkpoints_dir(job_id: str) -> Path:
    return job_dir(job_id) / "checkpoints"


def artifacts_dir(job_id: str) -> Path:
    return job_dir(job_id) / "artifacts"


def final_adapter_dir(job_id: str) -> Path:
    return artifacts_dir(job_id) / "final_adapter"


def ensure_job_dir(job_id: str) -> Path:
    d = job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    return d
