"""Recent-activity feed for the Dashboard -- derived entirely from existing
resources (datasets, models, jobs, checkpoint events) at request time. No
separate activity table/log is introduced; nothing here is fabricated.
"""
from __future__ import annotations

from typing import Any, Dict, List

from backend.datasets import DatasetStore
from backend.job_manager import JobManager
from backend.models import ModelStore

_RECENT_JOBS_SCANNED = 10  # cap events.jsonl reads for checkpoint activity


def build_recent_activity(
    job_manager: JobManager, dataset_store: DatasetStore, model_store: ModelStore, limit: int = 15
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []

    for d in dataset_store.list():
        events.append({
            "type": "dataset_created",
            "message": f"Dataset \"{d['name']}\" created ({d['source']}, {d['num_examples']} examples)",
            "timestamp": d["created_at"],
        })

    for m in model_store.list():
        if m["status"] == "ready" and m["completed_at"]:
            events.append({
                "type": "model_ready", "message": f"Model \"{m['name']}\" downloaded and ready",
                "timestamp": m["completed_at"],
            })
        elif m["status"] == "error" and m["completed_at"]:
            events.append({
                "type": "model_error", "message": f"Model \"{m['name']}\" download failed",
                "timestamp": m["completed_at"],
            })

    for j in job_manager.store.list()[:_RECENT_JOBS_SCANNED]:
        if j.get("started_at"):
            events.append({
                "type": "job_started", "message": f"Training job \"{j['name']}\" started",
                "timestamp": j["started_at"],
            })
        if j["status"] == "COMPLETED" and j.get("completed_at"):
            events.append({
                "type": "job_completed", "message": f"Training job \"{j['name']}\" completed",
                "timestamp": j["completed_at"],
            })
        if j["status"] == "FAILED" and j.get("completed_at"):
            events.append({
                "type": "job_failed", "message": f"Training job \"{j['name']}\" failed",
                "timestamp": j["completed_at"],
            })
        for cp in job_manager.get_checkpoints(j["job_id"])[-2:]:
            events.append({
                "type": "checkpoint_saved",
                "message": f"Checkpoint saved for \"{j['name']}\" (epoch {cp.get('epoch')})",
                "timestamp": cp.get("ts", j.get("completed_at") or j["created_at"]),
            })

    events.sort(key=lambda e: e["timestamp"], reverse=True)
    return events[:limit]
