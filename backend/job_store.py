"""SQLite-backed job metadata store.

Deliberately just persistence for the Jobs list/detail views (job_id, name,
model, dataset, status, timestamps, last event, config). It never touches
the events.jsonl file or checkpoint storage -- those are read directly by
JobManager/the API when needed (see job_manager.get_checkpoints()) rather
than duplicated into this store. stdlib sqlite3 only; no new dependency.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend import workspace
from backend.config import JobConfig
from backend.state_machine import JobStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    model TEXT NOT NULL,
    dataset TEXT NOT NULL,
    storage_backend TEXT NOT NULL,
    status TEXT NOT NULL,
    config_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    started_at REAL,
    completed_at REAL,
    error TEXT,
    last_event_json TEXT
)
"""


class JobStore:
    def __init__(self, db_path: Optional[str] = None):
        db_path = db_path if db_path is not None else str(workspace.RESOURCE_DB_PATH)
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(_SCHEMA)
            self._conn.commit()

    def create(self, job_id: str, config: JobConfig) -> Dict[str, Any]:
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (job_id, name, model, dataset, storage_backend, status, "
                "config_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id,
                    config.name,
                    config.model.name,
                    config.dataset.name,
                    config.storage.backend,
                    JobStatus.SUBMITTED.value,
                    config.model_dump_json(),
                    time.time(),
                ),
            )
            self._conn.commit()
        return self.get(job_id)

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        d["config"] = json.loads(d.pop("config_json"))
        last_event_json = d.pop("last_event_json")
        d["last_event"] = json.loads(last_event_json) if last_event_json else None
        started_at = d.get("started_at")
        completed_at = d.get("completed_at")
        if started_at is not None:
            end = completed_at if completed_at is not None else time.time()
            d["duration_s"] = max(0.0, end - started_at)
        else:
            d["duration_s"] = None
        return d

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._row_to_dict(row) if row is not None else None

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
        return [self._row_to_dict(r) for r in rows]

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        started_at: Optional[float] = None,
        completed_at: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        fields = ["status = ?"]
        values: List[Any] = [status.value]
        if started_at is not None:
            fields.append("started_at = ?")
            values.append(started_at)
        if completed_at is not None:
            fields.append("completed_at = ?")
            values.append(completed_at)
        if error is not None:
            fields.append("error = ?")
            values.append(error)
        values.append(job_id)
        with self._lock:
            self._conn.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE job_id = ?", values)
            self._conn.commit()

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            self._conn.commit()

    def set_last_event(self, job_id: str, event: Dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET last_event_json = ? WHERE job_id = ?",
                (json.dumps(event), job_id),
            )
            self._conn.commit()
