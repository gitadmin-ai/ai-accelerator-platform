"""SQLite-backed operator settings: one persisted singleton row choosing
"local" or "ddl" (Nebula) storage independently for models, datasets, and
checkpoints, plus the DDL connection details used whenever any of those is
"ddl". Mirrors backend/job_store.py's sqlite conventions (single
connection, threading.Lock, CREATE TABLE IF NOT EXISTS).

There's exactly one row (id=1) -- a global operator preference, not a
per-job/per-resource setting. New uploads and job submissions read it at
call time (see ModelStore/DatasetStore/JobManager), so changing it here
takes effect for whatever happens next, without restarting the backend.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from backend import workspace

_SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    model_storage_backend TEXT NOT NULL DEFAULT 'local',
    dataset_storage_backend TEXT NOT NULL DEFAULT 'local',
    checkpoint_storage_backend TEXT NOT NULL DEFAULT 'local',
    ddl_server TEXT,
    ddl_port INTEGER NOT NULL DEFAULT 58000,
    ddl_tenant INTEGER NOT NULL DEFAULT 1,
    ddl_cpu_base INTEGER NOT NULL DEFAULT 0
)
"""

_DEFAULT_ROW: Dict[str, Any] = {
    "id": 1,
    "model_storage_backend": "local",
    "dataset_storage_backend": "local",
    "checkpoint_storage_backend": "local",
    "ddl_server": None,
    "ddl_port": 58000,
    "ddl_tenant": 1,
    "ddl_cpu_base": 0,
}


class SettingsValidationError(ValueError):
    pass


class SettingsStore:
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

    def get(self) -> Dict[str, Any]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM settings WHERE id = 1").fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO settings (id, model_storage_backend, dataset_storage_backend, "
                    "checkpoint_storage_backend, ddl_server, ddl_port, ddl_tenant, ddl_cpu_base) "
                    "VALUES (1, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        _DEFAULT_ROW["model_storage_backend"], _DEFAULT_ROW["dataset_storage_backend"],
                        _DEFAULT_ROW["checkpoint_storage_backend"], _DEFAULT_ROW["ddl_server"],
                        _DEFAULT_ROW["ddl_port"], _DEFAULT_ROW["ddl_tenant"], _DEFAULT_ROW["ddl_cpu_base"],
                    ),
                )
                self._conn.commit()
                row = self._conn.execute("SELECT * FROM settings WHERE id = 1").fetchone()
        return dict(row)

    def update(self, **fields: Any) -> Dict[str, Any]:
        if not fields:
            return self.get()
        current = self.get()
        merged = {**current, **fields}
        for key in ("model_storage_backend", "dataset_storage_backend", "checkpoint_storage_backend"):
            if merged[key] not in ("local", "ddl"):
                raise SettingsValidationError(f"{key} must be 'local' or 'ddl', got {merged[key]!r}")
            if merged[key] == "ddl" and not merged.get("ddl_server"):
                raise SettingsValidationError(
                    f"{key} cannot be set to 'ddl' without ddl_server also being set"
                )

        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._lock:
            self._conn.execute(f"UPDATE settings SET {cols} WHERE id = 1", (*fields.values(),))
            self._conn.commit()
        return self.get()
