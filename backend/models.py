"""Model resource: SQLite-backed metadata (mirrors app/job_store.py's
pattern) over Hugging Face snapshots persisted through LocalResourceStorage.

A download can take minutes for a multi-GB model, so create_download()
returns immediately with status="pending" and does the real work
(huggingface_hub.snapshot_download) on a background thread -- the same
"launch and tail state" shape app/job_manager.py already uses for training
subprocesses, just without a subprocess. Callers poll get()/list() for the
pending -> downloading -> ready|error transition; there is no fabricated
percentage, only real state.

Once a model reaches "ready", its `path` is a local directory that
transformers.AutoModelForCausalLM.from_pretrained() accepts exactly like a
Hugging Face hub id -- training never needs to know a download happened.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend import workspace
from backend.db_migrate import ensure_column
from backend.resource_storage import LocalResourceStorage, push_staged_to_ddl
from backend.settings_store import SettingsStore

logger = logging.getLogger("models")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL,
    repo_id TEXT NOT NULL,
    path TEXT,
    storage_backend TEXT NOT NULL DEFAULT 'local',
    size_bytes INTEGER,
    architecture TEXT,
    status TEXT NOT NULL,
    error TEXT,
    created_at REAL NOT NULL,
    completed_at REAL
)
"""


class ModelNotFoundError(KeyError):
    pass


class ModelNameConflictError(ValueError):
    pass


class ModelValidationError(ValueError):
    pass


class ModelDeleteConflictError(ValueError):
    pass


def _read_architecture(target_dir: Path) -> Optional[str]:
    config_path = target_dir / "config.json"
    if not config_path.exists():
        return None
    try:
        config = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    architectures = config.get("architectures")
    if architectures:
        return ", ".join(architectures)
    return config.get("model_type")


class ModelStore:
    def __init__(
        self,
        db_path: Optional[str] = None,
        storage: Optional[LocalResourceStorage] = None,
        settings_store: Optional[SettingsStore] = None,
    ):
        db_path = db_path if db_path is not None else str(workspace.RESOURCE_DB_PATH)
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(_SCHEMA)
            # Migration: a database created before storage_backend existed
            # has a models table CREATE TABLE IF NOT EXISTS won't touch.
            ensure_column(self._conn, "models", "storage_backend", "storage_backend TEXT NOT NULL DEFAULT 'local'")
            self._conn.commit()
        self.storage = storage or LocalResourceStorage(workspace.MODELS_ROOT)
        # Consulted at download time (not fixed here) so a Settings change
        # takes effect for the next download without restarting the backend.
        self.settings_store = settings_store or SettingsStore()

    # ------------------------------------------------------------------
    def _get_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM models WHERE name = ?", (name,)).fetchone()
        return dict(row) if row is not None else None

    def get(self, model_id: str) -> Dict[str, Any]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
        if row is None:
            raise ModelNotFoundError(model_id)
        return dict(row)

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM models ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def abs_path(self, record: Dict[str, Any]) -> Path:
        if record["storage_backend"] == "ddl":
            raise ModelValidationError(
                f"model {record['id']!r} is stored on Nebula (DDL), not local disk -- "
                "no local path is available for it"
            )
        if record["path"] is None:
            raise ModelValidationError(f"model {record['id']!r} is not ready yet")
        return Path(record["path"])

    def delete(self, model_id: str) -> None:
        record = self.get(model_id)  # raises ModelNotFoundError if missing
        if record["status"] in ("pending", "downloading"):
            raise ModelDeleteConflictError(
                f"model {model_id!r} download is still in progress; wait for it to finish before deleting"
            )
        with self._lock:
            self._conn.execute("DELETE FROM models WHERE id = ?", (model_id,))
            self._conn.commit()
        self.storage.delete(model_id)

    def _update(self, model_id: str, **fields: Any) -> None:
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._lock:
            self._conn.execute(f"UPDATE models SET {cols} WHERE id = ?", (*fields.values(), model_id))
            self._conn.commit()

    # ------------------------------------------------------------------
    def create_download(self, name: str, repo_id: str) -> Dict[str, Any]:
        if not name.strip():
            raise ModelValidationError("model name is required")
        if not repo_id.strip():
            raise ModelValidationError("Hugging Face model repository is required")
        if self._get_by_name(name) is not None:
            raise ModelNameConflictError(f"a model named {name!r} already exists")

        model_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._conn.execute(
                "INSERT INTO models (id, name, source, repo_id, path, size_bytes, architecture, "
                "status, error, created_at, completed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (model_id, name, "huggingface", repo_id, None, None, None, "pending", None, time.time(), None),
            )
            self._conn.commit()

        thread = threading.Thread(target=self._download, args=(model_id, repo_id), daemon=True)
        thread.start()
        return self.get(model_id)

    def _download(self, model_id: str, repo_id: str) -> None:
        self._update(model_id, status="downloading")
        target_dir = self.storage.save(model_id)
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=repo_id, local_dir=str(target_dir))
            size_bytes = sum(f.stat().st_size for f in target_dir.rglob("*") if f.is_file())
            architecture = _read_architecture(target_dir)
            meta = {
                "id": model_id, "repo_id": repo_id, "path": "", "size": size_bytes,
                "architecture": architecture, "created_at": time.time(), "status": "ready",
            }
            (target_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

            settings = self.settings_store.get()
            storage_backend = settings["model_storage_backend"]
            if storage_backend == "ddl":
                push_staged_to_ddl("models", model_id, target_dir, settings)
                path = None
            else:
                path = str(target_dir)

            self._update(
                model_id, status="ready", path=path, storage_backend=storage_backend, size_bytes=size_bytes,
                architecture=architecture, completed_at=time.time(),
            )
        except Exception as exc:  # noqa: BLE001 - reported via status/error, not raised (background thread)
            logger.exception("model %s download from %r failed", model_id, repo_id)
            self.storage.delete(model_id)
            self._update(model_id, status="error", error=str(exc), completed_at=time.time())
