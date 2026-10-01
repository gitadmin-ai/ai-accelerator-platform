"""Dataset resource: SQLite-backed metadata (mirrors app/job_store.py's
pattern) over files persisted through LocalResourceStorage.

Every dataset -- however it arrived (local upload, Hugging Face import) --
ends up as one JSONL file understood by the existing training pipeline's
load_training_dataset() (a "text" field, or the first column if none), so
nothing downstream of this module needs to know or care where a dataset
came from. Training never talks to Hugging Face directly; only this import
path does (see create_from_huggingface).
"""
from __future__ import annotations

import json
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

_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL,
    format TEXT NOT NULL,
    path TEXT,
    storage_backend TEXT NOT NULL DEFAULT 'local',
    size_bytes INTEGER NOT NULL,
    num_examples INTEGER NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    hf_repo TEXT,
    created_at REAL NOT NULL
)
"""


class DatasetNotFoundError(KeyError):
    pass


class DatasetNameConflictError(ValueError):
    pass


class DatasetValidationError(ValueError):
    pass


def _count_jsonl_examples(path: Path) -> int:
    count = 0
    with open(path, "r") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {lineno} is not valid JSON: {exc}") from exc
            count += 1
    return count


def _format_example(row: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort readable rendering for the dataset preview. The existing
    demo data (and most instruction-tuning JSONL) uses a single "text" field
    shaped like "### Instruction:\\n...\\n\\n### Response:\\n..."; split that
    into user/assistant turns when present, otherwise show the raw row.
    """
    text = row.get("text") if isinstance(row, dict) else None
    if isinstance(text, str) and "### Instruction:" in text and "### Response:" in text:
        instruction = text.split("### Instruction:", 1)[1].split("### Response:", 1)[0].strip()
        response = text.split("### Response:", 1)[1].strip()
        return {"user": instruction, "assistant": response}
    if isinstance(text, str):
        return {"text": text}
    return {"raw": row}


def _write_metadata_json(
    target_dir: Path, dataset_id: str, name: str, source: str, fmt: str,
    size_bytes: int, num_examples: int, hf_repo: Optional[str],
) -> None:
    # Filesystem-visible companion to the SQLite row (Section 6's target
    # layout) -- SQLite remains the source of truth the API actually reads;
    # this is written once for transparency/debuggability, never parsed back.
    meta = {
        "id": dataset_id, "name": name, "source": source, "format": fmt,
        "file": "dataset.jsonl", "size": size_bytes, "num_examples": num_examples,
        "created_at": time.time(), "status": "ready", "hf_repo": hf_repo,
    }
    (target_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


class DatasetStore:
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
            # has a datasets table CREATE TABLE IF NOT EXISTS won't touch.
            ensure_column(self._conn, "datasets", "storage_backend", "storage_backend TEXT NOT NULL DEFAULT 'local'")
            self._conn.commit()
        self.storage = storage or LocalResourceStorage(workspace.DATASETS_ROOT)
        # Consulted at creation time (not fixed here) so a Settings change
        # takes effect for the next dataset without restarting the backend.
        self.settings_store = settings_store or SettingsStore()
        self._seed_legacy_demo_dataset()

    # ------------------------------------------------------------------
    def _seed_legacy_demo_dataset(self) -> None:
        """Registers the Phase 1 synthetic demo dataset (real project data
        that predates the Datasets resource) in place, without copying it --
        idempotent, so this is a no-op on every startup after the first.
        """
        path = workspace.LEGACY_DEMO_DATASET
        if not path.exists() or self._get_by_name("demo_onboarding_v1") is not None:
            return
        try:
            num_examples = _count_jsonl_examples(path)
        except ValueError:
            return
        self._insert(
            dataset_id=uuid.uuid4().hex[:12], name="demo_onboarding_v1", source="local", fmt="jsonl",
            path=str(path), storage_backend="local", size_bytes=path.stat().st_size,
            num_examples=num_examples, status="ready", hf_repo=None,
        )

    def _insert(
        self, *, dataset_id, name, source, fmt, path, storage_backend, size_bytes,
        num_examples, status, hf_repo, error=None,
    ):
        with self._lock:
            self._conn.execute(
                "INSERT INTO datasets (id, name, source, format, path, storage_backend, size_bytes, "
                "num_examples, status, error, hf_repo, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    dataset_id, name, source, fmt, path, storage_backend, size_bytes,
                    num_examples, status, error, hf_repo, time.time(),
                ),
            )
            self._conn.commit()
        return self.get(dataset_id)

    def _get_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM datasets WHERE name = ?", (name,)).fetchone()
        return dict(row) if row is not None else None

    def get(self, dataset_id: str) -> Dict[str, Any]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        if row is None:
            raise DatasetNotFoundError(dataset_id)
        return dict(row)

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def abs_path(self, record: Dict[str, Any]) -> Path:
        if record["storage_backend"] == "ddl":
            raise DatasetValidationError(
                f"dataset {record['id']!r} is stored on Nebula (DDL), not local disk -- "
                "no local path is available for it"
            )
        return Path(record["path"])

    def _finalize_storage(self, dataset_id: str, target_dir: Path) -> "tuple[Optional[str], str]":
        """After a dataset's files have been fully staged locally, either
        leaves them there (local) or pushes them to DDL and deletes the
        staging copy (ddl). Returns (path, storage_backend) for _insert().
        """
        settings = self.settings_store.get()
        storage_backend = settings["dataset_storage_backend"]
        if storage_backend == "ddl":
            push_staged_to_ddl("datasets", dataset_id, target_dir, settings)
            return None, storage_backend
        return str(target_dir / "dataset.jsonl"), storage_backend

    # ------------------------------------------------------------------
    def create_local(self, name: str, filename: str, content: bytes) -> Dict[str, Any]:
        if not name.strip():
            raise DatasetValidationError("dataset name is required")
        if self._get_by_name(name) is not None:
            raise DatasetNameConflictError(f"a dataset named {name!r} already exists")
        if not filename.lower().endswith(".jsonl"):
            raise DatasetValidationError("only .jsonl files are currently supported")

        dataset_id = uuid.uuid4().hex[:12]
        target_dir = self.storage.save(dataset_id)
        target_path = target_dir / "dataset.jsonl"
        target_path.write_bytes(content)

        try:
            num_examples = _count_jsonl_examples(target_path)
        except ValueError as exc:
            self.storage.delete(dataset_id)
            raise DatasetValidationError(f"invalid JSONL file: {exc}") from exc
        if num_examples == 0:
            self.storage.delete(dataset_id)
            raise DatasetValidationError("dataset file has no examples")

        size_bytes = target_path.stat().st_size
        _write_metadata_json(target_dir, dataset_id, name, "local", "jsonl", size_bytes, num_examples, None)
        path, storage_backend = self._finalize_storage(dataset_id, target_dir)
        return self._insert(
            dataset_id=dataset_id, name=name, source="local", fmt="jsonl",
            path=path, storage_backend=storage_backend,
            size_bytes=size_bytes, num_examples=num_examples, status="ready", hf_repo=None,
        )

    def create_from_huggingface(
        self, name: str, repo_id: str, config_name: Optional[str] = None, split: str = "train",
    ) -> Dict[str, Any]:
        if not name.strip():
            raise DatasetValidationError("dataset name is required")
        if not repo_id.strip():
            raise DatasetValidationError("Hugging Face dataset repository is required")
        if self._get_by_name(name) is not None:
            raise DatasetNameConflictError(f"a dataset named {name!r} already exists")

        from datasets import load_dataset  # already a project dependency (training pipeline)

        try:
            # Some HF datasets (e.g. nyu-mll/glue) are multi-config and
            # reject load_dataset(repo_id, split=...) with no config name --
            # config_name is None for the common single-config case, which
            # load_dataset accepts as its second positional arg unchanged.
            # Not every config has a "train" split either (glue's "ax"
            # diagnostic config only has "test") -- split defaults to
            # "train" but is caller-overridable for those cases; the
            # exception message on a bad split already lists what's valid.
            raw = load_dataset(repo_id, config_name, split=split)
        except Exception as exc:  # noqa: BLE001 - surfaced as a human-readable 400/502, not a 500
            raise DatasetValidationError(f"could not load Hugging Face dataset {repo_id!r}: {exc}") from exc

        text_field = "text" if "text" in raw.column_names else raw.column_names[0]
        dataset_id = uuid.uuid4().hex[:12]
        target_dir = self.storage.save(dataset_id)
        target_path = target_dir / "dataset.jsonl"
        num_examples = 0
        with open(target_path, "w") as fh:
            for row in raw:
                fh.write(json.dumps({"text": row[text_field]}) + "\n")
                num_examples += 1

        if num_examples == 0:
            self.storage.delete(dataset_id)
            raise DatasetValidationError(f"Hugging Face dataset {repo_id!r} has no rows")

        size_bytes = target_path.stat().st_size
        _write_metadata_json(target_dir, dataset_id, name, "huggingface", "jsonl", size_bytes, num_examples, repo_id)
        path, storage_backend = self._finalize_storage(dataset_id, target_dir)
        return self._insert(
            dataset_id=dataset_id, name=name, source="huggingface", fmt="jsonl",
            path=path, storage_backend=storage_backend,
            size_bytes=size_bytes, num_examples=num_examples, status="ready", hf_repo=repo_id,
        )

    def delete(self, dataset_id: str) -> None:
        self.get(dataset_id)  # raises DatasetNotFoundError if missing
        with self._lock:
            self._conn.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
            self._conn.commit()
        # No-ops for the legacy demo dataset (its path lives outside
        # storage.root, addressed by nothing storage.delete() would touch) --
        # correctly leaves that real, pre-existing project file on disk.
        self.storage.delete(dataset_id)

    def preview(self, dataset_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        record = self.get(dataset_id)
        path = self.abs_path(record)
        examples: List[Dict[str, Any]] = []
        with open(path, "r") as fh:
            for line in fh:
                if len(examples) >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                examples.append(_format_example(row))
        return examples
