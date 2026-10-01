"""Minimal storage abstraction for resource artifacts (dataset files, model
snapshots) -- keeps app/datasets.py and app/models.py from scattering raw
Path/os calls, the same way training/checkpoint/storage_backend.py keeps
filesystem details out of the checkpoint pipeline.

Deliberately separate from that checkpoint StorageBackend: this one stores
whole named directories of arbitrary files (a dataset's JSONL, a model's HF
snapshot) addressed by resource id, not content-addressed chunked blobs. The
two are unrelated and this module never imports from
training/checkpoint/.

A second, DDL (Nebula)-backed path now exists too, via push_staged_to_ddl()
below -- but it isn't a drop-in fifth LocalResourceStorage-shaped
implementation, because ModelStore/DatasetStore's downloads
(huggingface_hub.snapshot_download, HF `datasets` writes) need a real local
directory to write into regardless of final destination. So the flow is
always: stage locally via LocalResourceStorage exactly as before, then
optionally push that staged directory into DDL and delete the local copy --
see push_staged_to_ddl().
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from nebula_ddl_storage.resource_store import DdlResourceStore


class LocalResourceStorage:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, resource_id: str) -> Path:
        """Returns the directory a resource's files should be written into,
        creating it if necessary.
        """
        d = self.root / resource_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get(self, resource_id: str) -> Path:
        d = self.root / resource_id
        if not d.is_dir():
            raise FileNotFoundError(resource_id)
        return d

    def list(self) -> List[str]:
        return [p.name for p in self.root.iterdir() if p.is_dir()]

    def delete(self, resource_id: str) -> None:
        shutil.rmtree(self.root / resource_id, ignore_errors=True)


def push_staged_to_ddl(
    namespace: str, resource_id: str, staged_dir: Path, settings: Dict[str, Any],
) -> None:
    """Pushes a locally-staged resource directory into DDL and deletes the
    staged copy, leaving the resource genuinely DDL-resident. `settings` is
    a dict shaped like SettingsStore.get()'s return value -- callers pass
    the already-fetched dict rather than a store, since they've usually
    just read it anyway to decide whether to call this at all.
    """
    store = DdlResourceStore(
        namespace=namespace,
        server=settings["ddl_server"],
        port=settings["ddl_port"],
        tenant=settings["ddl_tenant"],
        ddl_cpu_base=settings["ddl_cpu_base"],
    )
    try:
        store.put_directory(resource_id, staged_dir)
    finally:
        store.close()
    shutil.rmtree(staged_dir, ignore_errors=True)
