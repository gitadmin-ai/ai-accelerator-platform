"""Dataset resource endpoints -- see app/datasets.py for the storage/import
logic this router only translates to/from HTTP. Mounted by create_app() in
app/api/main.py, following the same "router owns no business logic" split
job_manager.py already has from the job endpoints.
"""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile

from backend.datasets import (
    DatasetNameConflictError,
    DatasetNotFoundError,
    DatasetStore,
    DatasetValidationError,
)


def build_router(store: DatasetStore) -> APIRouter:
    router = APIRouter(prefix="/datasets", tags=["datasets"])

    @router.get("")
    def list_datasets() -> List[Dict[str, Any]]:
        return store.list()

    @router.get("/{dataset_id}")
    def get_dataset(dataset_id: str) -> Dict[str, Any]:
        try:
            return store.get(dataset_id)
        except DatasetNotFoundError:
            raise HTTPException(status_code=404, detail=f"dataset {dataset_id!r} not found")

    @router.get("/{dataset_id}/preview")
    def preview_dataset(dataset_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        try:
            return store.preview(dataset_id, limit=min(max(limit, 1), 20))
        except DatasetNotFoundError:
            raise HTTPException(status_code=404, detail=f"dataset {dataset_id!r} not found")
        except DatasetValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("")
    async def upload_dataset(name: str = Form(...), file: UploadFile = File(...)) -> Dict[str, Any]:
        content = await file.read()
        try:
            return store.create_local(name, file.filename or "", content)
        except DatasetNameConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except DatasetValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/import")
    def import_dataset(body: Dict[str, str]) -> Dict[str, Any]:
        name = body.get("name", "")
        repo_id = body.get("repo_id", "")
        config_name = body.get("config_name") or None
        split = body.get("split") or "train"
        try:
            return store.create_from_huggingface(name, repo_id, config_name=config_name, split=split)
        except DatasetNameConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except DatasetValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.delete("/{dataset_id}", status_code=204)
    def delete_dataset(dataset_id: str) -> Response:
        try:
            store.delete(dataset_id)
        except DatasetNotFoundError:
            raise HTTPException(status_code=404, detail=f"dataset {dataset_id!r} not found")
        return Response(status_code=204)

    return router
