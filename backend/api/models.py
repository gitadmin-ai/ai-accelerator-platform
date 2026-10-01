"""Model resource endpoints -- see app/models.py for the download logic this
router only translates to/from HTTP.
"""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Response

from backend.models import (
    ModelDeleteConflictError,
    ModelNameConflictError,
    ModelNotFoundError,
    ModelStore,
    ModelValidationError,
)


def build_router(store: ModelStore) -> APIRouter:
    router = APIRouter(prefix="/models", tags=["models"])

    @router.get("")
    def list_models() -> List[Dict[str, Any]]:
        return store.list()

    @router.get("/{model_id}")
    def get_model(model_id: str) -> Dict[str, Any]:
        try:
            return store.get(model_id)
        except ModelNotFoundError:
            raise HTTPException(status_code=404, detail=f"model {model_id!r} not found")

    @router.post("")
    def download_model(body: Dict[str, str]) -> Dict[str, Any]:
        name = body.get("name", "")
        repo_id = body.get("repo_id", "")
        try:
            return store.create_download(name, repo_id)
        except ModelNameConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except ModelValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.delete("/{model_id}", status_code=204)
    def delete_model(model_id: str) -> Response:
        try:
            store.delete(model_id)
        except ModelNotFoundError:
            raise HTTPException(status_code=404, detail=f"model {model_id!r} not found")
        except ModelDeleteConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return Response(status_code=204)

    return router
