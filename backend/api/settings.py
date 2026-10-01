"""Settings endpoints -- see backend/settings_store.py for the persisted
singleton row this router only translates to/from HTTP.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from backend.api.schemas import SettingsUpdate
from backend.settings_store import SettingsStore, SettingsValidationError


def build_router(store: SettingsStore) -> APIRouter:
    router = APIRouter(prefix="/settings", tags=["settings"])

    @router.get("")
    def get_settings() -> Dict[str, Any]:
        return store.get()

    @router.put("")
    def update_settings(body: SettingsUpdate) -> Dict[str, Any]:
        # PUT is a partial update: only fields the caller actually set are
        # applied. `ddl_server` is the one field that could legitimately be
        # cleared back to null, but that isn't exposed here since nothing
        # in this pass needs to unset a previously-configured target --
        # every other field is skipped when absent so it doesn't overwrite
        # a stored value with None.
        fields = {k: v for k, v in body.model_dump(exclude_unset=True).items() if v is not None}
        try:
            return store.update(**fields)
        except SettingsValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    return router
