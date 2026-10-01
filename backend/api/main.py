"""FastAPI app implementing the API contracts (Section F): Create Job, List
Jobs, Get Job, Start Job, Get Job Metrics (as an SSE event stream), Get
Checkpoints.

This module owns no training or storage logic -- it only talks to
JobManager. `create_app()` takes an optional pre-built JobManager so tests
can inject one pointed at a temp workspace/in-memory store.
"""
from __future__ import annotations

import io
import json
import os
import queue as queue_module
import zipfile
from typing import Any, Dict, Iterator, List, Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.activity import build_recent_activity
from backend.api import datasets as datasets_router
from backend.api import models as models_router
from backend.api import settings as settings_router
from backend.api.schemas import CatalogResponse, CheckpointSummary, JobDetail, JobSummary
from backend.catalog import get_catalog
from backend.config import JobConfig
from backend.datasets import DatasetStore
from backend.job_manager import (
    ArtifactNotFoundError,
    JobDeleteConflictError,
    JobManager,
    JobNotCompletedError,
    JobNotFoundError,
)
from backend.models import ModelStore
from backend.settings_store import SettingsStore
from backend.state_machine import IllegalTransitionError, TERMINAL_STATES

_SSE_HEARTBEAT_S = 15.0
_TERMINAL_VALUES = {s.value for s in TERMINAL_STATES}


def _sse_format(event: Dict[str, Any]) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


def _event_stream(manager: JobManager, job_id: str) -> Iterator[str]:
    # Subscribe before reading the backlog so a live event can't slip
    # through the gap between "read current file contents" and "start
    # listening for new ones" -- worst case an event is delivered twice
    # (backlog + live), never dropped.
    q = manager.subscribe(job_id)
    try:
        backlog = manager.event_backlog(job_id)
        for event in backlog:
            yield _sse_format(event)
            if event.get("stage") in _TERMINAL_VALUES:
                return
        while True:
            try:
                event = q.get(timeout=_SSE_HEARTBEAT_S)
            except queue_module.Empty:
                yield ": keep-alive\n\n"
                continue
            if event.get("event") == "stream_closed":
                return
            yield _sse_format(event)
            if event.get("stage") in _TERMINAL_VALUES:
                return
    finally:
        manager.unsubscribe(job_id, q)


def create_app(
    job_manager: Optional[JobManager] = None,
    dataset_store: Optional[DatasetStore] = None,
    model_store: Optional[ModelStore] = None,
    settings_store: Optional[SettingsStore] = None,
) -> FastAPI:
    settings = settings_store or SettingsStore()
    manager = job_manager or JobManager(settings_store=settings)
    datasets = dataset_store or DatasetStore(settings_store=settings)
    models = model_store or ModelStore(settings_store=settings)
    app = FastAPI(title="Fine-Tuning Pipeline Demo API")
    app.state.job_manager = manager
    app.state.dataset_store = datasets
    app.state.model_store = models
    app.state.settings_store = settings

    # No auth in Phase 1 (explicitly out of scope); the frontend runs on a
    # different origin (Vite dev server locally, a separate deployed origin
    # in prod) than this API. NEBULA_CORS_ORIGINS is a comma-separated list
    # of allowed origins for prod (e.g. "https://app.example.com"); leaving
    # it unset keeps the wide-open "*" default used for local dev.
    cors_origins_env = os.environ.get("NEBULA_CORS_ORIGINS")
    allow_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()] if cors_origins_env else ["*"]
    app.add_middleware(
        CORSMiddleware, allow_origins=allow_origins, allow_methods=["*"], allow_headers=["*"],
    )

    app.include_router(datasets_router.build_router(datasets))
    app.include_router(models_router.build_router(models))
    app.include_router(settings_router.build_router(settings))

    @app.get("/config", response_model=CatalogResponse)
    def get_config() -> Dict[str, Any]:
        return get_catalog()

    @app.get("/activity")
    def get_activity(limit: int = 15) -> List[Dict[str, Any]]:
        return build_recent_activity(manager, datasets, models, limit=min(max(limit, 1), 50))

    @app.post("/jobs", response_model=JobDetail)
    def create_job(config: JobConfig) -> Dict[str, Any]:
        return manager.submit(config)

    @app.get("/jobs", response_model=List[JobSummary])
    def list_jobs() -> List[Dict[str, Any]]:
        return manager.store.list()

    @app.get("/jobs/{job_id}", response_model=JobDetail)
    def get_job(job_id: str) -> Dict[str, Any]:
        record = manager.store.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")
        return record

    @app.post("/jobs/{job_id}/start", response_model=JobDetail)
    def start_job(job_id: str) -> Dict[str, Any]:
        try:
            return manager.start(job_id)
        except JobNotFoundError:
            raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")
        except IllegalTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @app.delete("/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str) -> Response:
        try:
            manager.delete(job_id)
        except JobNotFoundError:
            raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")
        except JobDeleteConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return Response(status_code=204)

    @app.get("/jobs/{job_id}/events")
    def stream_events(job_id: str) -> StreamingResponse:
        if manager.store.get(job_id) is None:
            raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")
        return StreamingResponse(_event_stream(manager, job_id), media_type="text/event-stream")

    @app.get("/jobs/{job_id}/checkpoints", response_model=List[CheckpointSummary])
    def get_checkpoints(job_id: str) -> List[Dict[str, Any]]:
        if manager.store.get(job_id) is None:
            raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")
        return manager.get_checkpoints(job_id)

    @app.get("/jobs/{job_id}/artifact")
    def download_artifact(job_id: str) -> Response:
        try:
            adapter_dir = manager.get_final_adapter_dir(job_id)
        except JobNotFoundError:
            raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")
        except JobNotCompletedError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except ArtifactNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

        # Zipped in memory -- the adapter is a handful of small files (LoRA
        # weights + tokenizer config), not worth a temp file or streaming
        # for a demo endpoint. The client only ever sees job_id in the URL,
        # never the on-disk path.
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(adapter_dir.rglob("*")):
                if path.is_file():
                    zf.write(path, arcname=path.relative_to(adapter_dir))
        return Response(
            content=buffer.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{job_id}-model.zip"'},
        )

    return app


app = create_app()
