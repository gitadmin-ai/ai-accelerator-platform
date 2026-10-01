"""Thin HTTP wrapper around pipeline.train_lora_with_gpu_stats, run on the
GPU node so backend.job_manager.HttpTrainingClient can trigger a training
run without needing direct filesystem/process access to that node.

Does exactly what backend.job_manager.SubprocessTrainingClient does
locally -- subprocess.Popen([sys.executable, "-m",
"pipeline.train_lora_with_gpu_stats", *flags]) -- just from a process
running on the GPU node instead. The worker still writes events.jsonl and
checkpoints to the paths given in `flags` (--events-file,
--checkpoint-local-dir, ...), which resolve onto the shared network mount
both nodes see at the same path -- this service never reads or forwards
that data itself.
"""
from __future__ import annotations

import subprocess
import sys
import threading
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

_ENTRYPOINT_MODULE = "pipeline.train_lora_with_gpu_stats"


class StartJobRequest(BaseModel):
    flags: List[str]


class _ProcessRegistry:
    """Tracks one subprocess.Popen per job_id so /status can report whether
    it's still running without this service needing its own event channel.
    """

    def __init__(self) -> None:
        self._processes: Dict[str, "subprocess.Popen"] = {}
        self._lock = threading.Lock()

    def start(self, job_id: str, flags: List[str]) -> None:
        argv = [sys.executable, "-m", _ENTRYPOINT_MODULE, *flags]
        proc = subprocess.Popen(argv)
        with self._lock:
            self._processes[job_id] = proc

    def exit_code(self, job_id: str) -> Optional[int]:
        with self._lock:
            proc = self._processes.get(job_id)
        if proc is None:
            raise KeyError(job_id)
        return proc.poll()


def create_app(registry: Optional[_ProcessRegistry] = None) -> FastAPI:
    registry = registry or _ProcessRegistry()
    app = FastAPI(title="Nebula Pipeline Training Service")
    app.state.registry = registry

    @app.get("/healthz")
    def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/jobs/{job_id}/start", status_code=202)
    def start_job(job_id: str, request: StartJobRequest) -> Dict[str, str]:
        try:
            registry.start(job_id, request.flags)
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"failed to launch worker: {exc}") from exc
        return {"job_id": job_id, "status": "started"}

    @app.get("/jobs/{job_id}/status")
    def job_status(job_id: str) -> Dict[str, Optional[int]]:
        try:
            exit_code = registry.exit_code(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown job_id {job_id!r}")
        return {"exit_code": exit_code}

    return app


app = create_app()
