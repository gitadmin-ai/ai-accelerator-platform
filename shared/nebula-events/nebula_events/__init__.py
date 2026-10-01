"""Structured lifecycle events for a training run.

The training worker runs as a separate process from the Job Manager (see
backend/job_manager.py) -- possibly on a different node entirely -- and this
is the only channel between the two. Each event is one JSON object per line,
flushed immediately so a tailing reader sees it as soon as it's written --
this is what lets the API stream live pipeline-stage updates to the browser
without parsing free-text logs.

Every event carries a `stage` matching a training-worker-observable subset
of backend/state_machine.py's JobStatus values (RUNNING, CHECKPOINTING,
EVALUATING, COMPLETED, FAILED -- never SUBMITTED/QUEUED, which are Job
Manager-only states the worker doesn't know about) plus an `event` name
and event-specific fields. Nothing here is estimated: every numeric field
passed in by train_lora_with_gpu_stats.py comes from a real measurement
(CheckpointMetrics, perf_counter deltas, an actual loss tensor).

This module is intentionally dependency-free (stdlib only) and lives in its
own package (`nebula-events`) rather than under `backend/` or `pipeline/` --
it's the one piece of code both sides need to agree on, and keeping it a
separate zero-dependency install is what lets `backend` stay installable
without pulling in any part of the `pipeline` (torch/transformers/peft)
package tree.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional, TextIO, Union


class JsonlEventEmitter:
    """Appends one JSON line per event to `path`. If `path` is None, every
    call is a no-op -- callers don't need to special-case "no events file
    requested" (e.g. running train_lora_with_gpu_stats.py directly from the
    CLI without the Job Manager).
    """

    def __init__(self, path: Optional[Union[str, "Path"]], job_id: Optional[str] = None):
        self.job_id = job_id
        self._fh: Optional[TextIO] = None
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(path, "a", buffering=1)  # line-buffered

    def emit(self, stage: str, event: str, **fields: Any) -> None:
        if self._fh is None:
            return
        record = {"ts": time.time(), "job_id": self.job_id, "stage": stage, "event": event}
        record.update(fields)
        self._fh.write(json.dumps(record, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "JsonlEventEmitter":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def read_events(path: Union[str, "Path"]) -> list:
    """Reads every event currently in `path`. Used by tests and by
    JobManager's initial backfill when it starts tailing a file that
    already has content.
    """
    p = Path(path)
    if not p.exists():
        return []
    events = []
    with open(p, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events
