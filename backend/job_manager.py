"""Job Manager: submits/launches/tracks fine-tuning jobs.

Delegates the actual launch of pipeline.train_lora_with_gpu_stats to a
TrainingClient (see below) -- either a local subprocess (dev/tests, worker
on the same node) or an HTTP trigger against the pipeline's own service
wrapper (prod, worker on a separate GPU node). This module still never
imports pipeline.checkpoint.blobstore_backend/ddl_backend or anything else
Nebula-specific -- it only ever talks to the training worker's CLI surface
(see _build_flags(), which reads config.storage.backend/model.source/
dataset.source and backend/settings_store.py's DDL connection details to
decide which --checkpoint-storage/--model-source/--dataset-source/--ddl-*
flags to pass) and reads back the JSONL events the worker emits from
workspace.events_path(job_id) -- a path that resolves onto shared storage
in prod, so it works the same regardless of which node actually wrote the
file.

A background thread per running job tails events.jsonl, applies each
event's `stage` to the job's state_machine.JobStatus, and fans the event
out to any SSE subscribers -- this is the only path job status changes
flow through, so the state machine's legal-transition check applies to
every status change a job goes through.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Protocol

from backend import workspace
from backend.config import JobConfig
from backend.job_store import JobStore
from backend.settings_store import SettingsStore
from backend.state_machine import JobStatus, IllegalTransitionError, TERMINAL_STATES, advance
from nebula_events import JsonlEventEmitter

logger = logging.getLogger("job_manager")

_POLL_INTERVAL_S = 0.2
_SSE_HEARTBEAT_S = 15.0


class TrainingHandle(Protocol):
    """Duck-compatible subset of subprocess.Popen that _tail_events needs:
    poll() returns None while the worker is still running, else its exit
    code (0 == success); returncode mirrors the same value once poll() has
    observed completion. This is satisfied by both a real subprocess.Popen
    and a handle that polls a remote training-service over HTTP.
    """

    returncode: Optional[int]

    def poll(self) -> Optional[int]: ...


class TrainingClient(Protocol):
    """Launches a training worker for `job_id` with the given CLI-style
    flags (job-specific args only -- no interpreter/module prefix, each
    implementation supplies its own) and returns a handle to track its
    lifecycle. Implementations never need to touch events.jsonl themselves
    -- the worker always writes it directly (to shared storage in the HTTP
    case), and JobManager tails it the same way regardless of which
    TrainingClient launched the worker.
    """

    def start(
        self, job_id: str, flags: List[str], log_file, *, python_executable: str, entrypoint_module: str
    ) -> TrainingHandle: ...


class SubprocessTrainingClient:
    """Launches the worker as a local child process. Used for local dev and
    tests (including backend/tests/fixtures/stub_trainer.py) where the
    backend and the training worker run on the same node/filesystem.
    """

    def start(
        self, job_id: str, flags: List[str], log_file, *, python_executable: str, entrypoint_module: str
    ) -> "subprocess.Popen":
        argv = [python_executable, "-m", entrypoint_module, *flags]
        return subprocess.Popen(argv, cwd=str(workspace.REPO_ROOT), stdout=log_file, stderr=subprocess.STDOUT)


# Exit code reported for a remote worker the training-service no longer knows
# about; the service's job table is in memory, so a restart (an OOM kill of the
# container, a redeploy) drops it and kills the worker with it.
_WORKER_LOST_EXIT_CODE = -1
_UNKNOWN_JOB_GRACE_S = 30.0


class _RemoteTrainingHandle:
    """Polls the training-service's /jobs/{job_id}/status endpoint instead
    of a local OS process -- everything else about how JobManager treats a
    handle (poll() -> None while running, else an exit code) is identical.

    A 404 means the service has no record of the job. Transient errors
    (connection refused, timeouts) keep returning None -- the service may just
    be restarting -- but a 404 that persists for `unknown_job_grace_s` means the
    worker is gone for good, so poll() reports it as exited instead of leaving
    the job RUNNING forever.
    """

    def __init__(self, base_url: str, job_id: str, *, unknown_job_grace_s: float = _UNKNOWN_JOB_GRACE_S):
        self._url = f"{base_url}/jobs/{job_id}/status"
        self._unknown_job_grace_s = unknown_job_grace_s
        self._unknown_since: Optional[float] = None
        self.returncode: Optional[int] = None
        self.failure_reason: Optional[str] = None

    def poll(self) -> Optional[int]:
        if self.returncode is not None:
            return self.returncode
        try:
            with urllib.request.urlopen(self._url, timeout=10) as resp:
                payload = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return self._poll_unknown_job(exc)
            logger.warning("training-service status check failed for %s: %s", self._url, exc)
            return None
        except (urllib.error.URLError, TimeoutError) as exc:
            logger.warning("training-service status check failed for %s: %s", self._url, exc)
            return None
        self._unknown_since = None
        self.returncode = payload.get("exit_code")
        return self.returncode

    def _poll_unknown_job(self, exc: Exception) -> Optional[int]:
        now = time.monotonic()
        if self._unknown_since is None:
            self._unknown_since = now
        unknown_for = now - self._unknown_since
        if unknown_for < self._unknown_job_grace_s:
            logger.warning("training-service status check failed for %s: %s", self._url, exc)
            return None
        self.failure_reason = (
            f"training-service no longer knows this job (HTTP 404 for {unknown_for:.0f}s); it was "
            "most likely restarted, e.g. the container was OOM-killed, which also kills the worker."
        )
        logger.error("%s (%s)", self.failure_reason, self._url)
        self.returncode = _WORKER_LOST_EXIT_CODE
        return self.returncode


class HttpTrainingClient:
    """Triggers the worker via HTTP against pipeline/service/app.py running
    on the GPU node (production default). `flags` are sent as-is -- the
    service prepends its own interpreter and always runs
    pipeline.train_lora_with_gpu_stats, then does the same subprocess.Popen
    call SubprocessTrainingClient makes locally, just from a process on the
    GPU node instead.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def start(
        self, job_id: str, flags: List[str], log_file, *, python_executable: str, entrypoint_module: str
    ) -> _RemoteTrainingHandle:
        body = json.dumps({"flags": flags}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/jobs/{job_id}/start", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status >= 300:
                raise OSError(f"training-service rejected job {job_id!r}: HTTP {resp.status}")
        return _RemoteTrainingHandle(self.base_url, job_id)


def default_training_client() -> "TrainingClient":
    """NEBULA_TRAINING_BACKEND selects the client: 'http' (the production
    default, talking to TRAINING_SERVICE_URL on the GPU node) or
    'subprocess' (local dev/tests, worker on the same node).
    """
    backend_kind = os.environ.get("NEBULA_TRAINING_BACKEND", "subprocess")
    if backend_kind == "http":
        base_url = os.environ.get("TRAINING_SERVICE_URL")
        if not base_url:
            raise RuntimeError("NEBULA_TRAINING_BACKEND=http requires TRAINING_SERVICE_URL to be set")
        return HttpTrainingClient(base_url)
    if backend_kind != "subprocess":
        raise ValueError(f"unknown NEBULA_TRAINING_BACKEND={backend_kind!r}; expected 'subprocess' or 'http'")
    return SubprocessTrainingClient()


class InvalidResumeError(ValueError):
    """The job asks to resume from a checkpoint it cannot use (unknown or still
    running source job, different storage/model/LoRA rank, no such checkpoint, or
    nothing left to train)."""


class JobNotFoundError(KeyError):
    pass


class JobNotCompletedError(Exception):
    """Raised when an artifact is requested for a job that hasn't reached COMPLETED."""


class ArtifactNotFoundError(Exception):
    """Raised when a job completed but its final artifact isn't on disk."""


class JobDeleteConflictError(Exception):
    """Raised when deleting a job that is still active (queued or running)."""


class JobManager:
    def __init__(
        self,
        store: Optional[JobStore] = None,
        python_executable: Optional[str] = None,
        entrypoint_module: str = "pipeline.train_lora_with_gpu_stats",
        training_client: Optional["TrainingClient"] = None,
        settings_store: Optional["SettingsStore"] = None,
    ):
        self.store = store or JobStore()
        self.python_executable = python_executable or sys.executable
        # Overridable so tests can point at a fast stub entrypoint
        # (backend/tests/fixtures/stub_trainer.py) instead of loading a real
        # model -- production code never sets this to anything else.
        self.entrypoint_module = entrypoint_module
        # Overridable so tests can inject a fake without touching env vars;
        # production picks 'subprocess' vs 'http' via NEBULA_TRAINING_BACKEND
        # (see default_training_client()).
        self.training_client = training_client or default_training_client()
        # Consulted at launch time (not fixed here) for the DDL connection
        # details a job needs when config.storage.backend/model.source/
        # dataset.source is "ddl" -- see _build_flags().
        self.settings_store = settings_store or SettingsStore()
        self._processes: Dict[str, "TrainingHandle"] = {}
        self._subscribers: Dict[str, List["queue.Queue"]] = {}
        self._pub_lock = threading.Lock()

    # ------------------------------------------------------------------
    # submit / start
    # ------------------------------------------------------------------

    def _validate_resume(self, config: JobConfig) -> None:
        """Rejects a resume request that could not work, at submit time and with a
        specific message, rather than letting the worker fail minutes later."""
        resume = config.checkpoint.resume_from
        if resume is None:
            return
        source_id = resume.job_id
        record = self.store.get(source_id)
        if record is None:
            raise InvalidResumeError(f"cannot resume from job {source_id!r}: no such job")
        status = JobStatus(record["status"])
        if status not in TERMINAL_STATES:
            raise InvalidResumeError(
                f"cannot resume from job {source_id!r}: it is {status.value}; wait for it to finish"
            )
        try:
            source = JobConfig.model_validate_json(workspace.config_path(source_id).read_text())
        except (OSError, ValueError) as exc:
            raise InvalidResumeError(f"cannot resume from job {source_id!r}: its config is unreadable ({exc})")

        if config.storage.backend != source.storage.backend:
            raise InvalidResumeError(
                f"job {source_id!r} stored its checkpoints in {source.storage.backend!r} storage; "
                f"this job uses {config.storage.backend!r}. Resume with the same checkpoint storage."
            )
        if (config.model.name, config.model.source) != (source.model.name, source.model.source):
            raise InvalidResumeError(
                f"job {source_id!r} trained model {source.model.name!r}; a checkpoint only fits the same "
                f"base model, but this job selected {config.model.name!r}"
            )
        if config.training.lora_r != source.training.lora_r:
            raise InvalidResumeError(
                f"job {source_id!r} used LoRA rank {source.training.lora_r}; the checkpoint cannot be "
                f"loaded with rank {config.training.lora_r}"
            )

        saved = self.get_checkpoints(source_id)
        if not saved:
            raise InvalidResumeError(f"job {source_id!r} did not save any checkpoint")
        if resume.checkpoint_id == "latest":
            chosen = max(saved, key=lambda c: (c.get("epoch", 0), c.get("global_step", 0)))
        else:
            matches = [c for c in saved if c.get("checkpoint_id") == resume.checkpoint_id]
            if not matches:
                known = ", ".join(c.get("checkpoint_id", "?") for c in saved)
                raise InvalidResumeError(
                    f"job {source_id!r} has no checkpoint {resume.checkpoint_id!r} (it has: {known})"
                )
            chosen = matches[0]
        done = int(chosen.get("epoch", 0))
        if config.training.epochs <= done:
            raise InvalidResumeError(
                f"checkpoint {chosen.get('checkpoint_id')!r} is after epoch {done}; set epochs above {done} "
                f"(epochs is the total to reach) or there is nothing left to train"
            )

    def submit(self, config: JobConfig) -> Dict[str, Any]:
        self._validate_resume(config)
        job_id = uuid.uuid4().hex[:12]
        workspace.ensure_job_dir(job_id)
        workspace.config_path(job_id).write_text(config.model_dump_json(indent=2))
        return self.store.create(job_id, config)

    def start(self, job_id: str) -> Dict[str, Any]:
        record = self.store.get(job_id)
        if record is None:
            raise JobNotFoundError(job_id)
        current = JobStatus(record["status"])
        if current != JobStatus.SUBMITTED:
            raise IllegalTransitionError(
                f"job {job_id!r} is {current.value}; only a SUBMITTED job can be started"
            )
        self.store.update_status(job_id, advance(current, JobStatus.QUEUED.value))
        self._publish(job_id, {"stage": JobStatus.QUEUED.value, "event": "queued", "job_id": job_id})
        thread = threading.Thread(target=self._run_job, args=(job_id,), daemon=True)
        thread.start()
        return self.store.get(job_id)

    def delete(self, job_id: str) -> None:
        """Removes a job's record and its runs/<job_id> workspace (config,
        events, logs, checkpoints, artifacts). Only allowed for a job that
        either never started (SUBMITTED) or has reached a terminal state --
        deleting the workspace out from under a live subprocess (QUEUED /
        RUNNING / CHECKPOINTING / EVALUATING) could corrupt its writes, so
        that's rejected rather than attempted.
        """
        record = self.store.get(job_id)
        if record is None:
            raise JobNotFoundError(job_id)
        status = JobStatus(record["status"])
        if status not in TERMINAL_STATES and status != JobStatus.SUBMITTED:
            raise JobDeleteConflictError(
                f"job {job_id!r} is {status.value}; wait for it to finish before deleting"
            )
        self.store.delete(job_id)
        shutil.rmtree(workspace.job_dir(job_id), ignore_errors=True)

    # ------------------------------------------------------------------
    # worker subprocess lifecycle
    # ------------------------------------------------------------------

    def _build_flags(self, job_id: str, config: JobConfig) -> List[str]:
        needs_ddl = "ddl" in (config.storage.backend, config.model.source, config.dataset.source)
        ddl_flags: List[str] = []
        if needs_ddl:
            settings = self.settings_store.get()
            if not settings["ddl_server"]:
                raise ValueError(
                    f"job {job_id!r} requests DDL storage but no ddl_server is configured -- set one via "
                    "PUT /settings first"
                )
            ddl_flags = [
                "--ddl-server", settings["ddl_server"],
                "--ddl-port", str(settings["ddl_port"]),
                "--ddl-tenant", str(settings["ddl_tenant"]),
                "--ddl-cpu-base", str(settings["ddl_cpu_base"]),
            ]

        # A resumed job reads (and keeps adding to) its source job's checkpoint run:
        # the checkpoint run id is the id of the job that wrote it, and a local run's
        # checkpoints live in that job's directory. Events/artifacts stay under this
        # job's own id and directory.
        resume = config.checkpoint.resume_from
        run_id = resume.job_id if resume else job_id
        resume_flags = ["--resume", resume.checkpoint_id] if resume else []

        if config.storage.backend == "ddl":
            checkpoint_flags = ["--checkpoint-storage", "ddl"]
        else:
            checkpoint_flags = [
                "--checkpoint-storage", "local",
                "--checkpoint-local-dir", str(workspace.checkpoints_dir(run_id)),
            ]

        return [
            "--model-path", config.model.name,
            "--model-source", config.model.source,
            "--dataset", config.dataset.name,
            "--dataset-source", config.dataset.source,
            "--output-dir", str(workspace.job_dir(job_id)),
            "--run-id", run_id,
            "--job-id", job_id,
            "--events-file", str(workspace.events_path(job_id)),
            "--epochs", str(config.training.epochs),
            "--batch-size", str(config.training.batch_size),
            "--gradient-accumulation-steps", str(config.training.gradient_accumulation_steps),
            "--learning-rate", str(config.training.learning_rate),
            "--max-seq-length", str(config.training.max_seq_length),
            "--lora-r", str(config.training.lora_r),
            "--lora-alpha", str(config.training.lora_alpha),
            "--lora-dropout", str(config.training.lora_dropout),
            "--checkpoint-every-epoch",
            *checkpoint_flags,
            *resume_flags,
            *ddl_flags,
            "--eval-split-ratio", str(config.evaluation.split_ratio if config.evaluation.enabled else 0),
            "--evaluate" if config.evaluation.enabled else "--no-evaluate",
            "--dataloader-workers", "0",
            "--monitor-steps", "0",
        ]

    def _run_job(self, job_id: str) -> None:
        record = self.store.get(job_id)
        config = JobConfig.model_validate_json(workspace.config_path(job_id).read_text())
        flags = self._build_flags(job_id, config)
        logger.info("Launching job %s via %s: %s", job_id, type(self.training_client).__name__, flags)

        log_file = open(workspace.log_path(job_id), "w")
        try:
            proc = self.training_client.start(
                job_id, flags, log_file,
                python_executable=self.python_executable, entrypoint_module=self.entrypoint_module,
            )
        except (OSError, urllib.error.URLError) as exc:
            log_file.close()
            self.store.update_status(
                job_id, JobStatus.FAILED, completed_at=time.time(), error=f"failed to launch worker: {exc}"
            )
            self._publish(job_id, {"stage": JobStatus.FAILED.value, "event": "job_failed", "error": str(exc)})
            return

        self._processes[job_id] = proc
        try:
            self._tail_events(job_id, proc)
        finally:
            log_file.close()
            self._processes.pop(job_id, None)

    def _drain_new_events(self, job_id: str, events_file: Path, pos: int) -> int:
        if not events_file.exists():
            return pos
        with open(events_file, "r") as fh:
            fh.seek(pos)
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._handle_event(job_id, event)
            pos = fh.tell()
        return pos

    def _tail_events(self, job_id: str, proc: "subprocess.Popen") -> None:
        events_file = workspace.events_path(job_id)
        pos = 0
        while proc.poll() is None:
            pos = self._drain_new_events(job_id, events_file, pos)
            time.sleep(_POLL_INTERVAL_S)
        # one final drain: the worker may have written its terminal event
        # in the window between the last poll and process exit
        pos = self._drain_new_events(job_id, events_file, pos)

        record = self.store.get(job_id)
        if record is None:
            # The job's record (and workspace) was deleted out from under
            # this thread while the subprocess was finishing up -- nothing
            # left to reconcile or publish to.
            return
        current = JobStatus(record["status"])
        if current not in TERMINAL_STATES:
            # The subprocess exited without emitting job_complete/job_failed
            # (a crash outside the try/except in _run_training, an OOM kill,
            # a signal) -- surfaced as FAILED rather than left stuck.
            #
            # This is also appended to events.jsonl itself, not just
            # published to live subscribers: the SSE endpoint's backlog
            # replay (api/main.py's _event_stream) reads only events.jsonl,
            # so a client that connects *after* this happens -- including a
            # page refresh -- would otherwise replay a backlog that never
            # reaches a terminal event and hang forever waiting for one that
            # already happened. Writing it here keeps events.jsonl the
            # single source of truth the backlog replay depends on.
            tail = self._log_tail(job_id)
            reason = getattr(proc, "failure_reason", None)
            error = f"worker exited with code {proc.returncode} without a terminal event.\n"
            if reason:
                error += f"{reason}\n"
            error += tail
            completed_at = time.time()
            with JsonlEventEmitter(events_file, job_id=job_id) as synth_emitter:
                synth_emitter.emit(JobStatus.FAILED.value, "job_failed", error=error)
            self.store.update_status(job_id, JobStatus.FAILED, completed_at=completed_at, error=error)
            self._publish(
                job_id,
                {"ts": completed_at, "job_id": job_id, "stage": JobStatus.FAILED.value,
                 "event": "job_failed", "error": error},
            )

        self._publish(job_id, {"event": "stream_closed"})

    def _log_tail(self, job_id: str, n_lines: int = 40) -> str:
        path = workspace.log_path(job_id)
        if not path.exists():
            return ""
        lines = path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n_lines:])

    def _handle_event(self, job_id: str, event: Dict[str, Any]) -> None:
        record = self.store.get(job_id)
        if record is None:
            return
        stage = event.get("stage")
        if stage:
            current = JobStatus(record["status"])
            try:
                new_status = advance(current, stage)
            except IllegalTransitionError:
                logger.warning(
                    "job %s: dropping illegal transition %s -> %s (event=%s)",
                    job_id, current.value, stage, event.get("event"),
                )
            else:
                if new_status != current:
                    extra: Dict[str, Any] = {}
                    if current == JobStatus.QUEUED and new_status == JobStatus.RUNNING:
                        extra["started_at"] = event.get("ts", time.time())
                    if new_status in TERMINAL_STATES:
                        extra["completed_at"] = event.get("ts", time.time())
                    if new_status == JobStatus.FAILED:
                        extra["error"] = event.get("error")
                    self.store.update_status(job_id, new_status, **extra)
        self.store.set_last_event(job_id, event)
        self._publish(job_id, event)

    # ------------------------------------------------------------------
    # checkpoints (sourced from checkpoint_saved events -- see
    # training/utils/events.py -- not by re-deriving from storage, so this
    # module never needs to import a storage backend at all)
    # ------------------------------------------------------------------

    def get_checkpoints(self, job_id: str) -> List[Dict[str, Any]]:
        events_file = workspace.events_path(job_id)
        if not events_file.exists():
            return []
        checkpoints = []
        with open(events_file, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event") == "checkpoint_saved":
                    checkpoints.append(event)
        return checkpoints

    # ------------------------------------------------------------------
    # final model artifact (Phase 2.1)
    # ------------------------------------------------------------------

    def get_final_adapter_dir(self, job_id: str) -> Path:
        """Returns the on-disk directory of the final LoRA adapter for a
        completed job, or raises. Callers (the API layer) never see or
        expose this path directly to the client -- it's read server-side
        to build a zip response.
        """
        record = self.store.get(job_id)
        if record is None:
            raise JobNotFoundError(job_id)
        if record["status"] != JobStatus.COMPLETED.value:
            raise JobNotCompletedError(f"job {job_id!r} is {record['status']}, not COMPLETED")
        adapter_dir = workspace.final_adapter_dir(job_id)
        if not adapter_dir.is_dir() or not any(adapter_dir.iterdir()):
            raise ArtifactNotFoundError(f"no final model artifact found for job {job_id!r}")
        return adapter_dir

    # ------------------------------------------------------------------
    # pub/sub for SSE
    # ------------------------------------------------------------------

    def subscribe(self, job_id: str) -> "queue.Queue":
        """Registers a new subscriber queue for `job_id`. Backlog delivery
        (replaying events.jsonl's current contents) is the caller's job --
        see api/main.py's event_stream(), which reads the backlog and
        subscribes inside the same request so no event is dropped or
        double-delivered.
        """
        q: "queue.Queue" = queue.Queue()
        with self._pub_lock:
            self._subscribers.setdefault(job_id, []).append(q)
        return q

    def unsubscribe(self, job_id: str, q: "queue.Queue") -> None:
        with self._pub_lock:
            subs = self._subscribers.get(job_id)
            if subs and q in subs:
                subs.remove(q)

    def _publish(self, job_id: str, event: Dict[str, Any]) -> None:
        with self._pub_lock:
            subs = list(self._subscribers.get(job_id, ()))
        for q in subs:
            q.put(event)

    def event_backlog(self, job_id: str) -> List[Dict[str, Any]]:
        from nebula_events import read_events

        return read_events(workspace.events_path(job_id))
