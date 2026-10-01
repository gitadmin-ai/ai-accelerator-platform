# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A LoRA fine-tuning platform with three independently deployable components plus a
zero-dependency contract package tying them together:

```
 web/            React SPA                  \  non-GPU node(s)
 backend/        FastAPI API + Job Manager   /  (or one box for local dev)
                        |
                        | HTTP (POST /jobs/{id}/start)
                        v
 pipeline/       training-service (FastAPI)  \  GPU node(s)
                        |                     /
                        | subprocess.Popen
                        v
                 pipeline.train_lora_with_gpu_stats
                        |
                        v
          events.jsonl / checkpoints / artifacts
          (shared storage -- see "Node topology" in README.md)
```

- **`web/`** — React + Vite SPA. Talks only to `backend/` over HTTP (`web/src/api.js`, base URL from `VITE_API_BASE`).
- **`backend/`** — FastAPI service: job/dataset/model metadata (SQLite), the Job Manager (job lifecycle, SSE event streaming). CPU-only — never imports torch/transformers/peft, so it never needs a GPU.
- **`pipeline/`** — The training agent: LoRA fine-tuning (torch/transformers/peft) plus a chunked, parallel-write checkpoint system. GPU-only. Ships its own thin `pipeline/service/` HTTP wrapper so `backend/` can trigger a run without local process/filesystem access to the GPU node.
- **`shared/nebula-events/`** — zero-dependency `nebula_events` package both `backend/` and `pipeline/` install: the JSONL event contract they communicate over. `backend/` never imports anything else from `pipeline/`.
- **`shared/nebula-ddl-storage/`** — `nebula_ddl_storage`: shared DDL3 (Nebula) connection wrapper and whole-resource (model/dataset directory) storage, used by both `backend/` and `pipeline/`. The native `ddl_client` extension is a lazy/guarded import, not a hard dependency — the package imports fine without it, it just can't talk to a real DDL target.

Each of the four is its own installable Python package with its own `pyproject.toml`; the root `pyproject.toml` has no `[project]` table (installs nothing) — it only holds shared pytest config.

## Commands

Install everything for local dev (no GPU / ML deps needed unless touching `pipeline/`'s real trainer):

```bash
pip install -e shared/nebula-events -e shared/nebula-ddl-storage -e "backend[dev]" -e "pipeline[dev]"
cd web && npm install
```

Run locally without Docker (two terminals):

```bash
# terminal 1 — backend, default NEBULA_TRAINING_BACKEND=subprocess launches
# training as a local child process, no GPU node or pipeline/service needed
NEBULA_TRAINING_BACKEND=subprocess uvicorn backend.api.main:app --reload
# terminal 2
cd web && npm run dev
```

Run the full three-container topology (mirrors prod's HTTP + shared-storage boundary). Pick a profile -- `docker-compose.cpu.yml` for local/CI (no GPU reservation) or `docker-compose.gpu.yml` for the real GPU node (requires the NVIDIA Container Toolkit on that host):

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build
```

Tests (all Python packages, from repo root — `pyproject.toml`'s `testpaths` covers all four):

```bash
pip install -e shared/nebula-events -e shared/nebula-ddl-storage -e "backend[dev]" -e "pipeline[dev]"
pytest
```

Run a single test file/case:

```bash
pytest backend/tests/test_job_manager.py
pytest backend/tests/test_job_manager.py::test_job_fails_on_silent_crash -v
```

Backend tests default to `NEBULA_TRAINING_BACKEND=subprocess` and a stub trainer entrypoint (`backend/tests/fixtures/stub_trainer.py`), so they run in milliseconds without a GPU or `pipeline/`'s ML dependencies. Real end-to-end behavior against the actual training worker lives in `backend/tests/test_e2e_real_training.py` (separate, slower).

`pipeline/` tests run without torch/transformers/peft/GPU installed — everything exercisable without them (serialization, chunking, manifest format, checksum validation, the parallel writer, save/load/verify/resume) has real tests using numpy tensors standing in for torch tensors. `train_lora.py`/`train_lora_with_gpu_stats.py` themselves are not exercised by the test suite.

Web:

```bash
cd web && npm run dev      # Vite dev server
cd web && npm run build    # production build
cd web && npm run lint     # oxlint
```

There is no Python linter/formatter configured in this repo.

## Architecture

### Job lifecycle (the core state machine)

`backend/state_machine.py` defines the one legal path a job's status can take:

```
SUBMITTED -> QUEUED -> RUNNING -> COMPLETED
```

with `RUNNING <-> CHECKPOINTING` and `RUNNING -> EVALUATING -> RUNNING` as repeating sub-loops (once per checkpoint/eval interval, not a single linear pass), and `FAILED` reachable from any non-terminal state. `COMPLETED` is only reached after the final checkpoint (and final evaluation, if enabled). This module is pure (no I/O) and unit-tested standalone; `backend/job_manager.py` is its only caller, and every status change in the system flows through `advance()`/`transition()` — an event whose `stage` implies an illegal jump is logged and dropped, never silently applied.

### How a job actually runs

1. `POST /jobs` (`backend/api/main.py`) validates a `JobConfig` (`backend/config.py` — model/dataset/training/storage/checkpoint/evaluation settings) and calls `JobManager.submit()`, which writes `runs/<job_id>/config.json` and creates a `SUBMITTED` row in `backend/job_store.py` (SQLite).
2. `POST /jobs/{id}/start` moves it to `QUEUED` and spawns a background thread (`JobManager._run_job`) that builds CLI flags from the `JobConfig` (`_build_flags()`) and hands them to a `TrainingClient`.
3. **`TrainingClient` is the local/remote seam** (`backend/job_manager.py`): `SubprocessTrainingClient` (`NEBULA_TRAINING_BACKEND=subprocess`, the default — local dev/tests) launches the worker as a local child process; `HttpTrainingClient` (`NEBULA_TRAINING_BACKEND=http`, prod) POSTs to `pipeline/service/app.py` running on the GPU node, which does the identical `subprocess.Popen` from over there. Both return a `TrainingHandle` (`poll()` → `None` while running, else an exit code) so the rest of `JobManager` doesn't care which one it's talking to.
4. The worker (`pipeline.train_lora_with_gpu_stats`, or `backend/tests/fixtures/stub_trainer.py` in tests) writes one JSON object per line to `events.jsonl` via `nebula_events.JsonlEventEmitter` — this file is the **only** channel between the training process and the Job Manager, which may be on different nodes.
5. `JobManager._tail_events` polls `events.jsonl` every 0.2s (`_POLL_INTERVAL_S`), applies each event's `stage` field through `state_machine.advance()`, updates the SQLite row, and fans the raw event out to SSE subscribers (`_publish`). If the worker process exits without ever emitting a terminal (`job_complete`/`job_failed`) event — a crash, OOM kill, signal — `_tail_events` synthesizes a `FAILED` event itself, written to `events.jsonl` too (not just published live), so a client that connects after the fact via SSE backlog replay still terminates instead of hanging forever.
6. `GET /jobs/{id}/events` (SSE) subscribes *before* reading the backlog (`api/main.py`'s `_event_stream`), so a live event can at worst be delivered twice, never dropped.

### Shared storage, not shared code

`backend/` and `pipeline/` never import training/ML code from each other. They agree on two things instead:
- **The event contract** (`nebula_events`) — line-delimited JSON, `stage` + `event` + free-form fields.
- **A shared filesystem root** — `NEBULA_DATA_ROOT` (downloaded models/datasets, see `backend/workspace.py`) and `NEBULA_RUNS_ROOT` (per-job config/events/checkpoints/artifacts). Both nodes must mount the same path at the same location; `backend/job_manager.py` reads `events.jsonl` by path regardless of whether the writer was a local child process or a process on another machine. In Docker Compose this is the `nebula-shared` named volume; in K8s it'd be an NFS/EFS-backed RWX PVC (see README.md's node topology sketch).

### Storage backends (`local` vs `ddl`)

`JobConfig.storage.backend` / `model.source` / `dataset.source` are each `Literal["local", "ddl"]`. `"ddl"` routes through Nebula's DDL3 storage via `shared/nebula-ddl-storage` (`nebula_ddl_storage.resource_store.DdlResourceStore`), configured through `backend/settings_store.py` (server/port/tenant/cpu-base — set via `PUT /settings`). `backend/_build_flags()` raises if a job requests `ddl` storage with no `ddl_server` configured. `backend/resource_storage.py`'s `LocalResourceStorage` always stages downloads locally first (huggingface_hub / HF `datasets` need a real directory to write into); `push_staged_to_ddl()` then optionally uploads and deletes the local copy. On the checkpoint side, `pipeline/checkpoint/ddl_backend.py` and `localfs_backend.py` are the two `StorageBackend` implementations `pipeline/checkpoint/manager.py` writes through — swapping backends never touches chunking/manifest/verification logic.

### `pipeline/checkpoint/` — chunked, parallel-write checkpointing

Not part of the day-to-day job-lifecycle loop above, but the piece of the codebase with the most internal architecture. Full deep-dive in `pipeline/README.md` (data flow diagram, checkpoint format, commit protocol, failure handling, benchmark methodology) — read it before touching anything under `pipeline/checkpoint/`. Highlights:
- A checkpoint is a `manifest.json` plus many independently addressable chunk blobs (`checkpoint/format.py`), not one blob — LoRA-adapter tensors, flattened optimizer/scheduler/RNG state (`checkpoint/state_flatten.py`), split into fixed-size zero-copy chunks (`checkpoint/chunker.py`).
- Parallelism is built at the Python layer via **sharding** (`checkpoint/parallel_writer.py`): N independent storage-backend instances, each owned by exactly one worker thread for its lifetime, chunks assigned round-robin. `--checkpoint-workers` controls N.
- **Commit protocol**: write+verify every chunk first; only if everything succeeds does it write the scheduler/optimizer-skeleton JSON, then `manifest.json`, then update `catalog.json` — the catalog update is the actual commit point, since discovery (`list_checkpoints`/`get_latest_checkpoint`) only ever reads the catalog, never scans for manifests.
- `--checkpoint-storage local` (`checkpoint/localfs_backend.py`) needs no native extension and is what local dev/tests use. The `ddl`/`eclib::BlobStore`-backed path needs a native extension built separately and put on `PYTHONPATH` — see `pipeline/README.md` §"Building the native extension" and §12 "Known limitations" (notably: that backend is currently an in-process memory simulation, not real persistent storage).

### API surface

`backend/api/main.py` (`create_app()`) is the FastAPI app; `backend/api/{datasets,models,settings}.py` are sub-routers included into it. `create_app()` takes optional pre-built stores/manager so tests inject one pointed at a temp workspace / in-memory SQLite (`SettingsStore(":memory:")`) instead of touching real state. Response shapes are defined in `backend/api/schemas.py`.

### Environment variables

| Variable | Used by | Purpose |
|---|---|---|
| `NEBULA_DATA_ROOT` | backend, pipeline | Root for downloaded models/datasets. Same shared mount on both nodes in prod. |
| `NEBULA_RUNS_ROOT` | backend, pipeline | Root for per-job workspaces (config, events, checkpoints, artifacts). Same shared mount. |
| `NEBULA_TRAINING_BACKEND` | backend | `subprocess` (default, local dev/tests) or `http` (prod, talks to `TRAINING_SERVICE_URL`). |
| `TRAINING_SERVICE_URL` | backend | Base URL of `pipeline/service` on the GPU node. Required when `NEBULA_TRAINING_BACKEND=http`. |
| `NEBULA_CORS_ORIGINS` | backend | Comma-separated allowed origins for the frontend. Defaults to `*` (dev only). |
| `VITE_API_BASE` | web | Base URL of the backend API. |
