# Implementation Plan: Foundation Refactor (Changes 1–20)

Source of truth for *what* and *why*: `architecture-review.md`. Intent: `docs/intent/foundation-refactor.md`.
This plan is only about *order, slicing and verification*. No code has been written.

## Overview

Turn the LoRA PoC into a base the roadmap features can be built on, without a rewrite. Every step is additive or behavior-preserving, lands behind a facade or flag, and leaves `pytest`, `npm run lint` and `npm run build` green. When a step looks like overkill during implementation, move it to `tasks/techdebt.md` (create on first use) and continue. Steps that are good candidates are marked **[defer-able]** with the condition that makes them safe to drop.

Scope: Changes 1–20 plus the **Postgres slice of Change 24** (metadata DB only: T13b–T13d). Change 16 is folded into 19. Change 17 goes last. The rest of Change 24 (event-table ingestion, node-local model cache, benchmarks), Changes 21, 22, 23, 25 and all new features are out of scope.

## Dependency graph

```
Phase 0  Safety net + ADRs
   T1 CI ─ T2 service tests ─ T3 xfail tests ─ T4 contract drift test ─ T5/T6 ADRs
        │
Phase 1  Hardening (Changes 1,2,3)           needs T1–T3
   T7 ─ T8 (recover)        T9 ─ T10        T11
        │
Phase 2  Behavior-preserving restructure     needs Phase 1
   T12 ─ T13 ─ T13b ─ T13c ─ T13d (Change 9 + Postgres)      T14 ─ T15 ─ T16 (Change 5)      T17 (10)   T18 ─ T19 (11)
        │
Phase 3  Contract, Task model, execution     needs T5/T6, Phase 2
   T20 ─ T21 ─ T22 (Change 4) ─ T23 ─ T24 (Change 18) ─ T25 ─ T26 ─ T27 ─ T28 (Change 19, 16)
        │
Phase 4  Registry + resource references      needs T13, T24
   T29 ─ T30 ─ T31 (Change 20)  ─ T32 (Change 6)
        │
Phase 5  Remaining items
   T33 (7)  T34 ─ T35 (8)  T36 (14)  T37 ─ T38 ─ T39 (13)  T40 (15)  T41 (17)
```

Phase 0 and Phase 1 can overlap after T1–T3. Within Phase 2, Change 5 (T14–T16) and Change 9 (T12–T13) are independent and can go in parallel. T10 and T11 touch `pipeline/service/app.py` and must be sequential. Everything in Phase 3 is sequential: it all edits the supervisor and the contract.

## Architecture decisions (taken in this plan, confirm at Checkpoint 0)

- **Facade, not rewrite.** `/jobs`, `JobConfig`, the SPA and existing tests keep working until the last step that replaces them. Old flag path stays one release after the spec path exists (T21–T22).
- **Events file stays the worker's output channel** (review Part D). Only the readers change.
- **Platform states become coarse; phases become events** (Change 18). The adapter between old `JobStatus` and new states lives in one module so the UI migrates later.
- **Postgres is the production metadata DB; SQLite stays for dev and fast tests.** `Database` (T12) is the single seam. The same suite runs against both (T13c), so every table added later (tasks T23, assets T29) is dialect-neutral from day one. Postgres is introduced right after the `Database` module so nothing after it is written twice.
- **Single backend replica for now.** Postgres removes the SQLite-on-shared-volume constraint but multi-replica needs a single-tailer-per-job rule (advisory lock); that is recorded as techdebt, not built here.
- **No external broker, no workflow engine, no serving engine** (review Part D).
- **Tenancy is out of scope** (Change 23), including nullable `tenant_id`/`owner` columns on the new tables. Decided; revisit after this plan.

## Phase 0 — Safety net and decisions

### T1: CI skeleton (Change 12e)
**Description:** Add one GitHub Actions workflow. No CI exists today (`.github/` absent).
**Acceptance:** workflow runs `pytest` (no GPU), `cd web && npm ci && npm run lint && npm run build`, and `docker compose -f docker-compose.yml -f docker-compose.cpu.yml config`. Fails on any step failing.
**Verification:** run each command locally; push branch and see the workflow green.
**Dependencies:** none. **Files:** `.github/workflows/ci.yml`. **Size:** XS.

### T2: Training-service tests (Change 12a)
**Description:** Pin down `pipeline/service/app.py` behavior with FastAPI `TestClient` and a fake `Popen`. It has no tests today.
**Acceptance:**
- Tests cover start → 202, status running → `exit_code: null`, status finished → exit code, unknown id → 404, `OSError` → 500.
- A test asserts that starting a still-running `job_id` is rejected with 409. Mark it `xfail(reason="Change 2")`.
**Verification:** `pytest pipeline/tests/test_service_app.py -v` (xfail shown as xfailed).
**Dependencies:** none. **Files:** `pipeline/tests/test_service_app.py`. **Size:** S.

### T3: Backend failure-mode tests, xfail until fixed (Change 12c/d)
**Description:** Write the tests that Changes 1 and 2 must turn green, using the stub trainer.
**Acceptance:** one test each, marked `xfail(reason="Change N")`: (1) unknown `stage` in an event does not kill the tailer; (2) bad config/flags (e.g. DDL with no server) ends in persisted `FAILED`, not stuck `QUEUED`; (3) a half-written trailing line is not consumed and not lost; (4) double `start()` launches one worker; (5) backend restart mid-job recovers or fails the job.
**Verification:** `pytest backend/tests/test_failure_modes.py -v` shows 5 xfailed, rest of suite green.
**Dependencies:** none. **Files:** `backend/tests/test_failure_modes.py`, maybe `backend/tests/fixtures/stub_trainer.py`. **Size:** M.

### T4: Contract drift test (Change 12b)
**Description:** AST-scan `pipeline/train_lora_with_gpu_stats.py` for every `emit(<stage>, …)` literal and assert each is a `JobStatus` value. Also assert every flag in `_build_flags()` output is accepted by the real `parse_args`.
**Acceptance:** both tests pass on today's code; adding a bad stage or flag fails them.
**Verification:** `pytest backend/tests/test_contract_drift.py -v`; locally break a stage name and confirm failure.
**Dependencies:** none. **Files:** `backend/tests/test_contract_drift.py`. **Size:** S.

### T5: ADR-1 and ADR-4 — Task model and worker contract
**Description:** Short ADRs in `docs/adr/`. ADR-1: Task (run-to-completion) generic model, phases as events, coarse platform states. ADR-4: typed `TrainSpec` in, typed events and output manifests out, versioned.
**Acceptance:** each ADR has context, decision, alternatives, consequences, migration; states what the `/jobs` facade keeps. You have reviewed and approved them.
**Verification:** review only.
**Dependencies:** none. **Files:** `docs/adr/0001-task-model.md`, `docs/adr/0004-worker-contract.md`. **Size:** S.

### T6: ADR-2, ADR-3, ADR-6 — execution backend, registry, system of record
**Description:** ADR-2: `ExecutionBackend` interface, DB-backed queue, K8s Job as the production backend, no broker. ADR-3: immutable asset versions, digest-addressed, lineage. ADR-6: metadata DB vs artifact store vs `runs/` scratch, including the Postgres decision (driver, SQLite for dev/test, in-cluster vs managed, how the two dialects are kept in sync).
**Acceptance:** as T5. ADR-5 (tenancy) and ADR-8 (dataset model) are explicitly recorded as *deferred, out of scope*.
**Verification:** review only.
**Dependencies:** T5. **Files:** `docs/adr/0002-…`, `0003-…`, `0006-…`. **Size:** S.

### Checkpoint 0
- [ ] CI green; T3 xfails all reference a Change number
- [ ] ADR-1, 2, 3, 4, 6 approved by you
- [ ] Open questions: none remain (see "Open questions")

## Phase 1 — Hardening (Changes 1, 2, 3)

### T7: Supervisor never dies silently (Change 1, steps 1, 2, 5)
**Description:** Wrap `_run_job` in a top-level handler that maps any exception, including `_build_flags` `ValueError`, to a persisted `FAILED` (written to `events.jsonl` too). Catch `ValueError` for unknown stages in `_handle_event`. `_drain_new_events` only consumes lines ending in `\n` and does not advance past a partial line.
**Acceptance:** T3 tests (1)–(3) flip from xfail to pass; `FAILED` events appear in `events.jsonl` and via SSE backlog.
**Verification:** `pytest backend/tests/test_failure_modes.py backend/tests/test_job_manager.py -v`.
**Dependencies:** T3. **Files:** `backend/job_manager.py`, `backend/tests/test_failure_modes.py`. **Size:** S.

### T8: Restart recovery (Change 1, steps 3, 4)
**Description:** Add `jobs.events_offset` (via `ensure_column`) and persist the tail offset. Add `JobManager.recover()`, called at app startup, behind a feature flag (`NEBULA_RECOVER_JOBS`, default on in http mode). HTTP mode: rebuild a `_RemoteTrainingHandle` from `job_id` and resume tailing from the saved offset. Subprocess mode: mark active jobs `FAILED("backend restarted")`.
**Acceptance:** T3 test (5) passes in both modes; offset survives restart; no event is applied twice.
**Verification:** `pytest backend/tests/test_failure_modes.py -k restart -v`; manual: start stub job, kill and restart backend, job reaches terminal state.
**Dependencies:** T7. **Files:** `backend/job_store.py`, `backend/job_manager.py`, `backend/api/main.py`, tests. **Size:** M.
**Risk:** the offset write per drain adds a DB write per event batch; measure.

### T9: Atomic transitions (Change 2, store part)
**Description:** `JobStore.transition(job_id, expected, new, **fields) -> bool` using `UPDATE … WHERE job_id=? AND status=?`. `start()` and `_handle_event` use it. Loser gets `IllegalTransitionError`.
**Acceptance:** two threads calling `start()` on one job: exactly one wins; T3 test (4) passes.
**Verification:** `pytest backend/tests/test_job_store.py backend/tests/test_failure_modes.py -v`.
**Dependencies:** T7. **Files:** `backend/job_store.py`, `backend/job_manager.py`, tests. **Size:** S.

### T10: Single-flight and capacity in the training service (Change 2, service part)
**Description:** `_ProcessRegistry.start()` returns 409 for a `job_id` still alive. Optional `NEBULA_MAX_CONCURRENT` (default unlimited) returns 429 when full. `JobManager` treats 429 as "stay QUEUED and retry" with backoff.
**Acceptance:** T2 xfail flips to pass; 429 path covered with a fake client; default behavior unchanged.
**Verification:** `pytest pipeline/tests/test_service_app.py backend/tests/test_job_manager.py -v`.
**Dependencies:** T2, T9. **Files:** `pipeline/service/app.py`, `backend/job_manager.py`, tests. **Size:** M.

### T11: Trust boundary (Change 3)
**Description:** Optional bearer token (`NEBULA_SERVICE_TOKEN`) required by the service and sent by `HttpTrainingClient`. Interim flag allow-list and path-under-root checks in the service. Default `NEBULA_CORS_ORIGINS` to empty when `NEBULA_TRAINING_BACKEND=http`. Token also pluggable on the backend API. K8s manifest reads the token from a Secret.
**Acceptance:** with the token set, requests without it get 401/403; with it unset, dev behavior is unchanged; a flag outside the allow-list or a path outside the roots is rejected; an allow-list drift test fails when `_build_flags` changes without the list.
**Verification:** `pytest pipeline/tests/test_service_app.py backend/tests/test_contract_drift.py -v`; `docker compose … up` still runs a job.
**Dependencies:** T4, T10. **Files:** `pipeline/service/app.py`, `backend/job_manager.py`, `backend/api/main.py`, `deploy/k8s/nebula-pipeline.yaml`, tests. **Size:** M.
**[defer-able]** the backend-API token part, if the deployment stays on a trusted network. The service token is the part that matters.

### Checkpoint 1: Hardening done
- [ ] `pytest` green, no xfails left except ones tagged for Changes ≥ 4
- [ ] Kill-and-restart demo with the stub trainer works
- [ ] Review with you before Phase 2

## Phase 2 — Behavior-preserving restructure (Changes 9, 5, 10, 11)

### T12: `Database` module and `JobStore` on it (Change 9a)
**Description:** `backend/db.py`: one connection factory, lock, `schema_version` table with ordered migrations replacing `ensure_column`. Move `JobStore` onto it first. Same file, same schema.
**Acceptance:** `Database(":memory:")` works as the test seam; existing job-store tests unchanged and green; migrations are idempotent on an existing `nebula.db`.
**Verification:** `pytest backend/tests/test_job_store.py backend/tests/test_job_manager.py -v`; open a copy of a real `nebula.db` and confirm startup migrates cleanly.
**Dependencies:** Checkpoint 1. **Files:** `backend/db.py`, `backend/db_migrate.py`, `backend/job_store.py`, test. **Size:** M.

### T13: Remaining stores on `Database`, WAL, DB path, stale-row sweep (Change 9b)
**Description:** Move `SettingsStore`, `DatasetStore`, `ModelStore` onto `Database`. Enable `journal_mode=WAL` and `busy_timeout`. Add `NEBULA_DB_PATH` (default unchanged). At startup, mark `downloading`/`pending` models `error("interrupted by restart")` (same hook as `recover()`).
**Acceptance:** one connection factory in the codebase; stale-row sweep covered by a test; WAL pragma verified.
**Verification:** `pytest backend/tests -k "settings or datasets or models" -v`.
**Dependencies:** T12. **Files:** `backend/settings_store.py`, `backend/datasets.py`, `backend/models.py`, `backend/db.py`, tests. **Size:** M.
**[defer-able]** `NEBULA_DB_PATH` split, if you stay single-node hostPath.

### T13b: Dialect-neutral `Database` with a Postgres driver (Change 9 + 24, Postgres slice)
**Description:** `Database` takes a URL (`NEBULA_DATABASE_URL`; unset = today's SQLite file). Add a Postgres implementation (psycopg 3 plus a small pool) as a hard dependency of the backend package and image (decided), but import it lazily inside the Postgres path so SQLite-only runs never load it. Audit and fix SQL that is SQLite-only: `?` placeholders, `PRAGMA table_info`/`ensure_column`, `INSERT OR REPLACE`, `REAL` timestamps, `check_same_thread`, `lastrowid`. Migrations from T12 run on both dialects.
**Acceptance:**
- No store contains dialect-specific SQL outside `db.py`.
- Migrations apply cleanly to an empty Postgres and to an existing `nebula.db`.
- `T9`'s `UPDATE … WHERE status=?` guard keeps single-winner semantics on Postgres (test with two connections).
**Verification:** `pytest backend/tests/test_db.py -v` against both dialects; manual `psql` check of the schema.
**Dependencies:** T13. **Files:** `backend/db.py`, `backend/job_store.py`, `backend/settings_store.py`, `backend/datasets.py`, `backend/models.py` (SQL only), `backend/pyproject.toml` (psycopg as a normal dependency), `backend/Dockerfile` if it pins deps. **Size:** M.

### T13c: Run the whole backend suite against Postgres in CI
**Description:** A fixture parametrizes the DB URL (SQLite in-memory by default; Postgres when `NEBULA_TEST_DATABASE_URL` is set, with a schema per test or per session). CI adds a Postgres service container and runs the backend tests twice.
**Acceptance:** the full backend suite passes on both; a deliberately SQLite-only query makes the Postgres run fail; local `pytest` still needs no Postgres.
**Verification:** CI shows both jobs green; `NEBULA_TEST_DATABASE_URL=… pytest backend/tests -v` locally with a throwaway container.
**Dependencies:** T13b, T1. **Files:** `backend/tests/conftest.py`, `.github/workflows/ci.yml`. **Size:** S.

### T13d: Deploy Postgres and migrate existing data
**Description:** Compose: a Postgres service in the `docker-compose.yml` profile used for the full topology. K8s: a Postgres StatefulSet (decided) with a volumeClaimTemplate PVC (default StorageClass, 5Gi, both overridable), headless Service, liveness/readiness (`pg_isready`), resource requests, credentials in a Secret, a backup note (`pg_dump` CronJob is techdebt), and the backend `Deployment` waits for it, backend reads `NEBULA_DATABASE_URL`, readiness (T19's `/readyz`) checks the DB. A one-shot `python -m backend.tools.sqlite_to_postgres` copies an existing `nebula.db` (jobs, settings, datasets, models) and is idempotent. The "SQLite lives on the shared hostPath" constraint and comment in the manifest are removed.
**Acceptance:** compose full topology runs a job end to end on Postgres; the migration tool on a copy of a real `nebula.db` yields identical row counts and the SPA shows the same jobs; backend pod restart keeps data; SQLite path still works when no URL is set.
**Verification:** `docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build` and run a stub job; `kubectl apply --dry-run=server`; compare counts with the tool's `--verify` flag.
**Dependencies:** T13b, T19 (readiness; if T19 is not done yet, add the DB check there later). **Files:** `docker-compose.yml`, `deploy/k8s/*.yaml`, `backend/tools/sqlite_to_postgres.py`, `README.md` (deploy env table), tests. **Size:** M.
**Not deferred:** the StatefulSet is part of this task. Single Postgres replica; HA/replication is out of scope.

### T14: Extract `resume_policy.py` and `event_bus.py` (Change 5a)
**Description:** Move `_validate_resume` (pure function taking config, store, checkpoint reader) and subscribe/unsubscribe/publish into their own modules. Move code verbatim and re-export names from `job_manager.py`.
**Acceptance:** `test_resume.py` and `test_job_manager.py` pass unmodified; `resume_policy` is unit-testable without a `JobManager`.
**Verification:** `pytest backend/tests/test_resume.py backend/tests/test_job_manager.py -v`.
**Dependencies:** Checkpoint 1. **Files:** `backend/resume_policy.py`, `backend/event_bus.py`, `backend/job_manager.py`, test. **Size:** S.

### T15: Extract `events_reader.py`; API stops reaching into the store (Change 5b)
**Description:** One reader module for read, offset-tail and `checkpoints_from_events`, used by the manager, the API and `activity.py`. Add `JobManager.get_job()`/`list_jobs()`; `manager.store` becomes private.
**Acceptance:** no `manager.store` use outside `job_manager.py`; `read_events` tolerates a malformed line.
**Verification:** `pytest backend/tests -v`; `grep -rn "manager.store" backend/api backend/activity.py` is empty.
**Dependencies:** T14. **Files:** `backend/events_reader.py`, `backend/job_manager.py`, `backend/api/main.py`, `backend/activity.py`, tests. **Size:** M.

### T16: Extract `launch.py` (Change 5c)
**Description:** Move the `TrainingClient` family, `default_training_client()` and `_build_flags` into `backend/launch.py`. `JobManager` keeps orchestration and the state projection.
**Acceptance:** `job_manager.py` under ~300 lines; old import paths still work via re-exports; all tests green.
**Verification:** `pytest backend/tests pipeline/tests -v`.
**Dependencies:** T15. **Files:** `backend/launch.py`, `backend/job_manager.py`, tests. **Size:** M.

### T17: Stop defaulting to the simulator; delete the dead trainer (Change 10)
**Description:** Make `--checkpoint-storage` required; rename the simulator choice `blobstore-sim` (tests and benchmark only); remove the simulator default in `checkpoint/manager.py`. Delete `pipeline/train_lora.py` after grep confirms no callers; fix docstring examples.
**Acceptance:** running the worker without `--checkpoint-storage` errors; benchmark script and tests updated and green.
**Verification:** `pytest pipeline/tests -v`; `grep -rn "train_lora\b" .` shows no live references.
**Dependencies:** none (can run in Phase 1). **Files:** `pipeline/train_lora_with_gpu_stats.py`, `pipeline/checkpoint/manager.py`, `pipeline/train_lora.py`, benchmark script, tests. **Size:** S–M.

### T18: Training logs reach the backend (Change 11a)
**Description:** The service redirects trainer stdout/stderr to `<runs>/<job>/train.log`, with the path derived and validated against `NEBULA_RUNS_ROOT`.
**Acceptance:** in HTTP mode a crashed worker's `FAILED` error includes a real log tail.
**Verification:** `pytest pipeline/tests/test_service_app.py -v`; manual crash with the stub in compose.
**Dependencies:** T10. **Files:** `pipeline/service/app.py`, test. **Size:** S.

### T19: Health and metrics (Change 11b)
**Description:** Backend `/healthz` (SQLite `SELECT 1`, `RUNS_ROOT` writable) and `/readyz`. Minimal Prometheus text-format counters (jobs by state, event lag, SSE clients) in one module. Structured logs carry `job_id`. Update k8s probes.
**Acceptance:** probes fail when the DB or runs root is unwritable; `/metrics` returns text exposition.
**Verification:** `pytest backend/tests/test_api.py -v`; `curl` the endpoints in compose.
**Dependencies:** T13. **Files:** `backend/api/main.py`, `backend/metrics.py`, `deploy/k8s/*.yaml`, tests. **Size:** M.
**[defer-able]** the metrics part; keep health.

### Checkpoint 2: Restructure done
- [ ] All tests and CI green; no behavior change visible in the SPA
- [ ] `job_manager.py` is orchestration only
- [ ] One DB module; WAL on for SQLite; full backend suite green on SQLite **and** Postgres in CI
- [ ] Compose full topology runs on Postgres; migration tool verified on a real `nebula.db`
- [ ] Review with you before Phase 3

## Phase 3 — Contract, Task model, execution (Changes 4, 18, 19, 16)

### T20: Contract package, log-only (Change 4a)
**Description:** In `shared/nebula-events`: `TrainSpec` (pydantic, versioned), event type and phase constants, a small `Event` model for the ~6 events consumers use. Backend validates events in log-only mode. Contract tests run the real argparse and the stub against the models.
**Acceptance:** package still has zero heavy dependencies beyond pydantic; contract tests pass; invalid events log a warning, nothing is dropped.
**Verification:** `pytest shared/nebula-events backend/tests/test_contract_drift.py -v`.
**Dependencies:** T5, T16. **Files:** `shared/nebula-events/nebula_events/{spec,events}.py`, `pyproject.toml`, tests. **Size:** M.
**Note:** adds pydantic to a package CLAUDE.md calls "zero-dependency" (decided). Update that sentence in CLAUDE.md in this task, not later, so the docs never contradict the code.

### T21: Spec path alongside flags (Change 4b)
**Description:** Worker accepts `--spec spec.json`; the service accepts `TrainSpec` JSON and builds argv itself (which supersedes T11's interim allow-list); the backend builds a spec and sends it. Flag path remains.
**Acceptance:** a job runs end to end via spec in subprocess and HTTP modes; flag path still works; defaults for `--gradient-accumulation-steps` come from one place.
**Verification:** `pytest -v`; `RUN_REAL_E2E_TEST=1 pytest backend/tests/test_e2e_real_training.py` on the GPU box (required).
**Dependencies:** T20, T11. **Files:** `pipeline/train_lora_with_gpu_stats.py`, `pipeline/service/app.py`, `backend/launch.py`, tests. **Size:** M.

### T22: Phases from the contract; flags retired (Change 4c)
**Description:** Worker emits phase names defined in the contract; the backend maps phase → status; the stub uses the contract. Enforce event validation. Delete `_build_flags` and the flag path after one release.
**Acceptance:** renaming a `JobStatus` value no longer breaks the worker without a failing test; the stub imports the contract instead of re-implementing it.
**Verification:** `pytest -v`; drift test replaced by contract test.
**Dependencies:** T21. **Files:** worker, `backend/job_manager.py`, `backend/tests/fixtures/stub_trainer.py`, tests. **Size:** M.
**[defer-able]** the final deletion of the flag path (leave it, record as techdebt).

### T23: Task model and handler registry behind the `/jobs` facade (Change 18a)
**Description:** `tasks` representation: `kind`, `spec_version` columns via migration; handler registry with spec model, validator and builder per kind; `TrainLoRA` is the first handler wrapping existing behavior. `/jobs` creates `kind=train.lora` tasks.
**Acceptance:** all `/jobs` API tests pass unmodified on SQLite and Postgres; a second handler can be registered in a test without touching the supervisor.
**Verification:** `pytest backend/tests -v`.
**Dependencies:** T22, T13, T6. **Files:** `backend/tasks/{registry,handlers}.py`, `backend/job_store.py`, `backend/job_manager.py`, tests. **Size:** M.

### T24: Coarse platform states, phases as progress events (Change 18b)
**Description:** Generic `PENDING → QUEUED → RUNNING → SUCCEEDED | FAILED | CANCELLED` in the platform; `CHECKPOINTING`/`EVALUATING` become phases carried on events. One adapter maps old `JobStatus` ↔ new states so the REST responses and SPA do not change.
**Acceptance:** `/jobs` responses byte-compatible with today's (snapshot test); state machine tests cover the new machine; adapter has its own tests.
**Verification:** `pytest backend/tests/test_state_machine.py backend/tests/test_api.py -v`; `cd web && npm run build`.
**Dependencies:** T23. **Files:** `backend/state_machine.py`, `backend/tasks/…`, `backend/job_manager.py`, tests. **Size:** M.
**Risk:** highest-regret step of Phase 3; keep the adapter until the UI reads phases from events.

### Checkpoint 3a: Contract and Task model
- [ ] Facade snapshot tests prove `/jobs` unchanged
- [ ] Second-kind smoke test with a dummy handler passes
- [ ] Review with you before Change 19

### T25: `ExecutionBackend` interface, cancel, durable run status (Change 19a)
**Description:** Interface `submit/status/cancel/logs`. Wrap the existing Subprocess and Http clients unchanged. Add cancel (service `DELETE /jobs/{id}` → SIGTERM then SIGKILL; backend `POST /jobs/{id}/cancel`). The wrapper writes `exit.json` so run status no longer depends on the in-memory `_ProcessRegistry`.
**Acceptance:** cancelling a running stub job ends `CANCELLED`; after a service restart `status` still answers from `exit.json`; the 404-grace hack in `_RemoteTrainingHandle` is no longer the only recovery path.
**Verification:** `pytest backend/tests pipeline/tests -v`; manual cancel in compose.
**Dependencies:** T24, T18. **Files:** `backend/execution/…`, `pipeline/service/app.py`, `backend/api/main.py`, tests. **Size:** M.

### T26: Capacity model and DB-backed dispatcher (Change 19b, folds 16)
**Description:** `QUEUED` becomes real. A dispatcher picks queued tasks in priority/FIFO order, honoring per-pool capacity and assigning GPUs (`CUDA_VISIBLE_DEVICES`). Infra-failure retry distinct from job failure. Default stays "immediate start" until capacity is configured.
**Acceptance:** with capacity 1, two started jobs run one after the other; default config behaves as today; dispatcher survives restart (reads queue from DB). Dispatch claim is a single atomic statement that works on both dialects (`FOR UPDATE SKIP LOCKED` on Postgres, the guarded `UPDATE` on SQLite), so a second dispatcher could not double-claim a task.
**Verification:** `pytest backend/tests/test_dispatcher.py -v`; stub-trainer concurrency test.
**Dependencies:** T25, T9. **Files:** `backend/dispatcher.py`, `backend/job_manager.py`, `backend/config` (capacity setting), tests. **Size:** M.
**[defer-able]** priorities/fair-share beyond FIFO + capacity.

### T27: Kubernetes Job backend, logic and tests (Change 19c)
**Description:** `KubernetesJobBackend` implementing `ExecutionBackend`, tested against a fake Kubernetes client. Pod template from the handler's runtime descriptor; K8s Job status as source of truth. Behind a per-kind feature flag.
**Acceptance:** submit/status/cancel/logs covered with the fake client; no `kubernetes` import unless the backend is selected.
**Verification:** `pytest backend/tests/test_k8s_backend.py -v`.
**Dependencies:** T26. **Files:** `backend/execution/k8s.py`, `backend/pyproject.toml` (optional extra), tests. **Size:** M.

### T28: K8s manifests, RBAC and pod template (Change 19c, deploy part)
**Description:** Backend ServiceAccount/Role to create Jobs, pod template carrying RDMA/hostNetwork specifics (reuse `rdma-device-plugin.yaml` resources), GPU resource requests.
**Acceptance:** a job runs on a real cluster via the K8s backend and writes `events.jsonl` to the shared volume; the platform API has no RDMA-specific fields.
**Verification:** manual on the cluster; `kubectl apply --dry-run=server`.
**Dependencies:** T27. **Files:** `deploy/k8s/*.yaml`, `README.md` deployment section. **Size:** M.

### Checkpoint 3b: Execution plane
- [ ] Capacity-1 test shows serialized runs; cancel works
- [ ] Backend restart mid-queue loses nothing
- [ ] Review with you before Phase 4

## Phase 4 — Registry and resource references (Changes 20, 6)

### T29: Asset tables and model (Change 20a)
**Description:** `assets` (type, name, version, digest, locations, metadata, parents, created_by_task, state) and immutable-version rules, per ADR-3. No behavior change yet. Types: `dataset`, `base_model`, `adapter`, `checkpoint`.
**Acceptance:** create/list/resolve-alias API in a `registry.py` module with unit tests on SQLite and Postgres; versions are immutable (unique `(type, name, version)`, enforced by the DB); lineage query works (recursive CTE, which both support).
**Verification:** `pytest backend/tests/test_registry.py -v`.
**Dependencies:** T13, T6, T24. **Files:** `backend/registry.py`, migration, tests. **Size:** M.

### T30: Backfill and dual-write (Change 20b)
**Description:** Backfill existing datasets, models, completed-job adapters and checkpoints as `v1`. The supervisor registers task outputs at task end (adapter, checkpoints). Dual-write for one release; physical layout unchanged.
**Acceptance:** every completed job has registered adapter and checkpoint assets; running the backfill twice is a no-op.
**Verification:** `pytest backend/tests -v`; compare asset counts to `runs/` on a real data dir.
**Dependencies:** T29. **Files:** `backend/registry.py`, `backend/job_manager.py`, `backend/datasets.py`, `backend/models.py`, tests. **Size:** M.

### T31: Checkpoint reads from the registry (Change 20c, absorbs Change 7's projection)
**Description:** `/jobs/{id}/checkpoints`, activity and resume validation read the registry, not `events.jsonl`. Events file stays the rebuildable source.
**Acceptance:** no per-request events scan on those paths; a test deletes registry rows and rebuilds them from events.
**Verification:** `pytest backend/tests/test_resume.py backend/tests/test_api.py -v`.
**Dependencies:** T30. **Files:** `backend/job_manager.py`, `backend/events_reader.py`, `backend/api/main.py`, `backend/activity.py`, tests. **Size:** M.

### T32: Jobs reference resources server-side (Change 6)
**Description:** Optional `model.resource_id`/`dataset.resource_id` on `JobConfig`; the backend resolves path and source at submit time, stores both, rejects non-ready resources. Block deleting a resource used by a non-terminal job. SPA sends ids and stops computing paths. Legacy `name/source` still accepted.
**Acceptance:** submitting with ids works and the job config shows resolved values; deleting an in-use dataset returns 409; legacy payloads still pass existing tests; `selectResumeJob` no longer reverse-looks-up by path.
**Verification:** `pytest backend/tests -v`; `cd web && npm run lint && npm run build`; manual create-job in the UI.
**Dependencies:** T30. **Files:** `backend/config.py`, `backend/job_manager.py`, `backend/api/datasets.py`, `web/src/pages/CreateJob.jsx`, tests. **Size:** M.

### Checkpoint 4: Registry
- [ ] Browser-ready: cross-job checkpoint query works from the registry
- [ ] SPA create-job flow works with ids
- [ ] Review with you before Phase 5

## Phase 5 — Remaining items (Changes 7, 8, 14, 13, 15, 17)

### T33: Bound event costs, async SSE, load test (Change 7 + 12 rest)
**Description:** Async SSE using `asyncio.Queue` fed via `call_soon_threadsafe`, bounded queues (drop-oldest for `step`), compact snapshot backlog with offset resume, cleanup of `_subscribers` keys. Add the load test (N SSE clients + dashboard polling).
**Acceptance:** with 50 SSE clients, unrelated endpoints keep sub-second latency; backlog sends a snapshot, not every `step`; the UI still terminates streams.
**Verification:** `pytest backend/tests/test_sse_load.py -v`; manual UI check of a live job.
**Dependencies:** T15, T31. **Files:** `backend/event_bus.py`, `backend/api/main.py`, `web/src/…` (only if backlog shape changes), tests. **Size:** M.
**[defer-able]** snapshot backlog; keep the async and bounding parts.

### T34: Resume guard (Change 8a, interim)
**Description:** Reject a resume whose source has a non-terminal dependent, and reject deleting a job referenced as a source.
**Acceptance:** both cases return a clear 409; existing resume tests pass.
**Verification:** `pytest backend/tests/test_resume.py -v`.
**Dependencies:** T14. **Files:** `backend/resume_policy.py`, `backend/job_manager.py`, tests. **Size:** S.

### T35: Separate read-from and write-to checkpoint runs (Change 8b)
**Description:** Add `--resume-run` (read) distinct from `--run-id` (write) via the spec; checkpoint manager loads from another run's namespace on both local and DDL backends.
**Acceptance:** two resumes from one source write to separate runs; deleting the source no longer breaks a running resume; verified on local and DDL.
**Verification:** `pytest pipeline/tests/test_resume_state.py backend/tests/test_resume.py -v`.
**Dependencies:** T22, T34. **Files:** `pipeline/checkpoint/manager.py`, worker, `backend/launch.py`, tests. **Size:** M.
**[defer-able]** if T34 guard is enough for the demo.

### T36: Model download off the API path (Change 14)
**Description:** Add an `ingest` task kind and run `ModelStore` downloads through the same execution backend; the backend keeps metadata only. In-process path stays behind a flag for local dev.
**Acceptance:** a 3 GB-class download no longer runs in the API process; backend memory limit can drop back; failure shows up as a normal task failure.
**Verification:** `pytest backend/tests/test_models.py -v`; manual download in compose.
**Dependencies:** T26, T23. **Files:** `backend/models.py`, `backend/tasks/handlers.py`, a new worker entrypoint, tests. **Size:** M.
**[defer-able]** until a feature needs it.

### T37: Worker split 1/3: CLI, resources, data (Change 13a)
**Description:** Extract `cli.py`, `resources.py`, `data.py` into `pipeline/worker/`; torch imports local. Entrypoint module path unchanged.
**Acceptance:** pure parts importable and testable without torch; no behavior change.
**Verification:** `pytest pipeline/tests -v`; stub run; `RUN_REAL_E2E_TEST=1` on the GPU box (required).
**Dependencies:** T22. **Files:** `pipeline/worker/{cli,resources,data}.py`, `pipeline/train_lora_with_gpu_stats.py`, tests. **Size:** M.

### T38: Worker split 2/3: modeling and checkpointing (Change 13b)
**Description:** Extract `modeling.py` and `checkpointing.py` (the two checkpoint functions plus `build_checkpoint_manager`).
**Acceptance/Verification:** as T37. **Dependencies:** T37, T35. **Files:** `pipeline/worker/{modeling,checkpointing}.py`, entrypoint, tests. **Size:** M.

### T39: Worker split 3/3: loop and thin entrypoint (Change 13c)
**Description:** Extract `loop.py`; entrypoint ≤100 lines.
**Acceptance/Verification:** as T37. **Dependencies:** T38. **Files:** `pipeline/worker/loop.py`, entrypoint, tests. **Size:** M.
**[defer-able]** T37–T39 as a group if nothing needs the worker internals soon.

### T40: One truthful docs/deploy story (Change 15)
**Description:** README = how to run; `architecture.md` = current state; `pipeline/README.md` = checkpoint internals. Retire overlap with the STE doc. Add a "Deployments" section (tested vs intended). Update `CLAUDE.md` to the new module layout. Add a CI stale-path grep (`app/`, `training/`).
**Acceptance:** each doc has one stated purpose; env tables include `NEBULA_DDL_*`, `UCX_*`, the new service token and DB path; install line includes `nebula-ddl-storage`.
**Verification:** CI grep step passes; follow README on a clean checkout.
**Dependencies:** none; finish after Phase 4. **Files:** `README.md`, `CLAUDE.md`, `architecture*.md`, `.github/workflows/ci.yml`. **Size:** S.

### T41: Remove import-time side effects (Change 17)
**Description:** `uvicorn backend.api.main:create_app --factory`; a `Settings` object read once and passed in; `workspace` functions take a root. Keep `app = create_app()` until Dockerfile, compose and k8s switch.
**Acceptance:** importing `backend.api.main` creates no directories or DB; tests no longer monkeypatch `workspace.RUNS_ROOT`.
**Verification:** `pytest -v`; compose and Dockerfile start commands updated and running.
**Dependencies:** all prior. **Files:** `backend/api/main.py`, `backend/workspace.py`, `backend/settings.py`, Dockerfile/compose, tests. **Size:** M.
**[defer-able]** P3.

### Checkpoint 5: Complete
- [ ] Changes 1–20 done or recorded in `tasks/techdebt.md` with a reason
- [ ] `pytest`, lint, build, compose validation green in CI
- [ ] Review against the review's "Minimum architectural work" list; you pick demo features

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| T24 state-machine change breaks UI or stored rows | High | Adapter plus byte-compat snapshot test; keep old states until UI reads phases |
| Worker (1,100 lines) has no tests; T21/T22/T35/T37–39 edit it | High | Do T37–39 only after the contract is stable; run the real e2e on a GPU box for each; stub run for the rest |
| Events offset write per batch slows tailing (T8) | Med | Write offset per drain, not per event; measure with the load test |
| Recovery in subprocess mode cannot re-adopt a process (T8) | Med | Mark `FAILED("backend restarted")`; documented |
| Two dialects drift apart | Med | T13c runs the same suite on both in CI; all SQL goes through `db.py` |
| Postgres changes locking and timing semantics (T8 offset writes, T9 guard, T26 claim) | Med | Each of those tasks has a Postgres-run acceptance test; use `FOR UPDATE SKIP LOCKED`, not app-level locks |
| Existing `nebula.db` data lost or altered in migration (T13d) | High | Idempotent tool with `--verify` row counts; run on a copy first; SQLite path stays available until you cut over |
| Multi-replica backend still unsafe (two tailers per job) | Med | Out of scope; single replica stays; advisory-lock tailer recorded as techdebt |
| Scope creep from the review's size | High | Defer-able markers; `tasks/techdebt.md`; checkpoint reviews |
| No GPU in CI | Med | Stub trainer covers the contract in CI; real e2e runs manually on the GPU box at the checkpoints listed in Decisions |

## Open questions

None open. Decisions taken:

1. **Tenancy:** left entirely for later; no `tenant_id`/`owner` columns anywhere in this plan.
2. **ADRs:** written for you only; no external review step.
3. **K8s Job backend:** T27–T28 stay in the plan and are not defer-able.
4. **Contract package:** pydantic is added to `shared/nebula-events`.
5. **GPU box:** available; `RUN_REAL_E2E_TEST=1` runs at Checkpoints 1, 2, 3a, 3b and 5, plus after T21, T22, T35 and T37–T39.
6. **Postgres hosting:** in-cluster StatefulSet on a PVC.
7. **Driver:** psycopg is a hard dependency of the backend package/image, imported lazily.
8. **Cutover:** SQLite stays supported for dev/tests for a long time; Postgres in prod.
9. **StatefulSet storage:** cluster default StorageClass, 5Gi PVC.
