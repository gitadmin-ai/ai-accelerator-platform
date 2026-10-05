# Task list: Foundation Refactor (Changes 1–20)

Details, acceptance criteria and verification for each task are in `tasks/plan.md`.
`[d]` = defer-able to `tasks/techdebt.md` if it turns out to be overkill.

## Phase 0 — Safety net and decisions
- [ ] T1 CI skeleton (12e)
- [ ] T2 Training-service tests, 409 case xfail (12a)
- [ ] T3 Backend failure-mode tests, xfail (12c/d)
- [ ] T4 Contract drift test (12b)
- [ ] T5 ADR-1, ADR-4
- [ ] T6 ADR-2, ADR-3, ADR-6
- [ ] **Checkpoint 0:** CI green, ADRs approved

## Phase 1 — Hardening
- [ ] T7 Supervisor never dies silently (1)
- [ ] T8 Restart recovery + persisted offset (1)
- [ ] T9 Atomic transitions (2)
- [ ] T10 Single-flight + capacity in service (2)
- [ ] T11 Trust boundary: service token, allow-list, CORS (3) [d: backend-API token part]
- [ ] **Checkpoint 1:** hardening done, kill-and-restart demo works

## Phase 2 — Behavior-preserving restructure
- [ ] T12 `Database` module + JobStore (9a)
- [ ] T13 Other stores on `Database`, WAL, sweep (9b) [d: DB path split]
- [ ] T13b Dialect-neutral `Database` + Postgres driver (9, 24-Postgres)
- [ ] T13c Full backend suite on SQLite and Postgres in CI
- [ ] T13d Postgres StatefulSet (k8s) + compose service + sqlite→postgres migration tool
- [ ] T14 Extract resume_policy + event_bus (5a)
- [ ] T15 Extract events_reader; API stops using manager.store (5b)
- [ ] T16 Extract launch.py (5c)
- [ ] T17 No simulator default; delete dead trainer (10)
- [ ] T18 Trainer logs reach backend (11a)
- [ ] T19 Health + metrics (11b) [d: metrics]
- [ ] **Checkpoint 2:** `job_manager.py` is orchestration only, no visible behavior change; suite green on SQLite and Postgres; compose runs on Postgres

## Phase 3 — Contract, Task model, execution
- [ ] T20 Contract package, log-only validation (4a)
- [ ] T21 Spec path alongside flags (4b)
- [ ] T22 Phases from contract; retire flags (4c) [d: flag deletion]
- [ ] T23 Task model + handler registry behind `/jobs` (18a)
- [ ] T24 Coarse states, phases as events, adapter (18b)
- [ ] **Checkpoint 3a:** `/jobs` unchanged (snapshot), dummy second kind works
- [ ] T25 ExecutionBackend, cancel, durable run status (19a)
- [ ] T26 Capacity model + dispatcher (19b, 16) [d: priorities/fair-share]
- [ ] T27 Kubernetes Job backend + fake-client tests (19c)
- [ ] T28 K8s manifests, RBAC, pod template (19c)
- [ ] **Checkpoint 3b:** serialized runs at capacity 1, cancel works

## Phase 4 — Registry and resource references
- [ ] T29 Asset tables and model (20a)
- [ ] T30 Backfill + dual-write (20b)
- [ ] T31 Checkpoint reads from registry (20c)
- [ ] T32 Jobs reference resources server-side (6)
- [ ] **Checkpoint 4:** cross-job checkpoint query works, SPA uses ids

## Phase 5 — Remaining
- [ ] T33 Bounded events, async SSE, load test (7, 12) [d: snapshot backlog]
- [ ] T34 Resume guard (8a)
- [ ] T35 Separate read/write checkpoint runs (8b) [d]
- [ ] T36 Model download as a task (14) [d]
- [ ] T37 Worker split 1/3 (13a) [d: T37–T39]
- [ ] T38 Worker split 2/3 (13b)
- [ ] T39 Worker split 3/3 (13c)
- [ ] T40 Docs and deploy story (15)
- [ ] T41 No import-time side effects (17) [d]
- [ ] **Checkpoint 5:** Changes 1–20 done or in techdebt; pick demo features
