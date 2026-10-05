# Architecture — simplified

Read time: 5–10 minutes.
For the full version, read `architecture.md`.

## 1. What Is This Project?

This project is a web platform that fine-tunes language models.
The fine-tune method is LoRA.
LoRA trains a small add-on called an *adapter*. It does not change the base model.

A user picks a model and a dataset in a browser.
The platform trains on a GPU computer.
The user watches progress live and downloads the adapter at the end.

## 2. What Problem Does It Solve?

- Training needs a GPU. The web pages do not.
- Training takes a long time. The user needs live progress.
- A long run can crash. The platform must save progress (*checkpoints*) and report failures.
- The team must test fast storage called *Nebula DDL3*. The platform can save checkpoints there.

## 3. The System in One Picture

```
Browser
   |  (HTTP)
   v
web (nginx + React)
   |  /api/
   v
backend (FastAPI)  <----- SQLite database (nebula.db)
   |  |
   |  | reads events.jsonl  <------------------+
   |  v                                         |
   |  shared storage  <-- writes -- trainer ----+
   |  (same path on both sides)        ^
   | HTTP: "start job"                 | Popen (starts a process)
   v                                   |
training-service (FastAPI, GPU computer)
```

Two words you need:

- **Job**: one training request. It has a 12-character id.
- **Event**: one line of JSON that the trainer writes to a file called `events.jsonl`.

## 4. Main Components

### web (`web/`)
- Does: shows pages. Starts jobs. Shows live progress.
- Receives: REST answers and an event stream (SSE) from the backend.
- Sends: REST calls to the backend. It never talks to any other component.

### backend (`backend/`)
- Does: stores data. Runs the *Job Manager*. Streams events to browsers.
- Receives: REST calls. Event lines from `events.jsonl`.
- Sends: "start job" to the training-service. Rows to SQLite. Events to browsers.
- Rule: it never imports `torch`. It needs no GPU.

### Job Manager (`backend/job_manager.py`)
- Does: moves a job through its states. One thread per running job.
- Receives: a job configuration. Events from the trainer.
- Sends: status changes to SQLite. Events to browsers.
- It is the only code that changes a job's status.

### training-service (`pipeline/service/app.py`)
- Does: starts the trainer process. Reports if the process still runs.
- Receives: `POST /jobs/{id}/start` with a list of command-line flags.
- Sends: the process exit code on `GET /jobs/{id}/status`.
- It remembers processes in memory only.

### trainer (`pipeline/train_lora_with_gpu_stats.py`)
- Does: loads model and data. Trains. Evaluates. Saves checkpoints.
- Receives: command-line flags.
- Sends: event lines to `events.jsonl`. Checkpoints to storage. The final adapter to `artifacts/`.

### checkpoint library (`pipeline/checkpoint/`)
- Does: saves training state as many small pieces (*chunks*). Writes them in parallel. Checks them.
- Receives: tensors and optimizer state.
- Sends: chunk files, `manifest.json`, `catalog.json`.

### shared packages (`shared/`)
- `nebula-events`: the event writer. Both sides use it.
- `nebula-ddl-storage`: the code that talks to Nebula DDL3.

## 5. How a Request Moves Through the System

This is how one job runs.

1. The user fills the wizard in the browser.
2. The browser sends `POST /jobs` with the configuration.
3. The backend writes `runs/<id>/config.json` and a SQLite row. Status is `SUBMITTED`.
4. The user clicks Start. The browser sends `POST /jobs/{id}/start`.
5. Status changes to `QUEUED`. The backend starts a thread for this job.
6. The thread turns the configuration into command-line flags.
7. The thread sends the flags to the training-service.
8. The training-service starts the trainer process.
9. The trainer appends events to `events.jsonl`.
10. The thread reads the new events every 0.2 seconds.
11. The thread updates the status. It sends each event to the open browsers.
12. The trainer writes a final event (`job_complete` or `job_failed`) and exits.
13. The thread asks the training-service for the exit code. It closes the stream.

## 6. How Data Moves Through the System

- **Configuration**: browser → backend → `config.json` and SQLite.
- **Flags**: backend → training-service → trainer.
- **Events**: trainer → `events.jsonl` → Job Manager → SQLite and browsers.
- **Checkpoints**: trainer → local folder or DDL3.
- **Models and datasets**: Hugging Face or upload → backend → disk (or DDL3) → trainer.
- **Final adapter**: trainer → `artifacts/final_adapter/` → backend zips it → browser.

## 7. Where Data Is Stored

| Data | Place |
|---|---|
| Jobs, datasets, models, settings | SQLite file `DATA_ROOT/nebula.db` |
| Job files (config, events, log, checkpoints, adapter) | `RUNS_ROOT/<job id>/` |
| Downloaded models and datasets | `DATA_ROOT/models/<id>/` and `DATA_ROOT/datasets/<id>/` |
| Nebula objects (optional) | DDL3 server |
| Running-process list | Memory only. It is lost on restart. |

`DATA_ROOT` and `RUNS_ROOT` must be the same path on the backend and on the trainer.
This is the shared storage.

## 8. Important Interfaces

- **Backend REST API**: `/jobs`, `/jobs/{id}/events` (SSE), `/jobs/{id}/checkpoints`, `/jobs/{id}/artifact`, `/datasets`, `/models`, `/settings`, `/config`, `/activity`.
- **training-service REST API**: `/jobs/{id}/start`, `/jobs/{id}/status`, `/healthz`.
- **Trainer command line**: about 50 flags. This is the real contract between backend and trainer.
- **Event line**: `{ts, job_id, stage, event, ...}`. `stage` is one of `RUNNING`, `CHECKPOINTING`, `EVALUATING`, `COMPLETED`, `FAILED`.
- **`TrainingClient`**: the backend's switch for how to start a trainer. `subprocess` starts it locally. `http` calls the training-service.
- **`StorageShard`**: the checkpoint library's switch for where to save chunks.

## 9. Important Runtime Processes

- **One thread per running job** in the backend. It reads events.
- **One thread per model download** in the backend.
- **One process per job** on the GPU computer (the trainer).
- **Checkpoint writer threads**: N threads. Each thread owns one storage connection.
- **Open browser streams**: each one holds a backend thread until the stream ends.

Job states:

```
SUBMITTED -> QUEUED -> RUNNING -> COMPLETED
                        |  ^
                        v  |  (once per checkpoint or evaluation)
              CHECKPOINTING / EVALUATING
FAILED can happen from any state that is not finished.
```

## 10. Failure Handling

- **Trainer raises an error**: it writes `job_failed`. The job becomes `FAILED`.
- **Trainer is killed** (for example, out of memory): it writes nothing. The backend sees the exit and writes a `job_failed` event itself.
- **training-service restarts**: it forgets the job. After 30 seconds the backend marks the job `FAILED`.
- **Checkpoint write fails**: the checkpoint is not added to `catalog.json`. The old checkpoints stay valid.
- **Backend restarts during a job**: nothing re-attaches. The job stays `RUNNING` forever. **This is a known gap.**
- **Unknown event stage or a bad flag build**: the job thread can stop. The job stays stuck. **This is a known gap.**
- **Logs in `http` mode**: `train.log` stays empty. Read the training pod's container log.

## 11. Testing

- Run all tests: `pytest` from the repo root.
- Backend tests use a fake trainer (`backend/tests/fixtures/stub_trainer.py`). They finish in milliseconds. They need no GPU.
- Checkpoint tests use numpy arrays instead of torch tensors.
- The real end-to-end test is opt-in: `RUN_REAL_E2E_TEST=1`.
- The web has no tests. It has only `npm run lint`.
- The training-service and the trainer script have no automated tests.

## 12. Important Things to Know Before Changing the Code

- Change job status only through `state_machine.advance()` in the Job Manager.
- Do not import `pipeline` from `backend`. Use `nebula_events` for shared things.
- If you add an event, update the UI and the stub trainer by hand. No schema checks them.
- If you add a trainer flag, update `_build_flags()` in `job_manager.py`.
- A resumed job writes into the **source job's** checkpoint folder.
- The browser converts a chosen model or dataset into a path. The API does not check it.
- There is no login. Do not expose ports 8000 or 8001 to an untrusted network.
- `train_lora.py` looks unused. `train_lora_with_gpu_stats.py` is the real trainer.
- The k8s files put both pods on one node with `hostPath`. This differs from the README.
- Environment variables are read when the module is imported. Set them before you start the process.

## 13. Five-Minute Mental Model

- The backend is a **file reader**. It starts a trainer and then reads the trainer's log of events.
- The trainer is a **separate program**. It knows nothing about the backend.
- The only link between them is **one file** on shared storage.
- The status in SQLite is a **copy** of what the events say.
- Checkpoints are **many small files** plus a catalog. The catalog write is the commit.

## If You Remember Only 10 Things

1. Three tiers: `web` → `backend` → `training-service` + trainer.
2. The trainer talks to the backend through one file: `events.jsonl`.
3. The backend never imports torch or `pipeline`.
4. The Job Manager is the only code that changes job status.
5. Job states: `SUBMITTED → QUEUED → RUNNING → COMPLETED`, with `FAILED` possible at any time.
6. The backend and the trainer must see the same `DATA_ROOT` and `RUNS_ROOT` paths.
7. The trainer command line is the real API between backend and trainer.
8. Checkpoints are chunked, written in parallel, checked, and committed by the catalog update.
9. There is no authentication, no cancel, and no recovery after a backend restart.
10. All metadata is in one SQLite file. Run only one backend replica.
