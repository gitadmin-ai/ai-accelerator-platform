# pipeline-demo

A LoRA fine-tuning platform with three independently deployable components:

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
          (shared storage -- see "Node topology" below)
```

- **`web/`** -- React + Vite SPA. Talks to `backend/` only over HTTP (`web/src/api.js`, base URL from `VITE_API_BASE`, default `/api` -- proxied by nginx to `BACKEND_URL`).
- **`backend/`** -- FastAPI service: job/dataset/model metadata (SQLite), the Job Manager (job lifecycle, SSE event streaming). CPU-only -- never imports torch/transformers/peft, so it never needs a GPU.
- **`pipeline/`** -- The training agent: LoRA fine-tuning (torch/transformers/peft) plus a chunked, parallel-write checkpoint system. GPU-only. Ships its own thin `pipeline/service/` HTTP wrapper so `backend/` can trigger a run without needing local process/filesystem access to the GPU node.
- **`shared/nebula-events/`** -- a small, zero-dependency package (`nebula_events`) both `backend/` and `pipeline/` install, defining the JSONL event contract they use to communicate. `backend/` never imports anything else from `pipeline/`.

## Node topology

In production, `web/` + `backend/` run together on a non-GPU node (or split further if you like -- they only talk over HTTP already); `pipeline/`'s `training-service` runs on a GPU node. The two nodes need:

1. **Network reachability**: `backend/` needs `TRAINING_SERVICE_URL` to reach `pipeline/service`'s port.
2. **A shared filesystem mount** (NFS/EFS/Filestore/etc.), mounted at the *same absolute path* on both nodes, holding `data/` (downloaded models/datasets) and `runs/` (per-job checkpoints, logs, `events.jsonl`). Both `backend` and the training worker resolve these through `NEBULA_DATA_ROOT`/`NEBULA_RUNS_ROOT`, so as long as both nodes mount the shared storage at the same path and set the same env vars, they see each other's writes without any other code change. `backend/job_manager.py` tails `events.jsonl` by path -- it doesn't care whether the training worker that wrote it is a local child process or a process on another machine.

For a fleet of GPU workers instead of one fixed node, or nodes that can't share a POSIX mount, swap the shared mount for object storage (S3/MinIO) and the direct HTTP trigger for a job queue (SQS/Redis) -- neither is needed at this project's scale.

### Illustrative Kubernetes sketch

```
Deployment "backend"    nodeSelector: {}                  (any non-GPU node)
Deployment "frontend"   nodeSelector: {}                  (any non-GPU node)
Deployment "training-service"
                         nodeSelector: {nvidia.com/gpu: "true"}
                         resources.limits: {nvidia.com/gpu: 1}
Both backend and training-service mount the same PersistentVolumeClaim
(ReadWriteMany, e.g. backed by EFS/Filestore) at /mnt/nebula-shared.
```

This is one example, not a maintained manifest set -- adapt to your cluster.

## Running locally

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build
```

This starts `frontend` (nginx, :5173), `backend` (:8000), and `training-service` (:8001) as separate containers on one compose network, sharing a named volume (`nebula-shared`) that stands in for the NFS mount in prod -- exercising the same HTTP + shared-storage boundary as production, just on one machine. `training-service` runs on CPU against the stub trainer / small models.

On the actual GPU node, swap the profile to reserve the host's GPU(s) (requires the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) installed on that host):

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

### Iterating on code without rebuilding images

The images install third-party dependencies in their own layer, keyed only on
`pyproject.toml` (plus a BuildKit pip cache), so a rebuild after a code-only edit
re-downloads nothing and takes ~20 s. To skip the rebuild entirely, add the dev
overlay, which runs the code from your working tree inside the containers:

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml -f docker-compose.dev.yml up -d
```

It bind-mounts `backend/`, `pipeline/` and `shared/*` over the baked-in copies and
runs `uvicorn --reload`. Training jobs are separate processes started per job, so
edits to the training and checkpoint code apply to the next job with no restart.
Rebuild (`up -d --build`) only when `pyproject.toml`, a Dockerfile, or the
`nebula-ddl-client` wheel changes. The first build after a Dockerfile change that
touches the torch layer re-downloads torch once (~3 GB).

### Without Docker

```bash
pip install -e shared/nebula-events -e backend -e "pipeline[dev]"
cd web && npm install

# terminal 1
NEBULA_TRAINING_BACKEND=subprocess uvicorn backend.api.main:app --reload
# terminal 2
cd web && npm run dev
```

`NEBULA_TRAINING_BACKEND=subprocess` (the default) launches training as a local child process, same as before this restructure -- no GPU node or `pipeline/service` needed for local dev.

## Environment variables

| Variable | Used by | Purpose |
|---|---|---|
| `NEBULA_DATA_ROOT` | backend, pipeline | Root for downloaded models/datasets. Point both nodes at the same shared mount in prod. |
| `NEBULA_RUNS_ROOT` | backend, pipeline | Root for per-job workspaces (config, events, checkpoints, artifacts). Same shared mount. |
| `NEBULA_TRAINING_BACKEND` | backend | `subprocess` (default, local dev/tests) or `http` (prod, talks to `TRAINING_SERVICE_URL`). |
| `TRAINING_SERVICE_URL` | backend | Base URL of `pipeline/service` on the GPU node. Required when `NEBULA_TRAINING_BACKEND=http`. |
| `NEBULA_CORS_ORIGINS` | backend | Comma-separated allowed origins for the frontend. Defaults to `*` (dev only). |
| `VITE_API_BASE` | web (build) | API base the SPA calls. Default `/api` (same-origin, proxied by nginx); `web/.env.development` sets `http://localhost:8000` for `npm run dev`. |
| `BACKEND_URL` | web (runtime) | Where the frontend container's nginx proxies `/api/`. Default `http://backend:8000`; `http://localhost:8000` when co-located in one pod. |

## Tests

```bash
pip install -e shared/nebula-events -e "backend[dev]" -e "pipeline[dev]"
pytest
```

Backend tests default to `NEBULA_TRAINING_BACKEND=subprocess` and a stub trainer entrypoint (`backend/tests/fixtures/stub_trainer.py`), so they run without a GPU or the pipeline's ML dependencies.

See `pipeline/README.md` for the checkpoint pipeline's own architecture deep dive (chunking, parallel writer, storage backends, benchmark numbers).
