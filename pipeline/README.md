# PyTorch LoRA Fine-Tuning with BlobStore-Backed Parallel Checkpointing

This is a LoRA fine-tuning pipeline for Qwen-family causal LMs
(`train_lora.py`) whose checkpoint persistence goes through NebulaAI's
`eclib::BlobStore` (`ddl/data/chunk_store.h`) instead of `torch.save()`,
via a chunked format and a bounded, shard-based parallel writer
(`checkpoint/`). `benchmark_checkpoint.py` measures whether that parallel
write path actually helps.

> **A note on `ddl/` paths in this doc.** `ddl/` refers to the external
> eclib/NebulaR source tree (its native `eclib::BlobStore` and DDL3
> `ddl_client` extensions) -- it is **not** a directory in this repo. It's
> built separately from its own source and put on `PYTHONPATH` (see
> "Building the native extension" below, and the build hints in
> `checkpoint/blobstore_backend.py` / `checkpoint/ddl_backend.py`).
> `--checkpoint-storage local` (`checkpoint/localfs_backend.py`) needs
> neither extension and works out of the box.

## 1. Architecture

```
                 PyTorch Training (train_lora.py)
                          |
                          v
                    LoRA Adapters (peft)
                          |
                          | save_checkpoint()
                          v
              BlobStoreCheckpointManager (checkpoint/manager.py)
                          |
        +-----------------+------------------+
        |                 |                  |
   model tensors   optimizer state    scheduler + training
   (LoRA only)      (flattened via     state (rng, epoch,
                     state_flatten.py)  global_step, ...)
        |                 |                  |
        +--------+--------+------------------+
                 v
        tensor_io.describe_tensor()  (GPU->CPU staging if needed;
                 |                    tensor_adapter.py)
                 v
        chunker.iter_chunks()  (zero-copy byte-range slices)
                 |
                 v
        ParallelWriter (checkpoint/parallel_writer.py)
                 |
     +-----------+-----------+-----------+
     v           v           v           v
  Shard 0     Shard 1     Shard 2   ...  Shard N-1
  (own eclib::BlobStore instance, own worker thread)
     |           |           |           |
     +-----------+-----------+-----------+
                 v
        manifest.json + catalog.json  (commit)
```

Each box above maps to a real file:

| Component | File |
|---|---|
| Tensor <-> bytes, dtype/shape preserved, no pickle | `checkpoint/serializer.py`, `checkpoint/tensor_adapter.py` |
| Framework dispatch (numpy vs torch) | `checkpoint/tensor_io.py` |
| Optimizer/scheduler/RNG state <-> JSON skeleton + tensor leaves | `checkpoint/state_flatten.py` |
| Fixed-size, zero-copy chunking | `checkpoint/chunker.py` |
| Manifest schema, checkpoint id / status | `checkpoint/format.py` |
| eclib::BlobStore Python adapter (one shard) | `checkpoint/blobstore_backend.py`, `ddl/python/blobstore_bindings.cpp` |
| Bounded N-worker parallel writer/verifier | `checkpoint/parallel_writer.py` |
| Checksum / size / manifest-status validation | `checkpoint/validation.py` |
| Orchestration, commit protocol, resume | `checkpoint/manager.py` |
| CLI entry points | `train_lora.py`, `benchmark_checkpoint.py` |

## 2. How LoRA training works here

`train_lora.py` loads a base Qwen causal LM frozen in place and wraps it
with `peft.get_peft_model()` using a `LoraConfig` built from CLI flags
(`--lora-r`, `--lora-alpha`, `--lora-dropout`, `--target-modules`,
`--lora-bias`). Only the LoRA adapter parameters are trainable; the base
model's weights are never part of the checkpoint (`--save-full-model` is
available but off by default -- see manager.py's `full_model_state`
parameter). Standard AdamW + a linear warmup/decay schedule
(`transformers.get_linear_schedule_with_warmup`) drive training, with
gradient accumulation (`--gradient-accumulation-steps`) and optional AMP
(`torch.cuda.amp.GradScaler` when CUDA is available).

## 3. Checkpoint format

A checkpoint is not one blob -- it's a manifest plus many independently
addressable chunk objects:

```
ckpt/<checkpoint_id>/
    manifest.json
    model/<tensor_name>/chunk_000000, chunk_000001, ...
    optimizer/<flattened_path>/chunk_000000, ...
    optimizer/skeleton.json           # JSON structure, tensor leaves marked
    scheduler/state.json              # JSON structure (no tensor leaves expected)
    training_state/state.json         # JSON structure
    training_state/<flattened_rng_path>/chunk_000000, ...   # RNG-state tensors
ckpt/<run_id>/catalog.json            # {checkpoint_id, status, epoch, global_step, timestamp}[]
```

Every `BlobId` (`ddl/core/types.h`) is just a string key -- there's no
real directory hierarchy in BlobStore, so these are naming conventions, not
filesystem paths.

Example manifest (trimmed):

```json
{
  "checkpoint_id": "epoch-2-step-160",
  "format_version": 1,
  "status": "COMPLETE",
  "epoch": 2,
  "global_step": 160,
  "best_metric": 0.42,
  "model_name_or_path": "Qwen/Qwen2.5-0.5B",
  "base_model_id": "Qwen/Qwen2.5-0.5B",
  "dataset_id": "onboarding-tutor-v1",
  "lora_config": {"r": 8, "lora_alpha": 16, "target_modules": ["q_proj"], "bias": "none"},
  "training_config": {"learning_rate": 0.0002, "per_device_train_batch_size": 2},
  "num_shards": 8,
  "chunk_size_bytes": 268435456,
  "timestamp": 1234567890.1,
  "model_tensors": [
    {
      "name": "base_model.model.layers.0.self_attn.q_proj.lora_A.weight",
      "dtype": "torch.float32",
      "shape": [8, 4096],
      "numel": 32768,
      "byte_size": 131072,
      "chunk_size": 268435456,
      "chunks": [
        {"index": 0, "blob_id": "ckpt/epoch-2-step-160/model/base_model.model.layers.0.self_attn.q_proj.lora_A.weight/chunk_000000",
         "shard": 3, "offset": 0, "length": 131072, "sha256": "…"}
      ]
    }
  ],
  "optimizer_tensors": ["…same shape as model_tensors, names like optimizer.state.0.exp_avg…"],
  "optimizer_skeleton_blob": {"name": "…", "blob_id": "ckpt/epoch-2-step-160/optimizer/skeleton.json", "shard": 0, "length": 512, "sha256": "…"},
  "scheduler_state": {"blob_id": "ckpt/epoch-2-step-160/scheduler/state.json", "shard": 0, "length": 96, "sha256": "…"},
  "training_state": {"blob_id": "ckpt/epoch-2-step-160/training_state/state.json", "shard": 0, "length": 210, "sha256": "…"},
  "training_state_tensors": ["…RNG generator state arrays…"]
}
```

### Tensor serialization

Each `TensorRecord` (`checkpoint/format.py`) carries `dtype` (a string like
`"torch.bfloat16"`, not a numpy dtype -- numpy can't represent bfloat16),
`shape`, `numel`, `byte_size`, `chunk_size`, and its list of `ChunkRecord`s.
No pickle touches tensor payload anywhere: `tensor_adapter.py` reinterprets
a CPU tensor as a `torch.uint8` view (`tensor.view(torch.uint8)`) before
handing it to `.numpy()`, so even a dtype numpy has no concept of round-trips
correctly as raw bytes; `tensor_adapter.reconstruct_torch_tensor()` reverses
this with `torch.frombuffer(...).view(dtype)`. Optimizer/scheduler/RNG
state (which mix tensors with plain Python scalars in arbitrarily nested
dicts/lists/tuples) go through `checkpoint/state_flatten.py`: it walks the
structure, replaces every tensor leaf with a `{"__tensor__": "<path>"}`
marker in a JSON-safe skeleton, and returns the tensor leaves separately so
they get the same chunked/checksummed treatment as model weights (rather
than pickling the whole optimizer state dict, which is what `torch.save`
would otherwise do).

## 4. Chunking strategy

`checkpoint/chunker.py` splits a tensor's flattened byte buffer into
`--checkpoint-chunk-size-mb`-sized pieces (configurable; there is no fixed
chunk size in the code; the default is 256 MiB, or 4 MiB with
`--checkpoint-storage ddl`, where `--checkpoint-workers` also defaults to 2
instead of 4: every DDL connection pre-allocates roughly 30x the chunk size,
so raise either with care, and keep the DDL target's `--max-object-mb` at
least the chunk size plus 8 bytes). Chunking is zero-copy: `iter_chunks()` returns
`memoryview` slices into the original buffer, so a 2 GB tensor split into
256 MB chunks never gets copied into 8 separate Python objects -- the
slices all alias the same underlying memory. A tensor smaller than the
chunk size still gets exactly one chunk, so tiny and huge tensors go through
the same code path.

**Packing small chunks.** A chunk of at least the pack size is stored as its
own object (its own `BlobId`). Smaller chunks -- whole small tensors, and the
short tail of a large one -- are concatenated, in order, into shared "pack"
objects of up to `--checkpoint-pack-size-mb` (default 4, capped at the chunk
size; 0 disables packing). Each chunk's manifest record still describes it
tensor by tensor and adds `blob_offset`/`blob_length` saying where its slice
lives; loading reads a pack once and slices it, and every member keeps its
own SHA-256. This matters on a network store, where a LoRA adapter is hundreds
of tensors of a few KiB each and the per-object round trip dominates: against a
real DDL3 target a 768-tensor, 13 MB checkpoint went from 768 objects and
~8 s to save (2 workers) to 5 objects and ~0.9 s. Manifests carry
`format_version` 2; version-1 manifests (no packing fields) still load.

### Inspecting a stored checkpoint

A DDL3 target has no listing operation and keys objects by hash, so a checkpoint
written with `--checkpoint-storage ddl` cannot be browsed with `ls` (the run
directory only holds `config.json`, `events.jsonl` and the exported
`artifacts/final_adapter`; the resumable state lives on the target). It is
reachable by name though: run id -> catalog -> manifest -> chunk records.
`pipeline/checkpoint/inspect_checkpoint.py` follows that chain, read-only:

```bash
python -m pipeline.checkpoint.inspect_checkpoint --run-id <job id> \
    --ddl-server 192.168.11.87 --ddl-tenant 7 --blobs --tensors --grep q_proj --verify
```

The run id is the job id (the `runs/<id>` directory name) and the tenant must match
the backend's `ddl_tenant` setting. It lists the catalog, summarises the latest (or
`--checkpoint-id`) checkpoint, and optionally shows the stored objects, each
tensor's dtype/shape/location (`pack_000000 @28672`), the raw manifest, and a full
SHA-256 re-read. Use `--local-dir <run dir>/checkpoints` for local-storage
checkpoints.

### Resuming from a checkpoint

Run the training script with the checkpoint's `--run-id` and `--resume <checkpoint id>`
(or `latest`), with `--epochs` set to the **total** to reach: a checkpoint taken after
epoch 1 resumed with `--epochs 3` trains epochs 2 and 3. From the UI this is the
"Resume from a previous run" control on the training step, which sends
`checkpoint.resume_from = {job_id, checkpoint_id}`; the backend checks the source job
is finished and used the same checkpoint storage, base model and LoRA rank, and runs
the new job under the source job's run id so it finds (and keeps adding to) that run's
checkpoints.

An end-of-epoch checkpoint records the number of *completed* epochs, so resuming it
does not repeat that epoch. Checkpoints written before this was fixed (named
`epoch-0-step-N` after the first epoch) recorded the epoch index instead, and resuming
one repeats its last epoch. A mid-epoch checkpoint (`--checkpoint-every-steps`) records
the epoch in progress, so resuming it restarts that epoch from its beginning.

## 5. Parallel writer architecture

BlobStore itself is documented **single-threaded**
(`ddl/data/chunk_store.h`): one `BlobStore` instance must only ever be
called from one thread. Its only built-in concurrency wrapper,
`AsyncBlobStore`, decouples the *caller* from execution but still runs
every request through a single worker thread (`ddl/data/async_blob_store.h`
declares one `std::thread worker_`) -- it does not parallelize writes. The
fully parallel implementation, `ReactorBlobStore`, is a separate
SPDK-reactor-based reimplementation that needs real NVMe/SPDK hardware and
was out of scope for this environment.

So parallelism here is built at the Python layer, without touching
BlobStore's internals, via **sharding**: `blobstore_backend.create_shards(N)`
constructs N independent `eclib::BlobStore` instances (each with its own
`ECLibConfig`-driven device pool), and `checkpoint/parallel_writer.py`
gives each one exactly one dedicated OS thread for its entire lifetime.
Chunks are assigned to shards round-robin
(`BlobStoreCheckpointManager._next_shard()`), submitted through one bounded
queue per shard, and the GIL is released for the duration of every native
`put_blob`/`get_blob` call (`ddl/python/blobstore_bindings.cpp`), so the N
worker threads genuinely execute concurrently. Because a shard is only ever
touched by its one owning thread, this respects BlobStore's single-threaded
contract with zero locking added to BlobStore itself -- `--checkpoint-workers`
directly controls N.

Backpressure: each per-shard queue has a bounded `queue_maxsize_per_shard`
(default 8), so a checkpoint with far more chunks than the queue can hold
blocks the producer instead of buffering every pending chunk's bytes in
memory at once.

## 6. GPU -> CPU data path

`tensor_adapter.stage_to_cpu()` copies a CUDA tensor into a pinned CPU
staging buffer with `non_blocking=True`; `manager._plan_chunked_group()`
issues this for every tensor in a group before any chunking/hashing
happens, so the D2H copies for later tensors can be in flight while earlier
ones are still copying, rather than a blocking `tensor.cpu()` per tensor.
The `gpu_to_cpu_s` timer in `utils/metrics.py` measures exactly this phase.
After staging, everything is zero-copy: `describe_tensor()` produces a
`memoryview` directly over the staged tensor's storage (via a `uint8` view,
see above), `chunker.iter_chunks()` slices that same memoryview, and
`blobstore_bindings.cpp`'s `put_blob()` takes the raw pointer straight from
that slice's buffer-protocol export -- there is no
"tensor -> Python bytes -> serialized buffer -> C++ buffer" copy chain.
The one place a real copy happens on the write path is `stage_to_cpu()`
itself (unavoidable: it's the actual device-to-host transfer) and, for a
non-contiguous source tensor, the `.contiguous()` call before that.

## 7. Resume semantics

`--resume latest` calls `manager.get_latest_checkpoint()` (reads the
catalog only, so a merely-written-but-uncommitted checkpoint is never
picked); `--resume <id>` loads that id directly. `resume_training_state()`
in `train_lora.py` restores, in order: LoRA adapter weights
(`peft.set_peft_model_state_dict`), the full optimizer state dict
(unflattened via `state_flatten.unflatten_state_dict`, so momentum/Adam
state comes back, not just the weights), the scheduler state dict, and
python/numpy/torch(+CUDA) RNG state (`utils/seed.py`) -- so training
continues from the same point in the data/RNG stream, not a fresh
restart with old weights.

## 8. Failure handling

- **BlobStore write failure**: `eclib::BlobStore::putBlob()` has no
  failure signal in its return type, and (confirmed by reading
  `chunk_store.cpp`) a failed device write inside it is not currently
  surfaced at all -- a real, pre-existing gap, not something invented here.
  So every chunk write is followed by an inline check
  (`--checkpoint-verify-mode checksum`, the default): a full read-back +
  SHA-256 compare, fanned out across the same N-shard worker pool as the
  writes (`VerifyJob` in `parallel_writer.py`) so it doesn't become a
  serial bottleneck (see "Performance" below for why that matters).
- **Worker failure**: an exception raised inside a worker thread's
  `put`/`get` call is caught in `_worker_loop` and reported as a failed
  `WriteResult`/`VerifyResult`, not raised into the thread and silently
  dropped -- see `test_parallel_writer.py::test_parallel_writer_reports_failure_without_raising`.
- **Any chunk write or verification failure** aborts the commit:
  `save_checkpoint()` raises `CheckpointWriteError` *before* writing the
  manifest or updating the catalog, so a failed checkpoint is never
  discoverable via `get_latest_checkpoint()`/`list_checkpoints()`
  (`test_failure_handling.py::test_write_failure_prevents_commit`).
- **Interrupted/incomplete checkpoint**: `list_checkpoints()`/
  `get_latest_checkpoint()` only return `status == "COMPLETE"` catalog
  entries; `load_checkpoint(id)` raises `CheckpointIncompleteError` unless
  called with `allow_incomplete=True` (the "unless explicitly requested"
  escape hatch).
- **Missing/corrupt chunk on load**: `_read_tensor_group()` re-checks
  every chunk's SHA-256 while reconstructing a tensor and raises
  `CheckpointCorruptError` on mismatch, or lets the native
  `BlobStoreError` (NotFound) propagate for a missing chunk -- never a
  silently wrong tensor.
- **Invalid manifest**: a corrupt `manifest.json` blob fails
  `json.loads()` inside `_read_manifest()` and raises, rather than being
  interpreted as a valid (if empty) manifest.
- **On-demand audit**: `manager.verify(checkpoint_id)` re-reads and
  re-hashes every chunk/object against the manifest and returns a
  `VerifyReport` (chunks/objects checked, list of issues, `ok` flag) --
  usable at any time, not just right after a save.

## 9. Commit protocol (atomicity)

BlobStore has no multi-object transaction API (`flush()` is a whole-store
durability barrier, not a per-checkpoint commit). The smallest safe
protocol built on top of what exists:

1. Chunk + hash + submit every model/optimizer/training-state tensor to
   the parallel writer (one fan-out across all groups, not phase-by-phase,
   to maximize parallelism).
2. Collect all write results; fan out checksum verification the same way.
3. If *anything* failed: raise `CheckpointWriteError`. Nothing further is
   written -- no manifest, no catalog entry.
4. Write the scheduler/training-state/optimizer-skeleton JSON objects.
5. Write `manifest.json` (status `"COMPLETE"`) -- one blob, one atomic
   `putBlob` call.
6. Update `catalog.json` last, appending/replacing this checkpoint's
   entry. **This is the actual commit point**: discovery
   (`list_checkpoints`/`get_latest_checkpoint`) only ever reads the
   catalog, never scans for manifests (BlobStore has no list/scan API --
   see "Known limitations"). A crash between steps 5 and 6 leaves a fully
   written, checksummed checkpoint that simply isn't in the catalog yet;
   `load_checkpoint(id, allow_incomplete=True)` can still read it directly.

## 10. Performance measurement

`CheckpointMetrics` (`utils/metrics.py`) is populated entirely from real
`time.perf_counter()` deltas and real byte counts taken during
`save_checkpoint()` -- nothing is estimated. Example real output (from
`benchmark_checkpoint.py`, 512 MB synthetic payload / 128 tensors / 1 MB
chunks, 12-core sandbox, no GPU):

```
============================================================
CHECKPOINT
============================================================
Checkpoint: bench-checkpoint
Size:       512.00 MB
Tensors:    128
Chunks:     512
Chunk size: 1 MB
Workers:    4

GPU->CPU:   0.000 sec
Chunking:   0.278 sec
Writes:     0.264 sec
Commit:     0.001 sec

Total:      0.544 sec
Throughput: 942.04 MB/s
============================================================
```

### Benchmark: worker count vs. throughput

Same 512 MB / 128-tensor / 1 MB-chunk payload, saved through a fresh
8-shard-configured manager at each worker count (`--checkpoint-verify-mode
checksum`, the default):

| workers | checkpoint_time | throughput |
|--------:|-----------------:|-----------:|
| 1  | 0.841 s | 608 MB/s |
| 2  | 0.597 s | 858 MB/s |
| 4  | 0.544 s | 942 MB/s |
| 8  | 0.551 s | 929 MB/s |
| 16 | 0.639 s | 801 MB/s |

Reproduce with:
```
python -m training.benchmark_checkpoint --workers 1,2,4,8,16 \
    --size-mb 512 --chunk-size-mb 1 --num-tensors 128
```

**Reading this honestly**: going from 1 to 4 workers gives a real ~55%
throughput improvement (BlobStore's parallel-shard writes doing their
job); beyond 4 workers on this 12-logical-core sandbox, throughput
plateaus (8) and then regresses (16) as thread/queue overhead and CPU
oversubscription outweigh the added concurrency for a 512 MB checkpoint.
This is expected for a CPU-bound simulated backend (each `putBlob` does
real Reed-Solomon EC encoding, not just a memcpy) with a fixed core count
-- it is not evidence that more workers never help; it says the useful
worker count for *this* payload size and *this* machine tops out around
4-8. Larger payloads, more real CPU cores, or a real (non-simulated,
I/O-bound rather than CPU-bound) device backend would shift that curve.
Do not read more into the specific numbers than that -- they are one real
measurement on one sandbox, not a general claim about BlobStore.

An earlier version of the inline write-validation step ran its checksum
re-reads in a plain sequential Python loop; that added an
O(checkpoint size) serial cost to *every* run regardless of worker count
and completely masked the speedup above (1/2/4/8/16 workers all landed
around 0.9-1.1 s). Fanning that verification out across the same N-shard
worker pool as the writes (`VerifyJob` in `parallel_writer.py`) fixed it.
This is worth calling out because it is exactly the kind of bug that
silently defeats a "why bother parallelizing" measurement without erroring
or looking wrong in isolation.

## 11. Running it

```bash
# from the repo root
python -m pipeline.train_lora \
    --model-path Qwen/Qwen2.5-0.5B \
    --dataset ./data/onboarding_tutor.jsonl \
    --output-dir ./outputs/onboarding-tutor \
    --epochs 3 --batch-size 2 --gradient-accumulation-steps 8 \
    --learning-rate 2e-4 --checkpoint-every-epoch \
    --checkpoint-workers 8 --checkpoint-chunk-size-mb 256

# resume
python -m pipeline.train_lora ... --resume latest

# benchmark
python -m training.benchmark_checkpoint --workers 1,2,4,8,16 \
    --size-mb 512 --chunk-size-mb 8 --num-tensors 128
```

`train_lora.py` needs `torch`, `transformers`, `peft`, and `datasets`
installed, plus the `eclib_blobstore` native extension on `PYTHONPATH`
(see "Building the native extension" below). Neither could be exercised
in the environment this was developed in (see "Known limitations").

### Inspecting / verifying a checkpoint

```python
from pipeline.checkpoint.manager import BlobStoreCheckpointManager

mgr = BlobStoreCheckpointManager(run_id="onboarding-tutor", num_workers=8)
print(mgr.list_checkpoints())
print(mgr.get_latest_checkpoint())
report = mgr.verify("epoch-2-step-160")
print(report.ok, report.issues)
```

Note the "process-lifetime persistence" limitation below: this only works
within the same process that wrote the checkpoint (or one that constructs
its shards identically), not across a restart of the training script,
until BlobStore has a real device backend.

### Building the native extension

```bash
cmake -S ddl -B ddl/build -DCMAKE_CXX_STANDARD=17 \
    -DECLIB_ENABLE_PYTHON_BINDINGS=ON \
    -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
cmake --build ddl/build --target eclib_blobstore -j
export PYTHONPATH=$PWD/ddl/build:$PYTHONPATH
```

## 12. Known limitations

- **BlobStore's storage backend is a pure in-process memory simulation.**
  `ddl/sim/mocks/device_sim.h`'s `DeviceSimulator` stores everything in a
  `std::vector<uint8_t>` and `PmemBuffer` (`ddl/data/pmem_buffer.h`) does
  the same despite its name -- there is no real file/NVMe backing
  anywhere in `ddl/`, and `ECLibConfig::cluster_mode = "production"` is a
  string nothing currently reads. This means: (a) a checkpoint does not
  survive the process that wrote it exiting; (b) shards do not share
  storage with each other -- each is its own independent simulated device
  pool; (c) the throughput numbers above measure this codebase's real
  in-process concurrency and EC-encoding cost, not real disk/NVMe I/O.
  Per the accepted scope for this task, no new persistent device backend
  was added to `ddl/` -- the checkpoint pipeline is built correctly against
  what exists today and this limitation is documented rather than
  papered over.
- **No torch/transformers/peft/datasets/GPU in the development
  environment.** `train_lora.py` and `checkpoint/tensor_adapter.py` are
  written against current stable APIs of those libraries but were not
  executed here. Everything that *can* be exercised without them --
  serialization, chunking, manifest format, checksum validation, the
  BlobStore Python bindings, the parallel writer, and the full
  save/load/verify/resume-support round trip using numpy tensors standing
  in for torch tensors -- has real, passing tests (`training/tests/`).
- **Read-path (`load_checkpoint`) is not parallelized.** The write path's
  parallelism was the point of this exercise; reads currently fetch
  chunks sequentially per tensor. Extending `ParallelWriter`'s job model
  (already generic enough after adding `VerifyJob`) to a `ReadJob` would
  be a natural, small follow-up.
- **BlobStore has no list/prefix-scan API.** `checkpoint discovery`
  (`list_checkpoints`/`get_latest_checkpoint`) is therefore backed by an
  application-level catalog blob (`ckpt/<run_id>/catalog.json`), not a
  BlobStore capability -- documented in the commit-protocol section
  above rather than silently assumed.
- **Scheduler state is assumed tensor-free.** `save_checkpoint()` raises
  if `scheduler.state_dict()` ever contains a tensor leaf, since this
  format has no storage channel for one (standard torch/HF LR schedulers
  never produce one in practice).
- Two small pre-existing, unrelated build breaks in `ddl/` were fixed as
  part of getting a baseline build working (see git history):
  `ddl/CMakeLists.txt`'s `eclib_tests` target referenced a deleted
  `tests/run_tests.cpp` and never linked `GTest::gtest_main` despite every
  test file already using `TEST()`/gtest macros, and `eclib` itself wasn't
  built with `-fPIC` (needed once linked into a `.so`). `tests/test_full_sim.cpp`
  was excluded from the standalone `ddl/CMakeLists.txt` build (root
  `CMakeLists.txt` already excludes/handles it separately) because it
  targets a `keyspace` API (`eclib::hash_keyspace_key`, `KeyspaceKey`
  fields) that has since moved/changed shape -- unrelated to BlobStore,
  left alone.
