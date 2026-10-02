# Pipeline architecture

Document type: Description.
Language: Simplified Technical English (ASD-STE100 style).

## 0 About this document

### 0.1 Purpose

This document describes the full architecture of the fine-tuning pipeline.
It covers the web application, the backend, the training agent, the checkpoint system, and the DDL3 storage target.
It also describes how to build and start the system.

Use this document to learn which part does which task.
Use it also to find the cause of a failure.

### 0.2 How the text is written

The text follows the rules of ASD-STE100 as closely as is practical.
A writer applied the rules by hand and checked sentence length with a script.
No licensed STE checker has approved the text.

These are the main rules:

- A descriptive sentence has 25 words or fewer.
- An instruction has 20 words or fewer.
- A paragraph has 6 sentences or fewer.
- Each term has one meaning.
- Instructions use the imperative form.

Technical names are not changed.
Examples are file names, command names, and field names.

### 0.3 Terms

Each term below has only one meaning in this document.

| Term | Meaning |
|---|---|
| Job | One request to train a model. A job has an ID. |
| Run ID | The name that groups the checkpoints of one training run. A new job uses its own job ID as its run ID. A resumed job uses the run ID of the source job. |
| Epoch | One full pass over the training data. |
| Step | One update of the model weights. |
| Checkpoint | A saved copy of the training state at one point in time. |
| Tensor | A named array of numbers, for example one LoRA weight. |
| Chunk | A part of one tensor. A small tensor has one chunk. |
| Pack | One stored object that holds many small chunks. |
| Object | One named block of bytes in a storage backend. |
| Manifest | The file that lists all tensors, chunks, and objects of one checkpoint. |
| Catalog | The file that lists all checkpoints of one run ID. |
| Shard | One storage connection that one writer thread uses. |
| Target | The DDL3 storage server. |
| Tenant | A number that separates the data of different users on the target. |
| Slab | A fixed-size block of registered memory that holds one payload in transit. |
| SM, MM, LM | Small, medium, and large message modes of the DDL3 protocol. |

## 1 System overview

### 1.1 What the system does

The system fine-tunes a language model with LoRA.
LoRA trains a small set of extra weights and keeps the base model unchanged.
The user selects a model, a dataset, and training settings in a web page.
The system then trains the model and saves checkpoints.
It also exports the final LoRA adapter for download.

### 1.2 The parts

| Part | Directory | Task |
|---|---|---|
| Web application | `web/` | Shows the user interface. Sends requests to the backend. |
| Backend | `backend/` | Stores job records. Starts jobs. Streams progress to the browser. |
| Training agent | `pipeline/service/` | Receives start requests. Starts the training process. |
| Training script | `pipeline/train_lora_with_gpu_stats.py` | Trains the model. Writes checkpoints. Writes events. |
| Checkpoint system | `pipeline/checkpoint/` | Splits, packs, writes, reads, and verifies checkpoints. |
| Event package | `shared/nebula-events/` | Defines the event format that the backend and the agent share. |
| DDL storage package | `shared/nebula-ddl-storage/` | Connects Python code to the DDL3 target. |
| `ddl_client` package | NebulaR repository, `client/python/` | Native Python module that sends PUT and GET requests to the target. |
| DDL3 target | NebulaR repository, `target/` | Stores objects on a device file or a disk. |

The backend never imports torch, transformers, or peft.
Therefore the backend does not need a GPU.

### 1.3 Diagram

```text
 Browser
    |  HTTP (port 5173)
    v
 Web application (nginx, static files)
    |  HTTP (port 8000)
    v
 Backend (FastAPI, SQLite)
    |  HTTP (port 8001)         reads events.jsonl
    v                               ^
 Training agent (FastAPI)           |
    |  starts a process             |
    v                               |
 Training script  -----------------+  writes events.jsonl
    |
    +--> Checkpoint system
            |
            +--> Local disk   (runs/<job>/checkpoints)
            |
            +--> DDL storage package --> ddl_client --> network (UCX)
                                                            |
                                                            v
                                                     DDL3 target --> MiniFS --> disk
```

### 1.4 How the parts communicate

The parts use four channels.

1. The browser and the backend use HTTP and Server-Sent Events.
2. The backend and the training agent use HTTP.
3. The training script writes an events file. The backend reads that file.
4. The checkpoint system and the DDL3 target use the UCX network library.

The backend and the agent must see the same shared directory.
In the compose setup, a Docker volume named `nebula-shared` provides it.
In production, a network file system provides it.
Both parts mount it at `/mnt/nebula-shared`.

## 2 Web application

The web application is a React single-page application.
Vite builds it.
An nginx server serves the built files.

The application sends all requests to the backend.
The base URL comes from `VITE_API_BASE`.
The build process writes this value into the files.
To change it, build the image again.

| Page | Task |
|---|---|
| Dashboard | Shows a summary and recent activity. |
| Training (Jobs) | Lists jobs. |
| Create Fine-Tuning Job | Collects the settings in three steps: Model, Data, and Training. |
| Job detail | Shows live progress, a loss chart, checkpoints, and the adapter download. |
| Models | Lists models. Adds a model from Hugging Face. |
| Datasets | Lists datasets. Creates or imports a dataset. |
| Settings | Sets the DDL3 target address and the default storage choices. |
| Checkpoints | Placeholder. It links to the job pages. |
| Deployments | Placeholder. Serving a trained adapter is not available. |

The Training step of the job form has a control named "Resume from a previous run".
When the user selects a finished job, the form copies the base model, the dataset, and the LoRA settings from that job.
The form also sets the epoch count higher than the count that the checkpoint completed.

## 3 Backend

### 3.1 Tasks

The backend is a FastAPI service.
It has four tasks.

1. It stores job, model, dataset, and settings records.
2. It checks and accepts new jobs.
3. It starts jobs on the training agent.
4. It reads the events of a job and sends them to the browser.

### 3.2 API

| Method and path | Task |
|---|---|
| `GET /config` | Returns the choices for the job form. |
| `GET /activity` | Returns recent activity. |
| `POST /jobs` | Creates a job in the state SUBMITTED. |
| `GET /jobs` | Lists jobs. |
| `GET /jobs/{id}` | Returns one job. |
| `POST /jobs/{id}/start` | Starts a job. |
| `DELETE /jobs/{id}` | Deletes a job that has finished or has not started. |
| `GET /jobs/{id}/events` | Streams events as Server-Sent Events. |
| `GET /jobs/{id}/checkpoints` | Lists the checkpoints that the job saved. |
| `GET /jobs/{id}/artifact` | Downloads the final adapter of a completed job. |
| `/models`, `/datasets` | Create, list, read, and delete models and datasets. |
| `GET /settings`, `PUT /settings` | Read and change the settings. |

`POST /jobs` returns status 422 and a reason when a resume request cannot work.

### 3.3 Job states

A job is always in one state.
A job can make only the changes in the table below.

| From | To |
|---|---|
| SUBMITTED | QUEUED, FAILED |
| QUEUED | RUNNING, FAILED |
| RUNNING | CHECKPOINTING, EVALUATING, COMPLETED, FAILED |
| CHECKPOINTING | RUNNING, EVALUATING, COMPLETED, FAILED |
| EVALUATING | RUNNING, COMPLETED, FAILED |
| COMPLETED | none |
| FAILED | none |

COMPLETED and FAILED are the final states.
The Job Manager applies an event to the state only through this table.

### 3.4 The Job Manager

The Job Manager starts a job in four steps.

1. It reads the saved job configuration.
2. It builds the command-line flags for the training script.
3. It sends the flags to the training client.
4. It starts a thread that reads the events file.

The thread checks the events file every 0.2 seconds.
It turns each event into a state change.
It also sends each event to every browser that listens.
It sends a heartbeat every 15 seconds.

The backend has two training clients.
The `subprocess` client starts the script on the same machine.
The `http` client sends a request to the training agent.
The variable `NEBULA_TRAINING_BACKEND` selects the client.

### 3.5 Stored data

The backend keeps job records, settings, models, and datasets in one SQLite file.
The file is `nebula.db` in the data directory.
Each job also has a directory in the runs directory.

| File | Content |
|---|---|
| `config.json` | The job configuration that the user submitted. |
| `events.jsonl` | The events that the training script wrote. |
| `train.log` | The output of the training process. |
| `checkpoints/` | The checkpoints, only when the job uses local storage. |
| `artifacts/final_adapter/` | The final LoRA adapter. |

A job that uses DDL storage has no `checkpoints/` directory.
The target holds its checkpoints.

### 3.6 Settings

The settings hold the address of the DDL3 target.
They hold these fields: `ddl_server`, `ddl_port`, `ddl_tenant`, and `ddl_cpu_base`.
They also hold the default storage choice for models, datasets, and checkpoints.

Each job has its own storage choice in its configuration.
The settings only supply the default value in the form.
The backend reads the target address from the settings when it starts a job.

### 3.7 Resume rules

A job can continue from a checkpoint of an earlier job.
The configuration field is `checkpoint.resume_from`.
It holds a job ID and a checkpoint ID or the word `latest`.

The backend rejects the request with status 422 in these cases:

- The source job does not exist.
- The source job has not finished.
- The source job used a different checkpoint storage.
- The source job used a different base model.
- The source job used a different LoRA rank.
- The source job saved no checkpoint, or it did not save the checkpoint that the request names.
- The new epoch count is not higher than the epochs that the checkpoint completed.

A resumed job uses the run ID of the source job.
Its events and artifacts stay in its own job directory.
For local storage, it uses the checkpoint directory of the source job.

The epoch count of the new job is the total to reach.
A checkpoint after epoch 1 that resumes with an epoch count of 3 trains epochs 2 and 3.

## 4 Training agent

The training agent is a small FastAPI service.
It runs in the `training-service` container.
It has three endpoints.

| Method and path | Task |
|---|---|
| `GET /healthz` | Returns the health of the service. |
| `POST /jobs/{id}/start` | Receives the flags. Starts the training script as a process. Returns status 202. |
| `GET /jobs/{id}/status` | Returns the exit code of the process, or null while it runs. |

The agent starts `python -m pipeline.train_lora_with_gpu_stats` with the flags.
The agent does not train.
It only starts and watches the process.

The agent keeps the process list in memory.
After a restart, the agent does not know the earlier jobs.

## 5 Training script

### 5.1 Stages

The script runs these stages in this order.

1. Load the tokenizer and the base model.
2. Add the LoRA adapters.
3. Load the dataset and split it into training and validation parts.
4. Open the checkpoint manager. This step opens the storage connections.
5. If the job resumes, load the checkpoint.
6. For each epoch, train and write a step event.
7. At the end of each epoch, write a checkpoint.
8. If evaluation is on, calculate the validation loss.
9. After the last epoch, export the adapter.
10. Write the event `job_complete`.

An error at any stage writes the event `job_failed`.
An operating-system kill writes no event.
The agent then reports an exit code and the backend marks the job as FAILED.

### 5.2 Hardware and number format

The script uses a GPU when one is available.
The default number format is fp32 on a CPU.
The default is fp16 on most GPUs.
The option `--dtype` changes the format.

### 5.3 What a checkpoint holds

A checkpoint holds these items:

- the LoRA weights of the model,
- the state of the AdamW optimizer,
- the state of the learning-rate scheduler,
- the epoch, the step, the best metric, and the random-number state.

The option `--save-full-model` also saves the full model.

### 5.4 Memory behavior

A CPU job holds a lot of memory.
For a 0.5-billion-parameter model in fp32, a measured job used about 3.8 to 3.9 GiB. The job used batch size 2 and sequence length 128.
The container limit is 5 GiB.

CAUTION
The kernel kills a job that uses more than the container limit.
The kill gives the exit code -9 and no final event.

The agent container sets two environment variables to lower the peak memory.
They are `MALLOC_MMAP_THRESHOLD_` and `MALLOC_TRIM_THRESHOLD_`.
Both have the value 131072.
They make the C library return large freed blocks to the system.

## 6 Event contract

The training script writes one JSON object per line to `events.jsonl`.
Each object has the fields `ts`, `job_id`, `stage`, and `event`.
Other fields depend on the event.

The package `nebula_events` defines the format.
The backend and the agent both install it.
The backend imports nothing else from the pipeline.

| Event | Stage | Meaning |
|---|---|---|
| `worker_started` | RUNNING | The script started. |
| `model_loading_start`, `model_loaded` | RUNNING | The base model loads. |
| `model_materializing_start`, `model_materialized` | RUNNING | The model comes from DDL storage and is copied to the local disk. |
| `dataset_loading_start`, `dataset_loaded` | RUNNING | The dataset loads. |
| `dataset_materializing_start`, `dataset_materialized` | RUNNING | The dataset comes from DDL storage and is copied to the local disk. |
| `epoch_start` | RUNNING | An epoch begins. |
| `step` | RUNNING | One step finished. It holds the loss. |
| `epoch_complete` | RUNNING | An epoch finished. |
| `checkpoint_start` | CHECKPOINTING | A checkpoint write begins. |
| `checkpoint_saved` | CHECKPOINTING | The checkpoint is committed. It holds sizes and times. |
| `eval_start`, `eval_complete` | EVALUATING | The validation pass runs. |
| `job_complete` | COMPLETED | The job finished. |
| `job_failed` | FAILED | The job stopped with an error. |

The event `checkpoint_saved` holds these fields:
`checkpoint_id`, `epoch`, `global_step`, `size_bytes`, `num_tensors`, `num_chunks`, `num_blobs`, `chunk_size_bytes`, `num_workers`, `write_s`, `commit_s`, `total_s`, and `throughput_mb_s`.
The field `num_chunks` counts chunk records.
The field `num_blobs` counts the stored objects.

## 7 Checkpoint system

### 7.1 Parts

| Part | File | Task |
|---|---|---|
| Checkpoint manager | `manager.py` | Controls save, load, and verify. |
| State flattening | `state_flatten.py` | Separates tensors from the other values in a state. |
| Chunker | `chunker.py` | Splits a tensor into chunks without copying. |
| Format | `format.py` | Defines the manifest and the object names. |
| Parallel writer | `parallel_writer.py` | Runs one thread for each shard. |
| Validation | `validation.py` | Checks the stored data. |
| Storage backend | `storage_backend.py` | Defines the interface that all backends implement. |
| Sizing | `sizing.py` | Selects the default chunk size and writer count. |
| Inspector | `inspect_checkpoint.py` | Shows the content of a stored checkpoint. |

### 7.2 Save procedure

The manager saves a checkpoint in this order.

1. Split the state into tensors and a JSON description.
2. Split each tensor into chunks and calculate a SHA-256 hash for each chunk.
3. Pack the small chunks together into pack objects.
4. Send all objects to the parallel writer.
5. Wait for all writes.
6. Verify the writes, as the verify mode requires.
7. Write the small JSON state objects.
8. Write the manifest.
9. Write the catalog.

The catalog write is the commit point.
Only the catalog makes the checkpoint visible to a search for the latest checkpoint.
A failure before step 9 leaves no visible checkpoint.
In that case the manager raises an error. It writes no manifest and no catalog entry.

### 7.3 Object names

| Name | Content |
|---|---|
| `ckpt/<checkpoint id>/<group>/pack_NNNNNN` | A pack of small chunks. |
| `ckpt/<checkpoint id>/<group>/<tensor>/chunk_NNNNNN` | One large chunk. |
| `ckpt/<checkpoint id>/optimizer/skeleton.json` | The structure of the optimizer state. |
| `ckpt/<checkpoint id>/scheduler/state.json` | The scheduler state. |
| `ckpt/<checkpoint id>/training_state/state.json` | The training state. |
| `ckpt/<checkpoint id>/manifest.json` | The manifest. |
| `ckpt/<run id>/catalog.json` | The catalog. |

The groups are `model`, `optimizer`, `training_state`, and `full_model`.
The checkpoint ID has the form `epoch-<completed epochs>-step-<step>`.

### 7.4 Packing

A LoRA checkpoint has hundreds of small tensors.
On a network store, each object needs a round trip.
The round trips then take most of the time.

The manager packs chunks that are smaller than the pack size.
The default pack size is the smaller of the chunk size and 4 MiB.
The manager packs the chunks in tensor order.
Each chunk record stores `blob_offset` and `blob_length`.

In one measured test, a checkpoint of 768 tensors and 13 MB changed from 768 objects to 5.
The save time changed from 8.1 seconds to 0.9 seconds.
The test used a real target and 2 writers.

### 7.5 Manifest

The manifest is a JSON file.
The current format version is 2.
The loader also reads version 1.
It refuses a version that it does not know.

The manifest holds these items:

- the checkpoint ID, the status, the epoch, and the step,
- the model, the dataset, and the LoRA settings,
- the chunk size and the shard count,
- one record for each tensor, with its dtype, shape, and chunk records,
- one record for each JSON state object.

Each chunk record holds its own SHA-256 hash.
The loader checks the hash of each chunk.

### 7.6 Verify modes

| Mode | Check after the write |
|---|---|
| `checksum` (default) | Read every stored object and compare the SHA-256 hash. |
| `head` | Read only the size of every stored object. |
| `none` | No check. |

The `checksum` mode doubles the data that crosses the network.
Use `head` to reduce the time when the link is slow.

### 7.7 Load and resume

The loader reads the manifest first.
It then reads each pack once and cuts the chunks from it.
It checks each chunk against its hash.

The training script rebuilds torch tensors for torch data types.
It rebuilds numpy arrays for numpy data types.
It then restores the optimizer state, the scheduler state, and the random-number state.
It converts the integer keys of the optimizer state back from text.

An end-of-epoch checkpoint stores the number of completed epochs.
A resume therefore starts the next epoch.
A checkpoint from before this rule (`epoch-0-step-N` after the first epoch) repeats its last epoch.
A step checkpoint stores the epoch in progress.
A resume from it starts that epoch again from the beginning.

### 7.8 Defaults

| Item | DDL storage | Other storage |
|---|---|---|
| Chunk size | 4 MiB | 256 MiB |
| Writer count | 2 | 4 |
| Pack size | 4 MiB | smaller of chunk size and 4 MiB |
| Verify mode | `checksum` | `checksum` |

A DDL connection reserves memory in proportion to the chunk size.
The reason is in section 11.3.
Therefore the DDL defaults are small.

### 7.9 Inspector

A DDL3 target has no list function.
It also hashes the object names.
A user therefore cannot see a checkpoint with a file listing.

The inspector follows the chain of names: the run ID, the catalog, the manifest, and the chunk records.
It only reads.
Use this command:

```bash
python -m pipeline.checkpoint.inspect_checkpoint --run-id <job id> \
    --ddl-server <address> --ddl-tenant <tenant> --blobs --tensors --verify
```

The option `--local-dir` shows a checkpoint on a local disk.

## 8 Storage backends

The manager works with any class that implements the storage interface.
The interface has these operations: `put`, `get`, `head`, `exists`, `delete`, `flush`, and `close`.
A backend creates one shard for each writer thread.

| Backend | Where the data goes |
|---|---|
| Local | Files under the job directory, one directory for each shard. |
| BlobStore | A native library. It is optional and the container does not include it. |
| DDL | The DDL3 target, through the DDL storage package. |

The manager closes all shards when it shuts down.
A backend that holds a network connection must close it.
The target keeps each connection until the client closes it.

## 9 DDL storage package

### 9.1 Tasks

The package `nebula_ddl_storage` connects Python code to the target.
The checkpoint system and the model and dataset storage both use it.

| Class | Task |
|---|---|
| `DdlConnection` | Owns one `ddl_client` connection and one thread. |
| `DdlShard` | Implements the shard interface on one connection. |
| `DdlBackend` | Creates the shards. Gives each a core ID and a connection ID. |
| `DdlResourceStore` | Stores a model or dataset directory as many objects. |

### 9.2 Object names and IDs

The target knows only a 128-bit number for each object.
It does not know text names.
The package therefore turns each name into a number.

For a checkpoint, do these steps:

1. Join the run ID, a slash, and the object name.
2. Calculate the SHA-256 hash of the UTF-8 text.
3. Take the first 16 bytes.
4. Use the first 8 bytes as the high part and the next 8 bytes as the low part.

The name is the same each time.
A reader can therefore find an object from its name alone.

### 9.3 Length frame

The target has no function to read the size of an object.
It also refuses a write of zero bytes.
The package adds an 8-byte length at the start of each object.
The length is a big-endian number.

A size check reads only these 8 bytes.
A read of the content skips them.
A write is always at least 8 bytes long.

### 9.4 Threads and connection IDs

The network library accepts calls from only one thread for each connection.
Each `DdlConnection` therefore owns one thread.
It creates the client in that thread and runs all calls in that thread.
Other threads send their requests through a queue.

The target keeps a table of connections.
It refuses a new connection when an active connection has the same ID.
Each `DdlBackend` therefore picks a random start value for its connection IDs.
Two programs can then use the same target at the same time.

A missing object returns status 1 from the client.
The package changes that status into a not-found error.

## 10 The `ddl_client` package

The package `nebula-ddl-client` holds one native Python module named `ddl_client`.
A pybind11 binding builds the module on top of the DDL3 client library.
The module has a `DdlClient` class with `put`, `get`, `close`, and `ready` methods.
It also has the `DdlError` class, which carries a `status` number, and the string `__version__`.

The module releases the Python GIL while it waits for the network.
The client sends one request at a time.

The NebulaR repository builds the package.
The file `client/python/Dockerfile.wheel` makes an image that holds two things:

- the wheel file, in `/wheels`,
- the UCX library, in `/opt/ucx`.

The image uses UCX 1.20, built from source.
The UCX version in Ubuntu 22.04 is too old for the client code.
The wheel is for Python 3.11.

The agent Dockerfile reads this image as a named build context with the name `ddlwheel`.
It copies the wheel and UCX into the agent image.
It then runs `import ddl_client` during the build.
The build fails when the import fails.

## 11 DDL3 target

### 11.1 Parts

The program is `ddl_minifs_target_node`.
It has a DDL3 server and a MiniFS storage engine.
The server receives PUT and GET requests.
MiniFS writes the data and its metadata to a device.

| Option | Meaning |
|---|---|
| `--devices` | Device files or disks. One storage core serves each device. |
| `--metadata` | The directory for the MiniFS metadata. |
| `--device-size-gb` | Creates and formats a device file of this size. |
| `--bind`, `--port` | The address and port to listen on. The default port is 58000. |
| `--max-object-mb` | The largest object that the target accepts. |
| `--max-inflight` | The number of requests that the target processes at the same time. |

The target supports two operations: PUT (opcode 1) and GET (opcode 2).
It has no delete, list, or size-check operation.

CAUTION
Data written to the target stays there.
No client function can delete it.
A full device stops all writes.

A request has a tenant number and a 128-bit object ID.
The target stores the object at the path `/ddl/t<tenant>/d<high>/o<low>`.

### 11.2 Message modes

The sender picks the mode from the size of the payload.

| Mode | Payload size | How the data moves |
|---|---|---|
| SM, small | 8,192 bytes or less | The data travels in the request message. |
| MM, medium | up to 262,144 bytes | The sender writes the data into a ring on the receiver. Then it sends a short notice. |
| LM, large | more than 262,144 bytes | The sender sends an address. The receiver reads the data from the sender memory. |

A GET response uses the same three modes in the other direction.

### 11.3 Memory and slabs

RDMA and UCX can use only memory that the program registers in advance.
Registration is slow.
Each side therefore registers one large block at start and cuts it into slabs.

A slab holds one payload while the payload is in transit.
A client copies each payload into a slab before it sends the payload.
A payload that is larger than a slab fails with a memory error.
The slab size must therefore be at least as large as the largest object.

The two sides set the slab size separately.
The messages between them do not carry the slab size.

| Side | Slab size | Slab count | Other registered memory |
|---|---|---|---|
| Target | `--max-object-mb` | 2 times `--max-inflight`, plus 16, plus 16 standby | 64 MiB ring, one buffer, and read buffers of `--max-inflight` times the slab size |
| Client (`ddl_client`) | the chunk size | 16, plus 8 standby | 8 MiB ring and one buffer |

The client registers about 30 times the chunk size for each connection.
A measured value is 121 MiB at a 4 MiB chunk size and 821 MiB at 32 MiB.
The client touches all of this memory when it connects.

### 11.4 Connections

A connection starts with a handshake.
Each side sends a HELLO message with its buffer sizes, its ring location, and its memory key.
The target stores the connection in a table with 64 entries.
The table key is the connection ID.

A second connection with the same ID as an active connection does not complete its handshake.
The ring of the target has 64 lanes.
Each client uses one lane.
The target therefore serves at most 64 clients.

### 11.5 Limits

| Limit | Value |
|---|---|
| Largest object | The `--max-object-mb` value. A stored object is the payload plus 8 bytes. |
| Write status when the object is too large | 3 (no memory). |
| Write status when the target cannot write | 6 (I/O error). |
| Status when the connection is lost | 8 (disconnected). |
| Status when the object does not exist | 1 (not found). |
| Default request time limit | 30 seconds. |

## 12 Data flows

### 12.1 Run a job

1. The user fills in the job form and selects Start Training.
2. The browser sends `POST /jobs`. The backend checks the request and returns a job ID.
3. The browser sends `POST /jobs/{id}/start`.
4. The backend builds the flags and sends them to the agent.
5. The agent starts the training script.
6. The script writes events. The backend reads them and updates the state.
7. The backend sends each event to the browser.
8. The script exports the adapter and writes `job_complete`.
9. The user downloads the adapter with `GET /jobs/{id}/artifact`.

### 12.2 Write a checkpoint to the target

1. The script calls the checkpoint manager at the end of an epoch.
2. The manager makes chunks and packs.
3. The parallel writer gives each pack to one shard thread.
4. The shard adds the length frame and calculates the object ID.
5. The shard sends the request to its connection thread.
6. The `ddl_client` module copies the payload into a slab and sends it.
7. The target writes the object to MiniFS and replies.
8. The manager reads each object back and checks the hash.
9. The manager writes the manifest and then the catalog.

### 12.3 Resume a job

1. The user selects a finished job in the "Resume from a previous run" control.
2. The backend checks the request. See section 3.7.
3. The backend starts the new job with `--run-id <source job ID>` and `--resume <checkpoint>`.
4. The script reads the catalog and the manifest from the storage.
5. The script restores the weights, the optimizer, the scheduler, and the random-number state.
6. The script trains only the epochs that remain.

### 12.4 Inspect a checkpoint

1. Find the job ID. It is the name of the directory in `runs/`.
2. Find the tenant number in the settings.
3. Run the inspector command from section 7.9.
4. Read the summary. Add `--verify` to check every chunk.

## 13 Deployment

### 13.1 Images

| Image | Base | Content |
|---|---|---|
| `pipeline-frontend` | node 22 build, nginx 1.27 | The built web application. |
| `pipeline-backend` | python 3.11 slim | The backend and the shared packages. |
| `pipeline-fine-tuning-agent` | NVIDIA CUDA 12.4 on Ubuntu 22.04 | Python 3.11, torch 2.6.0, the pipeline, `ddl_client`, and UCX. |

The image name has the form `<registry>/<name>:<tag>`.
The variables `REGISTRY` and `IMAGE_TAG` set the registry and the tag.

### 13.2 Compose files

| File | Task |
|---|---|
| `docker-compose.yml` | Defines the three services, the shared volume, and the environment. |
| `docker-compose.cpu.yml` | Profile for a machine without a GPU. |
| `docker-compose.gpu.yml` | Profile that reserves one NVIDIA GPU for the agent. |
| `docker-compose.dev.yml` | Mounts the source code into the containers. No image build is necessary after a code change. |

### 13.3 Ports, volume, and limits

| Item | Value |
|---|---|
| Web application | host port 5173 |
| Backend | host port 8000 |
| Training agent | host port 8001 |
| Shared volume | `nebula-shared`, mounted at `/mnt/nebula-shared` |
| Agent memory limit | `AGENT_MEM_LIMIT`, default 5 GiB, with swap included |

CAUTION
Do not stop the agent container while a user starts a job.
The backend cannot resolve the name `training-service` when the container stops.
The job then fails at once with the message "Name or service not known".

### 13.4 Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `NEBULA_DATA_ROOT` | backend, agent | The directory for models, datasets, and `nebula.db`. |
| `NEBULA_RUNS_ROOT` | backend, agent | The directory for the job directories. |
| `NEBULA_TRAINING_BACKEND` | backend | `subprocess` or `http`. |
| `TRAINING_SERVICE_URL` | backend | The URL of the agent. Required for `http`. |
| `NEBULA_CORS_ORIGINS` | backend | The origins that the browser can use. |
| `UCX_TLS` | agent | The UCX transports. The compose setup uses `tcp`. |
| `VITE_API_BASE` | web build | The backend URL that the browser uses. |

### 13.5 Build the `ddl_client` image

Do this procedure in the NebulaR repository.

1. Go to the root of the repository.
2. Run `docker build -f client/python/Dockerfile.wheel -t nebula-ddl-client:0.1.0 .`
3. Wait for the build to end. The first build compiles UCX and takes several minutes.

### 13.6 Start the system

Do this procedure in the `ai-accelerator-platform` repository.

1. Make sure the image `nebula-ddl-client:0.1.0` exists.
2. Run `docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --build`.
3. Open `http://localhost:5173`.
4. Open Settings. Enter the target address and tenant.
5. In the job form, select DDL as the checkpoint storage, if the job must use the target.

A code-only rebuild does not download the dependencies again.
The reason is that the dependency layer depends only on `pyproject.toml`.

### 13.7 Work on the code without a rebuild

1. Add `-f docker-compose.dev.yml` to the compose command.
2. Run `up -d`.
3. Change the code.

The backend and the agent service reload when the code changes.
A training job starts a new process.
It therefore uses the new code at the next job.

A rebuild is necessary only after a change to a dependency list, a Dockerfile, or the `ddl_client` image.

## 14 Failure behavior

| Condition | Effect | Action |
|---|---|---|
| The agent container is stopped. | The job fails at launch with "Name or service not known". | Start the container. Submit the job again. |
| The job exceeds the memory limit. | The kernel kills the process. The job fails with exit code -9 and no final event. | Use a smaller job, or increase `AGENT_MEM_LIMIT`. |
| The target is not reachable when the job starts. | The DDL connection fails after the time limit (30 seconds). | Check the address, the port, and the network. |
| The target loses the connection during a write. | The write returns status 8. The manager raises an error. No manifest or catalog entry exists. | Start the target. Run the job again. |
| An object is larger than `--max-object-mb`. | The write returns status 3. No checkpoint is committed. | Reduce the chunk size, or increase `--max-object-mb`. |
| The target cannot write. | Each write returns status 6. A full device is the likely cause. | Add space or use a new device. |
| A stored object is damaged. | `verify` reports a checksum error. A load raises an error. | Use an earlier checkpoint. |
| The backend stops while a job runs. | The job record can stay in the state RUNNING. | Confirm that no process runs. Correct the record in the database. The API cannot delete a job in this state. |

## 15 Known limits

- The Checkpoints and Deployments pages are placeholders.
- The DDL3 target has no delete, list, or size-check operation.
- The `ddl_client` module sends one request at a time on a connection.
- The backend does not correct a job record that stays in RUNNING after a restart.
- A checkpoint from before the epoch correction repeats its last epoch when it resumes.
- All speed values in this document come from one test setup. That setup connected through a network path of about 360 Mbit/s.
- On that path, the target gave 30 to 44 MB/s. Local disk was faster, because it did not use the network.
