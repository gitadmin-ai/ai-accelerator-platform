"""BlobStoreCheckpointManager -- the integration point between PyTorch
training state and the BlobStore-backed chunked/parallel checkpoint format.

Commit protocol (see training/README.md for the full design writeup):
  1. generate/accept checkpoint_id
  2-3. describe+flatten+chunk+hash model & optimizer tensors, submit every
       chunk to the N-shard ParallelWriter in one fan-out (this is the
       actual "parallel BlobStore writers" the project exists to
       demonstrate)
  4-5. write scheduler + training-state skeleton objects (small, unchunked,
       written directly on the coordinator shard)
  6. validate all writes (head_blob size check per chunk -- BlobStore's own
     putBlob() cannot report failure, see blobstore_bindings.cpp)
  7. write the manifest (single blob, single atomic write)
  8. update the catalog last -- this is the actual commit point:
     get_latest_checkpoint()/list_checkpoints() only ever consult the
     catalog, so a crash between 7 and 8 leaves a fully-written, checksummed
     checkpoint that is simply not yet discoverable (load_checkpoint(id)
     still works on it directly; see docstring there)
If step 2-6 fails for any chunk, save_checkpoint raises
CheckpointWriteError and neither the manifest nor the catalog is written
-- there is no way to observe a partially-written checkpoint as successful.
"""
from __future__ import annotations

import dataclasses
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from pipeline.checkpoint import state_flatten, tensor_io, validation
from pipeline.checkpoint.blobstore_backend import BlobStoreBackend
from pipeline.checkpoint.chunker import iter_chunks
from pipeline.checkpoint.exceptions import (
    CheckpointCorruptError,
    CheckpointIncompleteError,
    CheckpointNotFoundError,
    CheckpointWriteError,
)
from pipeline.checkpoint.format import (
    CHECKPOINT_FORMAT_VERSION,
    STATUS_COMPLETE,
    ChunkRecord,
    Manifest,
    ObjectRecord,
    TensorRecord,
    catalog_blob_id,
    manifest_blob_id,
    now_timestamp,
)
from pipeline.checkpoint.parallel_writer import ParallelWriter, VerifyJob, WriteJob
from pipeline.checkpoint.serializer import numel_of, sha256_hex
from pipeline.checkpoint.storage_backend import StorageBackend, StorageShard
from pipeline.utils.metrics import CheckpointMetrics, Timing

_COORDINATOR_SHARD = 0  # owns manifest / catalog / scheduler / skeleton blobs


def _sanitize(name: str) -> str:
    return name.replace("/", "__")


@dataclasses.dataclass
class LoadedCheckpoint:
    manifest: Manifest
    model_tensors_raw: Dict[str, Tuple[str, List[int], bytes]]
    full_model_tensors_raw: Dict[str, Tuple[str, List[int], bytes]]
    optimizer_skeleton: Optional[dict]
    optimizer_tensors_raw: Dict[str, Tuple[str, List[int], bytes]]
    scheduler_state: Optional[dict]
    training_state_skeleton: Optional[dict]
    training_state_tensors_raw: Dict[str, Tuple[str, List[int], bytes]]


class BlobStoreCheckpointManager:
    def __init__(
        self,
        run_id: str,
        storage: Optional[StorageBackend] = None,
        num_workers: int = 4,
        chunk_size_bytes: int = 256 * 1024 * 1024,
        queue_maxsize_per_shard: int = 8,
        verify_mode: str = "checksum",
        config_overrides: Optional[Dict[str, Any]] = None,
    ):
        """`storage` selects where checkpoint chunks are persisted -- an
        instance of a StorageBackend (see checkpoint/storage_backend.py),
        e.g. BlobStoreBackend() (the default, DDL-backed) or
        LocalFsBackend(path) (plain files on a local filesystem). Everything
        else in the pipeline (chunking, manifest, catalog, validation) is
        backend-agnostic and unaffected by this choice.

        `config_overrides` only applies to the default BlobStoreBackend and
        is kept for backward compatibility with callers that predate
        `storage`; pass a pre-configured BlobStoreBackend(**overrides)
        instead if you're also passing `storage`.
        """
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        if verify_mode not in ("checksum", "head", "none"):
            raise ValueError("verify_mode must be 'checksum', 'head', or 'none'")
        if storage is not None and config_overrides:
            raise ValueError(
                "config_overrides only applies to the default BlobStore backend; "
                "configure a custom `storage` backend directly instead of combining "
                "it with config_overrides"
            )
        self.run_id = run_id
        self.chunk_size_bytes = chunk_size_bytes
        self.verify_mode = verify_mode
        storage = storage if storage is not None else BlobStoreBackend(**(config_overrides or {}))
        self.shards: List[StorageShard] = storage.create_shards(num_workers)
        self.writer = ParallelWriter(self.shards, queue_maxsize_per_shard=queue_maxsize_per_shard)
        self._shard_rr = 0

    @property
    def num_workers(self) -> int:
        return len(self.shards)

    def shutdown(self) -> None:
        self.writer.shutdown()

    def _next_shard(self) -> int:
        s = self._shard_rr % self.num_workers
        self._shard_rr += 1
        return s

    # ------------------------------------------------------------------
    # save
    # ------------------------------------------------------------------

    def _plan_chunked_group(
        self, items: Dict[str, Any], group: str, checkpoint_id: str, timing: Timing
    ) -> Tuple[List[WriteJob], List[TensorRecord]]:
        jobs: List[WriteJob] = []
        records: List[TensorRecord] = []
        for name, value in items.items():
            dtype_str, shape, buf = tensor_io.describe_tensor(value, timing=timing)
            byte_size = len(buf)
            chunks: List[ChunkRecord] = []
            for cs in iter_chunks(buf, self.chunk_size_bytes):
                shard = self._next_shard()
                blob_id = f"ckpt/{checkpoint_id}/{group}/{_sanitize(name)}/chunk_{cs.index:06d}"
                digest = sha256_hex(cs.data)
                chunks.append(
                    ChunkRecord(
                        index=cs.index,
                        blob_id=blob_id,
                        shard=shard,
                        offset=cs.offset,
                        length=cs.length,
                        sha256=digest,
                    )
                )
                jobs.append(WriteJob(shard=shard, blob_id=blob_id, data=cs.data, tag=blob_id))
            records.append(
                TensorRecord(
                    name=name,
                    dtype=dtype_str,
                    shape=shape,
                    numel=numel_of(shape),
                    byte_size=byte_size,
                    chunk_size=self.chunk_size_bytes,
                    chunks=chunks,
                )
            )
        return jobs, records

    def _put_json_object(self, blob_id: str, obj: Any) -> ObjectRecord:
        data = json.dumps(obj, indent=2).encode("utf-8")
        digest = sha256_hex(data)
        self.shards[_COORDINATOR_SHARD].put(blob_id, data)
        return ObjectRecord(
            name=blob_id, blob_id=blob_id, shard=_COORDINATOR_SHARD, length=len(data), sha256=digest
        )

    def _get_json_object(self, obj: ObjectRecord) -> Any:
        shard = self.shards[obj.shard % self.num_workers]
        data = bytes(shard.get(obj.blob_id, obj.length))
        if sha256_hex(data) != obj.sha256:
            raise CheckpointCorruptError(f"checksum mismatch reading {obj.blob_id}")
        return json.loads(data.decode("utf-8"))

    def save_checkpoint(
        self,
        checkpoint_id: str,
        model_state: Dict[str, Any],
        optimizer_state: Optional[Dict[str, Any]],
        scheduler_state: Optional[Dict[str, Any]],
        training_state: Dict[str, Any],
        *,
        epoch: int,
        global_step: int,
        model_name_or_path: str,
        base_model_id: str,
        dataset_id: str,
        lora_config: Dict[str, Any],
        training_config: Dict[str, Any],
        best_metric: Optional[float] = None,
        full_model_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Manifest, CheckpointMetrics]:
        timing = Timing()
        t_start = time.perf_counter()
        self._shard_rr = 0

        with timing.measure("chunking_s"):
            model_jobs, model_records = self._plan_chunked_group(
                model_state, "model", checkpoint_id, timing
            )

        full_model_jobs: List[WriteJob] = []
        full_model_records: List[TensorRecord] = []
        if full_model_state:
            with timing.measure("chunking_s"):
                full_model_jobs, full_model_records = self._plan_chunked_group(
                    full_model_state, "full_model", checkpoint_id, timing
                )

        opt_skeleton = None
        opt_jobs: List[WriteJob] = []
        opt_records: List[TensorRecord] = []
        if optimizer_state is not None:
            opt_skeleton, opt_tensors_flat = state_flatten.flatten_state_dict(
                optimizer_state, tensor_io.is_tensor_like, prefix="optimizer"
            )
            with timing.measure("chunking_s"):
                opt_jobs, opt_records = self._plan_chunked_group(
                    opt_tensors_flat, "optimizer", checkpoint_id, timing
                )

        sched_skeleton = None
        if scheduler_state is not None:
            sched_skeleton, sched_tensors_flat = state_flatten.flatten_state_dict(
                scheduler_state, tensor_io.is_tensor_like, prefix="scheduler"
            )
            if sched_tensors_flat:
                raise CheckpointWriteError(
                    "scheduler.state_dict() contains tensor leaves "
                    f"({list(sched_tensors_flat)}); this checkpoint format does not store "
                    "tensors for the scheduler group (standard torch/HF LR schedulers never "
                    "have any, so this input is treated as unsupported)."
                )

        ts_skeleton, ts_tensors_flat = state_flatten.flatten_state_dict(
            training_state, tensor_io.is_tensor_like, prefix="training_state"
        )
        ts_jobs: List[WriteJob] = []
        ts_records: List[TensorRecord] = []
        if ts_tensors_flat:
            with timing.measure("chunking_s"):
                ts_jobs, ts_records = self._plan_chunked_group(
                    ts_tensors_flat, "training_state", checkpoint_id, timing
                )

        all_jobs = model_jobs + full_model_jobs + opt_jobs + ts_jobs

        with timing.measure("write_s"):
            for job in all_jobs:
                self.writer.submit(job)
            results = self.writer.collect(len(all_jobs)) if all_jobs else []

            failed = [r for r in results if not r.ok]

            # Inline post-write validation -- also timed as part of the
            # write path, it's real I/O, not free. BlobStore's putBlob()
            # can't report failure on its own, and a head-only check can't
            # catch a write that silently landed on a failed device (see
            # validation.py's module docstring), so "checksum" (a full
            # read-back + SHA-256 compare) is the default despite costing
            # extra I/O. That I/O is fanned out across the same N-shard
            # worker pool as the writes themselves (VerifyJob, not a plain
            # Python loop) -- an earlier version of this ran the checksum
            # re-reads serially on the caller's thread, which independently
            # of worker count added O(checkpoint size) wall-clock time and
            # completely masked the parallel writers' speedup in
            # benchmark_checkpoint.py.
            validation_issues = []
            if self.verify_mode == "checksum":
                all_chunks = [
                    c for rec_list in (model_records, full_model_records, opt_records, ts_records)
                    for rec in rec_list for c in rec.chunks
                ]
                for chunk in all_chunks:
                    self.writer.submit(
                        VerifyJob(
                            shard=chunk.shard,
                            blob_id=chunk.blob_id,
                            length=chunk.length,
                            expected_sha256=chunk.sha256,
                            tag=chunk.blob_id,
                        )
                    )
                for r in self.writer.collect(len(all_chunks)) if all_chunks else []:
                    if not r.ok:
                        validation_issues.append(
                            validation.VerifyIssue("checksum_mismatch", r.job.blob_id, r.error or "")
                        )
            elif self.verify_mode == "head":
                for rec_list in (model_records, full_model_records, opt_records, ts_records):
                    for rec in rec_list:
                        for chunk in rec.chunks:
                            issue = validation.check_chunk_head(self.shards, chunk)
                            if issue:
                                validation_issues.append(issue)

        if failed or validation_issues:
            detail = "; ".join(
                [f"{r.job.blob_id}: {r.error}" for r in failed]
                + [f"{i.name}: {i.kind} ({i.detail})" for i in validation_issues]
            )
            raise CheckpointWriteError(
                f"checkpoint {checkpoint_id!r} write failed for "
                f"{len(failed) + len(validation_issues)} chunk(s); checkpoint NOT committed "
                f"(no manifest/catalog entry was written): {detail}"
            )

        scheduler_obj = None
        if sched_skeleton is not None:
            scheduler_obj = self._put_json_object(
                f"ckpt/{checkpoint_id}/scheduler/state.json", sched_skeleton
            )
        training_state_obj = self._put_json_object(
            f"ckpt/{checkpoint_id}/training_state/state.json", ts_skeleton
        )
        optimizer_skeleton_obj = None
        if opt_skeleton is not None:
            optimizer_skeleton_obj = self._put_json_object(
                f"ckpt/{checkpoint_id}/optimizer/skeleton.json", opt_skeleton
            )

        with timing.measure("commit_s"):
            manifest = Manifest(
                checkpoint_id=checkpoint_id,
                format_version=CHECKPOINT_FORMAT_VERSION,
                status=STATUS_COMPLETE,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
                model_name_or_path=model_name_or_path,
                base_model_id=base_model_id,
                dataset_id=dataset_id,
                lora_config=lora_config,
                training_config=training_config,
                num_shards=self.num_workers,
                chunk_size_bytes=self.chunk_size_bytes,
                timestamp=now_timestamp(),
                model_tensors=model_records,
                optimizer_tensors=opt_records,
                optimizer_skeleton_blob=optimizer_skeleton_obj,
                scheduler_state=scheduler_obj,
                training_state=training_state_obj,
                training_state_tensors=ts_records,
                full_model_tensors=full_model_records,
            )
            self.shards[_COORDINATOR_SHARD].put(manifest_blob_id(checkpoint_id), manifest.to_json())
            self._update_catalog(checkpoint_id, STATUS_COMPLETE, epoch, global_step, manifest.timestamp)

        total_s = time.perf_counter() - t_start
        size_bytes = sum(c.length for c in manifest.all_chunks()) + sum(
            o.length for o in manifest.all_objects()
        )
        metrics = CheckpointMetrics(
            checkpoint_id=checkpoint_id,
            size_bytes=size_bytes,
            num_tensors=len(model_records) + len(opt_records) + len(ts_records) + len(full_model_records),
            num_chunks=sum(1 for _ in manifest.all_chunks()),
            chunk_size_bytes=self.chunk_size_bytes,
            num_workers=self.num_workers,
            gpu_to_cpu_s=timing.gpu_to_cpu_s,
            chunking_s=timing.chunking_s,
            write_s=timing.write_s,
            commit_s=timing.commit_s,
            total_s=total_s,
        )
        return manifest, metrics

    # ------------------------------------------------------------------
    # catalog / discovery
    # ------------------------------------------------------------------

    def _read_catalog(self) -> List[dict]:
        shard = self.shards[_COORDINATOR_SHARD]
        blob_id = catalog_blob_id(self.run_id)
        if not shard.exists(blob_id):
            return []
        info = shard.head(blob_id)
        data = bytes(shard.get(blob_id, info["size"]))
        return json.loads(data.decode("utf-8"))

    def _update_catalog(
        self, checkpoint_id: str, status: str, epoch: int, global_step: int, timestamp: float
    ) -> None:
        catalog = [e for e in self._read_catalog() if e["checkpoint_id"] != checkpoint_id]
        catalog.append(
            dict(
                checkpoint_id=checkpoint_id,
                status=status,
                epoch=epoch,
                global_step=global_step,
                timestamp=timestamp,
            )
        )
        data = json.dumps(catalog, indent=2).encode("utf-8")
        self.shards[_COORDINATOR_SHARD].put(catalog_blob_id(self.run_id), data)

    def list_checkpoints(self, include_incomplete: bool = False) -> List[dict]:
        catalog = self._read_catalog()
        if include_incomplete:
            return catalog
        return [e for e in catalog if e["status"] == STATUS_COMPLETE]

    def get_latest_checkpoint(self) -> Optional[str]:
        entries = self.list_checkpoints(include_incomplete=False)
        if not entries:
            return None
        best = max(entries, key=lambda e: (e["epoch"], e["global_step"], e["timestamp"]))
        return best["checkpoint_id"]

    # ------------------------------------------------------------------
    # load
    # ------------------------------------------------------------------

    def _read_manifest(self, checkpoint_id: str) -> Manifest:
        shard = self.shards[_COORDINATOR_SHARD]
        blob_id = manifest_blob_id(checkpoint_id)
        if not shard.exists(blob_id):
            raise CheckpointNotFoundError(f"no manifest for checkpoint {checkpoint_id!r}")
        info = shard.head(blob_id)
        data = bytes(shard.get(blob_id, info["size"]))
        return Manifest.from_json(data)

    def _read_tensor_group(
        self, records: List[TensorRecord]
    ) -> Dict[str, Tuple[str, List[int], bytes]]:
        out: Dict[str, Tuple[str, List[int], bytes]] = {}
        for rec in records:
            buf = bytearray(rec.byte_size)
            for chunk in rec.chunks:
                shard = self.shards[chunk.shard % self.num_workers]
                data = shard.get(chunk.blob_id, chunk.length)
                if sha256_hex(data) != chunk.sha256:
                    raise CheckpointCorruptError(f"checksum mismatch reading {chunk.blob_id}")
                buf[chunk.offset : chunk.offset + chunk.length] = data
            out[rec.name] = (rec.dtype, rec.shape, bytes(buf))
        return out

    def load_checkpoint(self, checkpoint_id: str, allow_incomplete: bool = False) -> LoadedCheckpoint:
        """Loads by explicit checkpoint_id, bypassing the catalog -- this is
        the "unless explicitly requested" escape hatch for an incomplete
        checkpoint the brief calls for; get_latest_checkpoint()/
        list_checkpoints() never surface one on their own.
        """
        manifest = self._read_manifest(checkpoint_id)
        if manifest.status != STATUS_COMPLETE and not allow_incomplete:
            raise CheckpointIncompleteError(
                f"checkpoint {checkpoint_id!r} has status {manifest.status!r}; "
                "pass allow_incomplete=True to load it anyway"
            )

        model_raw = self._read_tensor_group(manifest.model_tensors)
        full_model_raw = self._read_tensor_group(manifest.full_model_tensors)
        opt_raw = self._read_tensor_group(manifest.optimizer_tensors)
        ts_raw = self._read_tensor_group(manifest.training_state_tensors)

        opt_skeleton = (
            self._get_json_object(manifest.optimizer_skeleton_blob)
            if manifest.optimizer_skeleton_blob
            else None
        )
        sched_state = (
            self._get_json_object(manifest.scheduler_state) if manifest.scheduler_state else None
        )
        ts_skeleton = (
            self._get_json_object(manifest.training_state) if manifest.training_state else None
        )

        return LoadedCheckpoint(
            manifest=manifest,
            model_tensors_raw=model_raw,
            full_model_tensors_raw=full_model_raw,
            optimizer_skeleton=opt_skeleton,
            optimizer_tensors_raw=opt_raw,
            scheduler_state=sched_state,
            training_state_skeleton=ts_skeleton,
            training_state_tensors_raw=ts_raw,
        )

    def verify(self, checkpoint_id: str) -> validation.VerifyReport:
        manifest = self._read_manifest(checkpoint_id)
        return validation.verify_deep(self.shards, manifest)

    def materialize_numpy_state(self, loaded: LoadedCheckpoint) -> Dict[str, Any]:
        """Convenience reconstruction used by the numpy-based test suite
        (and any consumer that doesn't need torch): tensors come back as
        numpy arrays and optimizer/training_state are unflattened back into
        plain Python structures. Real PyTorch training code should
        reconstruct via pipeline.checkpoint.tensor_adapter directly instead
        (needed for dtypes numpy can't represent, e.g. bfloat16) -- see
        train_lora.py's resume path.
        """
        model = {n: tensor_io.reconstruct_tensor(*v) for n, v in loaded.model_tensors_raw.items()}
        full_model = {
            n: tensor_io.reconstruct_tensor(*v) for n, v in loaded.full_model_tensors_raw.items()
        }
        optimizer = None
        if loaded.optimizer_skeleton is not None:
            opt_tensors = {
                n: tensor_io.reconstruct_tensor(*v) for n, v in loaded.optimizer_tensors_raw.items()
            }
            optimizer = state_flatten.unflatten_state_dict(loaded.optimizer_skeleton, opt_tensors)
        training_state = None
        if loaded.training_state_skeleton is not None:
            ts_tensors = {
                n: tensor_io.reconstruct_tensor(*v) for n, v in loaded.training_state_tensors_raw.items()
            }
            training_state = state_flatten.unflatten_state_dict(
                loaded.training_state_skeleton, ts_tensors
            )
        scheduler = None
        if loaded.scheduler_state is not None:
            scheduler = state_flatten.unflatten_state_dict(loaded.scheduler_state, {})

        return dict(
            model=model,
            full_model=full_model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=training_state,
        )
