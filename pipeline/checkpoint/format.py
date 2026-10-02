"""Checkpoint manifest schema.

The manifest is the single JSON document that describes every object/chunk
belonging to one checkpoint. It is the last thing written for a checkpoint's
own contents (see manager.py's commit protocol) and it is what verify()
walks to check integrity.

Deliberately JSON, not pickle: the manifest is metadata (names, shapes,
dtypes, blob ids, checksums), never tensor payload, so there is no reason to
touch an unsafe deserializer for it (constraint #6 in the project brief).
"""
from __future__ import annotations

import dataclasses
import json
import time
from typing import Any, Dict, List, Optional

# 2: ChunkRecord may be one slice of a packed object (blob_offset/blob_length).
# Version-1 manifests have neither field and read back unchanged (blob_length 0
# means "this chunk is the whole stored object").
CHECKPOINT_FORMAT_VERSION = 2
SUPPORTED_FORMAT_VERSIONS = (1, 2)

# Default ceiling for a packed object. Small tensors (LoRA adapters are a few KiB
# each) would otherwise be one stored object apiece, and on a network store the
# per-object round trip dominates the whole checkpoint.
DEFAULT_PACK_SIZE_BYTES = 4 * 1024 * 1024

STATUS_INCOMPLETE = "INCOMPLETE"
STATUS_COMPLETE = "COMPLETE"


@dataclasses.dataclass
class ChunkRecord:
    """One physically-written blob holding a contiguous byte range of a
    tensor's (or other object's) flattened payload.
    """

    index: int
    blob_id: str
    shard: int
    offset: int  # of this chunk within its tensor's payload
    length: int
    sha256: str  # of this chunk's own bytes (not of the whole stored object)
    # Packing. When blob_length > 0 this chunk is the slice
    # [blob_offset, blob_offset + length) of a stored object that is blob_length
    # bytes long and shared with other chunks (see manager._plan_chunked_group).
    # blob_length == 0 (the default, and everything written by format version 1)
    # means the stored object is exactly this chunk.
    blob_offset: int = 0
    blob_length: int = 0

    @property
    def packed(self) -> bool:
        return self.blob_length > 0

    @property
    def stored_length(self) -> int:
        """Length of the stored object this chunk lives in."""
        return self.blob_length if self.packed else self.length

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ChunkRecord":
        return ChunkRecord(**d)


@dataclasses.dataclass
class TensorRecord:
    """Metadata for one tensor, split across one or more ChunkRecords.

    `dtype` is a string (e.g. "torch.float32", "torch.bfloat16") rather than
    a numpy dtype, since numpy cannot represent every torch dtype
    (bfloat16 in particular) — see checkpoint/tensor_adapter.py.
    """

    name: str
    dtype: str
    shape: List[int]
    numel: int
    byte_size: int
    chunk_size: int
    chunks: List[ChunkRecord]

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TensorRecord":
        chunks = [ChunkRecord.from_dict(c) for c in d["chunks"]]
        return TensorRecord(
            name=d["name"],
            dtype=d["dtype"],
            shape=list(d["shape"]),
            numel=d["numel"],
            byte_size=d["byte_size"],
            chunk_size=d["chunk_size"],
            chunks=chunks,
        )


@dataclasses.dataclass
class ObjectRecord:
    """A non-chunked, single-blob object (scheduler state, training state).

    Still checksummed and size-recorded so verify() covers it the same way
    as chunked tensors.
    """

    name: str
    blob_id: str
    shard: int
    length: int
    sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ObjectRecord":
        return ObjectRecord(**d)


@dataclasses.dataclass
class Manifest:
    checkpoint_id: str
    format_version: int
    status: str

    epoch: int
    global_step: int
    best_metric: Optional[float]

    model_name_or_path: str
    base_model_id: str
    dataset_id: str
    lora_config: Dict[str, Any]
    training_config: Dict[str, Any]

    num_shards: int
    chunk_size_bytes: int
    timestamp: float

    model_tensors: List[TensorRecord]
    optimizer_tensors: List[TensorRecord]
    optimizer_skeleton_blob: Optional[ObjectRecord]
    scheduler_state: Optional[ObjectRecord]
    training_state: Optional[ObjectRecord]
    # RNG-state tensors (torch CPU/CUDA generator state, grad scaler tensors)
    # embedded inside training_state's skeleton — routed through the same
    # chunked tensor pipeline as model/optimizer tensors rather than a
    # second ad-hoc inline-bytes representation.
    training_state_tensors: List[TensorRecord] = dataclasses.field(default_factory=list)

    # populated only when --save-full-model was requested
    full_model_tensors: List[TensorRecord] = dataclasses.field(default_factory=list)

    def to_json(self) -> bytes:
        def enc(o):
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            raise TypeError(f"not JSON serializable: {type(o)}")

        return json.dumps(dataclasses.asdict(self), default=enc, indent=2).encode("utf-8")

    @staticmethod
    def from_json(data: bytes) -> "Manifest":
        d = json.loads(data.decode("utf-8"))
        d["model_tensors"] = [TensorRecord.from_dict(t) for t in d["model_tensors"]]
        d["optimizer_tensors"] = [TensorRecord.from_dict(t) for t in d["optimizer_tensors"]]
        d["optimizer_skeleton_blob"] = (
            ObjectRecord.from_dict(d["optimizer_skeleton_blob"])
            if d.get("optimizer_skeleton_blob")
            else None
        )
        d["scheduler_state"] = (
            ObjectRecord.from_dict(d["scheduler_state"]) if d.get("scheduler_state") else None
        )
        d["training_state"] = (
            ObjectRecord.from_dict(d["training_state"]) if d.get("training_state") else None
        )
        d["full_model_tensors"] = [
            TensorRecord.from_dict(t) for t in d.get("full_model_tensors", [])
        ]
        d["training_state_tensors"] = [
            TensorRecord.from_dict(t) for t in d.get("training_state_tensors", [])
        ]
        return Manifest(**d)

    def all_chunks(self):
        """Yield every ChunkRecord in the checkpoint (model + optimizer + training state)."""
        for t in self.model_tensors:
            yield from t.chunks
        for t in self.optimizer_tensors:
            yield from t.chunks
        for t in self.full_model_tensors:
            yield from t.chunks
        for t in self.training_state_tensors:
            yield from t.chunks

    def all_objects(self):
        for o in (self.optimizer_skeleton_blob, self.scheduler_state, self.training_state):
            if o is not None:
                yield o


def manifest_blob_id(checkpoint_id: str) -> str:
    return f"ckpt/{checkpoint_id}/manifest.json"


def catalog_blob_id(run_id: str) -> str:
    return f"ckpt/{run_id}/catalog.json"


def now_timestamp() -> float:
    return time.time()
