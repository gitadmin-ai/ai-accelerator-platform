from pipeline.checkpoint.format import (
    CHECKPOINT_FORMAT_VERSION,
    STATUS_COMPLETE,
    ChunkRecord,
    Manifest,
    ObjectRecord,
    TensorRecord,
    catalog_blob_id,
    manifest_blob_id,
)


def _sample_manifest() -> Manifest:
    chunk = ChunkRecord(index=0, blob_id="ckpt/x/model/w/chunk_000000", shard=0, offset=0, length=16, sha256="a" * 64)
    tensor = TensorRecord(
        name="w", dtype="torch.float32", shape=[4], numel=4, byte_size=16, chunk_size=1024, chunks=[chunk]
    )
    obj = ObjectRecord(name="training_state", blob_id="ckpt/x/training_state/state.json", shard=0, length=10, sha256="b" * 64)
    return Manifest(
        checkpoint_id="epoch-1-step-10",
        format_version=CHECKPOINT_FORMAT_VERSION,
        status=STATUS_COMPLETE,
        epoch=1,
        global_step=10,
        best_metric=0.5,
        model_name_or_path="/models/qwen",
        base_model_id="Qwen/Qwen2.5-0.5B",
        dataset_id="onboarding-tutor-v1",
        lora_config={"r": 8, "lora_alpha": 16},
        training_config={"learning_rate": 2e-4},
        num_shards=4,
        chunk_size_bytes=1024,
        timestamp=1234.5,
        model_tensors=[tensor],
        optimizer_tensors=[],
        optimizer_skeleton_blob=None,
        scheduler_state=None,
        training_state=obj,
        training_state_tensors=[],
    )


def test_manifest_json_round_trip():
    m = _sample_manifest()
    data = m.to_json()
    restored = Manifest.from_json(data)
    assert restored.checkpoint_id == m.checkpoint_id
    assert restored.epoch == m.epoch
    assert restored.lora_config == m.lora_config
    assert len(restored.model_tensors) == 1
    assert restored.model_tensors[0].chunks[0].sha256 == "a" * 64
    assert restored.training_state.blob_id == "ckpt/x/training_state/state.json"


def test_manifest_all_chunks_and_objects():
    m = _sample_manifest()
    chunks = list(m.all_chunks())
    objects = list(m.all_objects())
    assert len(chunks) == 1
    assert len(objects) == 1
    assert objects[0] is m.training_state


def test_manifest_blob_id_naming():
    assert manifest_blob_id("epoch-1-step-10") == "ckpt/epoch-1-step-10/manifest.json"
    assert catalog_blob_id("run-abc") == "ckpt/run-abc/catalog.json"


def test_manifest_json_is_human_readable_text_not_pickle():
    m = _sample_manifest()
    data = m.to_json()
    # must be valid UTF-8 JSON, not a pickle stream
    text = data.decode("utf-8")
    assert text.strip().startswith("{")
    assert "checkpoint_id" in text
