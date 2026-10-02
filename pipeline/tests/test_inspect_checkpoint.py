import json

import numpy as np
import pytest

from pipeline.checkpoint import inspect_checkpoint as ic
from pipeline.checkpoint.localfs_backend import LocalFsBackend
from pipeline.checkpoint.manager import BlobStoreCheckpointManager


def _save(mgr, cid="c1"):
    rng = np.random.default_rng(0)
    model = {f"layer{i}.lora_A.weight": rng.standard_normal((8, 96)).astype("float32") for i in range(30)}
    model["big.weight"] = rng.standard_normal(300_000).astype("float32")
    opt = {"exp_avg": rng.standard_normal((8, 96)).astype("float32")}
    manifest, _ = mgr.save_checkpoint(
        cid, model, opt, None, {}, epoch=1, global_step=18, model_name_or_path="Qwen/x",
        base_model_id="b", dataset_id="d", lora_config={"r": 8}, training_config={},
    )
    return manifest


@pytest.fixture
def stored(tmp_path):
    root = tmp_path / "checkpoints"
    mgr = BlobStoreCheckpointManager("run1", storage=LocalFsBackend(str(root)), num_workers=2,
                                     chunk_size_bytes=512 * 1024)
    manifest = _save(mgr)
    mgr.shutdown()
    return root, manifest


def test_describe_counts_tensors_chunks_and_stored_objects(stored):
    _, manifest = stored
    info = ic.describe_checkpoint(manifest)
    assert info["groups"]["model"]["tensors"] == 31
    assert info["groups"]["optimizer"]["tensors"] == 1
    assert info["total_chunks"] == sum(1 for _ in manifest.all_chunks())
    # packing: far fewer stored objects than chunk records
    assert len(info["stored_objects"]) < info["total_chunks"] / 4
    packed = [b for b in info["stored_objects"] if b["packed"]]
    assert packed and all(b["payload_bytes"] <= b["bytes"] for b in packed)
    assert info["payload_bytes"] == sum(t.byte_size for _, a in ic._GROUPS for t in getattr(manifest, a))
    json.dumps(info)  # must be JSON-serializable


def test_tensor_rows_filter_and_location(stored):
    _, manifest = stored
    rows = ic.tensor_rows(manifest, grep="layer1")
    assert rows and all("layer1" in r["name"] for r in rows)
    small = next(r for r in ic.tensor_rows(manifest) if r["name"] == "layer0.lora_A.weight")
    assert small["blob_offset"] is not None and "/pack_" in small["blob_id"]
    big = next(r for r in ic.tensor_rows(manifest) if r["name"] == "big.weight")
    assert big["chunks"] > 1


def test_renderers_mention_the_essentials(stored):
    _, manifest = stored
    info = ic.describe_checkpoint(manifest)
    text = ic.render_summary(info)
    for needle in ("checkpoint c1", "COMPLETE", "global_step 18", "Qwen/x", "stored as"):
        assert needle in text
    assert "pack_" in ic.render_blobs(info)
    assert "layer0.lora_A.weight" in ic.render_tensors(ic.tensor_rows(manifest))


def test_cli_end_to_end_on_a_local_checkpoint(stored, capsys):
    root, _ = stored
    rc = ic.main(["--run-id", "run1", "--local-dir", str(root), "--blobs", "--tensors", "--grep", "layer2", "--verify"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "1 checkpoint(s) in the catalog" in out and "verify: ok=True" in out
    assert "layer2" in out and "layer1." not in out.split("tensor(s):")[1].split("verify:")[0].replace("layer10", "")


def test_cli_reports_a_missing_run(stored, capsys):
    root, _ = stored
    rc = ic.main(["--run-id", "nope", "--local-dir", str(root)])
    assert rc == 1 and "no checkpoints found" in capsys.readouterr().out


def test_cli_verify_fails_on_corruption(stored, capsys):
    root, manifest = stored
    pack = next(c for c in manifest.all_chunks() if c.packed)
    victim = next(root.rglob(pack.blob_id.rsplit("/", 1)[-1]))
    data = bytearray(victim.read_bytes()); data[len(data) // 2] ^= 0xFF; victim.write_bytes(bytes(data))
    rc = ic.main(["--run-id", "run1", "--local-dir", str(root), "--verify"])
    out = capsys.readouterr().out
    assert rc == 2 and "ISSUE checksum_mismatch" in out
