"""Read-only inspection of a stored checkpoint: what is in it, where each chunk
lives, and whether it is intact.

A DDL3 target has no listing operation and stores objects under hashed ids, so you
cannot browse a checkpoint with `ls`. Everything is reachable by name, though:
run id -> catalog -> manifest -> chunk records. This tool follows that chain.

    python -m pipeline.checkpoint.inspect_checkpoint --run-id <job id> \\
        --ddl-server 192.168.11.87 --ddl-tenant 7 [--checkpoint-id ID] \\
        [--blobs] [--tensors [--grep NAME]] [--manifest] [--verify]

For checkpoints on a local disk use --local-dir <run dir>/checkpoints instead of the
--ddl-* flags. Nothing is written to the store.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from pipeline.checkpoint.format import Manifest
from pipeline.checkpoint.manager import BlobStoreCheckpointManager

_GROUPS = (
    ("model", "model_tensors"),
    ("optimizer", "optimizer_tensors"),
    ("full_model", "full_model_tensors"),
    ("training_state", "training_state_tensors"),
)


def describe_checkpoint(manifest: Manifest) -> Dict[str, Any]:
    """Summary of a manifest: tensor counts and bytes per group, and the stored
    objects (a pack shared by many chunks counts once) with their layout."""
    groups: Dict[str, Dict[str, int]] = OrderedDict()
    blobs: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for name, attr in _GROUPS:
        tensors = getattr(manifest, attr)
        if not tensors:
            continue
        groups[name] = {
            "tensors": len(tensors),
            "chunks": sum(len(t.chunks) for t in tensors),
            "bytes": sum(t.byte_size for t in tensors),
        }
        for t in tensors:
            for c in t.chunks:
                b = blobs.setdefault(
                    c.blob_id,
                    {"blob_id": c.blob_id, "shard": c.shard, "bytes": c.stored_length, "packed": c.packed,
                     "chunks": 0, "payload_bytes": 0, "group": name},
                )
                b["chunks"] += 1
                b["payload_bytes"] += c.length
    objects = [
        {"name": o.name, "blob_id": o.blob_id, "bytes": o.length}
        for o in manifest.all_objects()
    ]
    return {
        "checkpoint_id": manifest.checkpoint_id,
        "status": manifest.status,
        "format_version": manifest.format_version,
        "epoch": manifest.epoch,
        "global_step": manifest.global_step,
        "best_metric": manifest.best_metric,
        "timestamp": manifest.timestamp,
        "model_name_or_path": manifest.model_name_or_path,
        "base_model_id": manifest.base_model_id,
        "dataset_id": manifest.dataset_id,
        "lora_config": manifest.lora_config,
        "chunk_size_bytes": manifest.chunk_size_bytes,
        "num_shards": manifest.num_shards,
        "groups": groups,
        "stored_objects": list(blobs.values()),
        "json_objects": objects,
        "payload_bytes": sum(g["bytes"] for g in groups.values()),
        "total_chunks": sum(g["chunks"] for g in groups.values()),
    }


def tensor_rows(manifest: Manifest, grep: Optional[str] = None) -> List[Dict[str, Any]]:
    rows = []
    for group, attr in _GROUPS:
        for t in getattr(manifest, attr):
            if grep and grep not in t.name:
                continue
            first = t.chunks[0] if t.chunks else None
            rows.append({
                "group": group, "name": t.name, "dtype": t.dtype, "shape": t.shape, "bytes": t.byte_size,
                "chunks": len(t.chunks), "blob_id": first.blob_id if first else None,
                "blob_offset": first.blob_offset if first and first.packed else None,
            })
    return rows


def _mb(n: float) -> str:
    return f"{n / 1e6:,.2f} MB"


def render_summary(info: Dict[str, Any]) -> str:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(info["timestamp"]))
    lines = [
        f"checkpoint {info['checkpoint_id']}   status {info['status']}   format v{info['format_version']}",
        f"  epoch {info['epoch']}  global_step {info['global_step']}  best_metric {info['best_metric']}  saved {ts}",
        f"  model {info['model_name_or_path']}   dataset {info['dataset_id']}",
        f"  lora {json.dumps(info['lora_config'])}",
        f"  written with chunk size {info['chunk_size_bytes'] / 1048576:g} MiB across {info['num_shards']} shard(s)",
        "  contents:",
    ]
    for g, v in info["groups"].items():
        lines.append(f"    {g:<15} {v['tensors']:>5} tensors  {v['chunks']:>5} chunks  {_mb(v['bytes']):>12}")
    packed = sum(1 for b in info["stored_objects"] if b["packed"])
    lines.append(
        f"  payload {_mb(info['payload_bytes'])} in {info['total_chunks']} chunks, stored as "
        f"{len(info['stored_objects'])} objects ({packed} packs, {len(info['stored_objects']) - packed} single-chunk)"
        f" + {len(info['json_objects'])} JSON state objects"
    )
    return "\n".join(lines)


def render_blobs(info: Dict[str, Any]) -> str:
    lines = ["stored objects (name as stored, shard, size, chunks inside):"]
    for b in info["stored_objects"]:
        kind = "pack" if b["packed"] else "chunk"
        lines.append(f"  {b['blob_id']}   shard {b['shard']}  {b['bytes']:>11,} B  {b['chunks']:>4} chunk(s)  [{kind}]")
    for o in info["json_objects"]:
        lines.append(f"  {o['blob_id']}   {o['bytes']:>20,} B  [json state]")
    return "\n".join(lines)


def render_tensors(rows: List[Dict[str, Any]]) -> str:
    lines = [f"{len(rows)} tensor(s):", f"  {'group':<14} {'name':<64} {'dtype':<15} {'shape':<14} {'bytes':>9}  where"]
    for r in rows:
        where = r["blob_id"].rsplit("/", 1)[-1] + (f" @{r['blob_offset']}" if r["blob_offset"] is not None else "")
        if r["chunks"] > 1:
            where += f" (+{r['chunks'] - 1} more chunks)"
        lines.append(f"  {r['group']:<14} {r['name'][:64]:<64} {r['dtype']:<15} {str(r['shape']):<14} {r['bytes']:>9,}  {where}")
    return "\n".join(lines)


def _open_manager(args: argparse.Namespace, max_object_bytes: int, num_workers: int) -> BlobStoreCheckpointManager:
    if args.local_dir:
        from pipeline.checkpoint.localfs_backend import LocalFsBackend

        storage = LocalFsBackend(args.local_dir)
    else:
        from pipeline.checkpoint.ddl_backend import DdlBackend

        storage = DdlBackend(
            args.ddl_server, args.run_id, port=args.ddl_port, tenant=args.ddl_tenant,
            max_chunk_bytes=max_object_bytes, timeout_seconds=args.timeout,
        )
    return BlobStoreCheckpointManager(
        args.run_id, storage=storage, num_workers=num_workers,
        chunk_size_bytes=max_object_bytes, verify_mode="none",
    )


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Inspect a stored checkpoint (read-only).")
    p.add_argument("--run-id", required=True, help="the job/run id the checkpoint was written under")
    p.add_argument("--ddl-server")
    p.add_argument("--ddl-port", type=int, default=58000)
    p.add_argument("--ddl-tenant", type=int, default=1, help="must match the job's tenant (the backend's ddl_tenant setting)")
    p.add_argument("--local-dir", help="inspect a local-storage checkpoint directory instead of DDL")
    p.add_argument("--checkpoint-id", help="default: the latest committed checkpoint")
    p.add_argument("--max-object-mb", type=int, default=4,
                   help="largest stored object to be read (DDL pre-allocates ~30x this per connection)")
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--blobs", action="store_true", help="list the stored objects and what each contains")
    p.add_argument("--tensors", action="store_true", help="list every tensor with dtype, shape and location")
    p.add_argument("--grep", help="with --tensors: only names containing this text")
    p.add_argument("--manifest", action="store_true", help="print the raw manifest JSON")
    p.add_argument("--verify", action="store_true", help="re-read every chunk and check its SHA-256")
    args = p.parse_args(argv)
    if not args.local_dir and not args.ddl_server:
        p.error("give --ddl-server (and --ddl-tenant) or --local-dir")

    mb = args.max_object_mb * 1024 * 1024
    workers = 1
    if args.local_dir:
        import os
        workers = max(1, len([d for d in os.listdir(args.local_dir) if d.startswith("shard-")]))
    mgr = _open_manager(args, mb, workers)
    try:
        catalog = mgr.list_checkpoints(include_incomplete=True)
        if not catalog:
            print(f"no checkpoints found for run {args.run_id!r} (check --run-id and --ddl-tenant)")
            return 1
        print(f"run {args.run_id}: {len(catalog)} checkpoint(s) in the catalog")
        for c in catalog:
            when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(c["timestamp"]))
            print(f"  {c['checkpoint_id']:<24} {c['status']:<10} epoch {c['epoch']}  step {c['global_step']}  {when}")
        cid = args.checkpoint_id or mgr.get_latest_checkpoint() or catalog[-1]["checkpoint_id"]
        manifest = mgr.read_manifest(cid)
        info = describe_checkpoint(manifest)
        print()
        print(render_summary(info))
        if args.blobs:
            print(); print(render_blobs(info))
        if args.tensors:
            print(); print(render_tensors(tensor_rows(manifest, args.grep)))
        if args.manifest:
            print(); print(manifest.to_json().decode("utf-8"))
        if args.verify:
            print()
            if not args.local_dir and manifest.chunk_size_bytes > mb:
                # objects in this checkpoint can be bigger than the client was sized for
                mgr.shutdown()
                mgr = _open_manager(args, manifest.chunk_size_bytes, workers)
            report = mgr.verify(cid)
            print(f"verify: ok={report.ok}  chunks_checked={report.chunks_checked}  objects_checked={report.objects_checked}")
            for issue in report.issues:
                print(f"  ISSUE {issue.kind}: {issue.name} {issue.detail}")
            return 0 if report.ok else 2
        return 0
    finally:
        mgr.shutdown()


if __name__ == "__main__":
    sys.exit(main())
