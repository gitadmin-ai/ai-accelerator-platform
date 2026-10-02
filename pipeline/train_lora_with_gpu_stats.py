#!/usr/bin/env python3
"""LoRA fine-tuning for a Qwen-family causal LM, checkpointed through
BlobStoreCheckpointManager (chunked, parallel-write, BlobStore-backed).

This script requires torch, transformers, peft, and datasets, none of
which are installed in the sandbox this was developed in (no GPU either) --
see training/README.md's "Known limitations" section. It is written
against the current stable APIs of those libraries and exercises the same
BlobStoreCheckpointManager that training/tests/test_checkpoint_manager.py
covers end-to-end with real (numpy-tensor) data, but the torch-specific
code paths here (tensor_adapter, the training loop itself) are unexecuted.

Usage:
    python -m pipeline.train_lora \\
        --model-path Qwen/Qwen2.5-0.5B \\
        --dataset ./data/onboarding_tutor.jsonl \\
        --output-dir ./outputs/onboarding-tutor \\
        --epochs 3 --batch-size 2 --gradient-accumulation-steps 8 \\
        --learning-rate 2e-4 --checkpoint-every-epoch \\
        --checkpoint-workers 8 --checkpoint-chunk-size-mb 256 \\
        --log-file ./outputs/onboarding-tutor/train.log

    python -m pipeline.train_lora ... --resume latest
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from pipeline.checkpoint import tensor_adapter
from pipeline.checkpoint.blobstore_backend import BlobStoreBackend
from pipeline.checkpoint.localfs_backend import LocalFsBackend
from pipeline.checkpoint.manager import BlobStoreCheckpointManager
from pipeline.checkpoint.sizing import resolve_checkpoint_sizing
from pipeline.utils import seed as seed_utils
from pipeline.utils.events import JsonlEventEmitter
from pipeline.utils.metrics import CheckpointMetrics

logger = logging.getLogger("train_lora")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(log_file: Optional[str], log_level: str) -> None:
    """Configure the module logger, HF-Trainer-style: timestamped console
    output, mirrored to --log-file when one is given.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        log_dir = os.path.dirname(os.path.abspath(log_file))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info("Logging to file: %s", os.path.abspath(log_file))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--model-path", required=True, help="Base Qwen model path or HF hub id, or a "
                   "backend ModelStore resource id when --model-source ddl")
    p.add_argument("--dataset", required=True, help="Path to a JSONL file, or a HF datasets id, or a "
                   "backend DatasetStore resource id when --dataset-source ddl")
    p.add_argument(
        "--model-source", default="local", choices=["local", "ddl"],
        help="'local' (default): --model-path is a path/HF hub id, read as-is. 'ddl': --model-path is "
        "a resource id materialized from Nebula into --output-dir/_ddl_staging/model first (requires "
        "--ddl-server; see shared/nebula-ddl-storage/).",
    )
    p.add_argument(
        "--dataset-source", default="local", choices=["local", "ddl"],
        help="Same as --model-source but for --dataset (materialized into --output-dir/_ddl_staging/dataset).",
    )
    p.add_argument("--output-dir", required=True, help="Local dir for logs/config snapshots")
    p.add_argument("--run-id", default=None, help="Checkpoint run id (default: derived from output-dir)")

    # Logging
    p.add_argument(
        "--log-file",
        default=None,
        help="Path to a file to mirror training logs into (always also logs to stdout)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    p.add_argument(
        "--logging-steps",
        type=int,
        default=1,
        help="Log training loss/lr every N optimizer steps (HF Trainer-style)",
    )

    # Training configuration
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", "--per-device-train-batch-size", dest="per_device_train_batch_size", type=int, default=2)
    p.add_argument("--per-device-eval-batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)

    # DataLoader / performance monitoring
    p.add_argument("--dataloader-workers", type=int, default=2,
                   help="Number of DataLoader worker processes (0 disables multiprocessing)")
    p.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True,
                   help="Pin CPU tensors for faster CPU->GPU transfers")
    p.add_argument("--monitor-steps", type=int, default=100,
                   help="Emit detailed performance/GPU-memory metrics every N optimizer steps; 0 disables")
    p.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "fp16", "fp32"],
        help="Model/training dtype. 'auto' (default) picks fp32 on CPU and "
        "on Pascal GPUs with compute capability 6.1 (e.g. Quadro P6000, "
        "GTX 1080 Ti, Titan Xp) since those lack fast native fp16 "
        "throughput -- unlike P100 (capability 6.0) -- and otherwise fp16",
    )

    # LoRA configuration
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated module name suffixes LoRA adapters attach to",
    )
    p.add_argument("--lora-bias", default="none", choices=["none", "all", "lora_only"])

    # Checkpointing
    p.add_argument("--checkpoint-every-epoch", action="store_true")
    p.add_argument("--checkpoint-every-steps", type=int, default=0, help="0 disables step-based checkpointing")
    p.add_argument(
        "--checkpoint-workers", type=int, default=None,
        help="Parallel checkpoint writers (each owns one storage connection). Default: 2 for "
        "--checkpoint-storage ddl (every DDL connection pre-allocates memory), 4 otherwise.",
    )
    p.add_argument(
        "--checkpoint-chunk-size-mb", type=int, default=None,
        help="Checkpoint chunk size in MiB. Default: 4 for --checkpoint-storage ddl (each DDL "
        "connection pre-allocates ~30x this much memory, and the target's --max-object-mb must "
        "be at least this + 8 bytes), 256 for every other backend.",
    )
    p.add_argument(
        "--checkpoint-pack-size-mb", type=int, default=None,
        help="Chunks smaller than this are packed together into one stored object, so a "
        "checkpoint of many small tensors (a LoRA adapter) is a handful of writes instead of "
        "hundreds. Default: 4, capped at the chunk size; 0 disables packing.",
    )
    p.add_argument(
        "--checkpoint-verify-mode",
        default="checksum",
        choices=["checksum", "head", "none"],
        help="Inline post-write validation strength (see checkpoint/validation.py)",
    )
    p.add_argument(
        "--checkpoint-storage",
        default="blobstore",
        choices=["blobstore", "local", "ddl"],
        help="Where checkpoint chunks are persisted: 'blobstore' (DDL "
        "eclib::BlobStore, default), 'local' (plain files under "
        "--checkpoint-local-dir, see checkpoint/localfs_backend.py), or "
        "'ddl' (a real DDL3 MiniFS target over the network, see "
        "checkpoint/ddl_backend.py -- requires --ddl-server)",
    )
    p.add_argument(
        "--checkpoint-local-dir",
        default=None,
        help="Root directory for --checkpoint-storage local (required in that mode)",
    )
    p.add_argument(
        "--ddl-server",
        default=None,
        help="DDL3 MiniFS target address for --checkpoint-storage ddl (required in that mode)",
    )
    p.add_argument("--ddl-port", type=int, default=58000, help="DDL3 MiniFS target port")
    p.add_argument(
        "--ddl-tenant", type=int, default=1,
        help="DDL3 tenant id -- gives runs/jobs a separate on-disk path when isolation "
        "stronger than the per-run key namespace (see checkpoint/ddl_backend.py) is wanted",
    )
    p.add_argument(
        "--ddl-cpu-base", type=int, default=0,
        help="First DDL3 owner CPU; shard i uses ddl-cpu-base + i (one owner core per "
        "--checkpoint-workers worker)",
    )
    p.add_argument("--resume", default=None, help="'latest' or an explicit checkpoint id")
    p.add_argument(
        "--save-full-model",
        action="store_true",
        help="Additionally save the merged full model (off by default -- LoRA training "
        "only needs the adapter weights, see checkpoint/manager.py)",
    )

    # Evaluation
    p.add_argument(
        "--evaluate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a held-out-loss evaluation pass after each epoch checkpoint",
    )
    p.add_argument(
        "--eval-split-ratio",
        type=float,
        default=0.1,
        help="Fraction of the dataset held out for evaluation (0 disables the split "
        "and evaluation, regardless of --evaluate)",
    )

    # Job Manager integration (see app/job_manager.py) -- both optional so this
    # script still runs standalone from the CLI exactly as before.
    p.add_argument(
        "--job-id",
        default=None,
        help="Job id to tag emitted lifecycle events with (default: --run-id)",
    )
    p.add_argument(
        "--events-file",
        default=None,
        help="Path to append newline-delimited JSON lifecycle events to. "
        "Unset by default -- no events are emitted and this script behaves "
        "exactly as a plain CLI training run.",
    )

    return p.parse_args(argv)


def materialize_ddl_resource(args: argparse.Namespace, namespace: str, resource_id: str, subdir: str) -> "Path":
    """Pulls a model/dataset resource id out of Nebula (DDL) into
    --output-dir/_ddl_staging/<subdir> and returns that local directory --
    used by --model-source/--dataset-source ddl so build_model_and_tokenizer()/
    load_training_dataset() only ever see a plain local path from here on,
    exactly as they do for the local case. Lazy import: DDL support is
    optional and shouldn't be required to run --model-source local jobs.
    """
    from pathlib import Path

    from nebula_ddl_storage.resource_store import DdlResourceStore

    if not args.ddl_server:
        raise ValueError(f"--model-source/--dataset-source ddl requires --ddl-server (materializing {namespace!r})")
    target_dir = Path(args.output_dir) / "_ddl_staging" / subdir
    store = DdlResourceStore(
        namespace=namespace, server=args.ddl_server, port=args.ddl_port,
        tenant=args.ddl_tenant, ddl_cpu_base=args.ddl_cpu_base,
    )
    try:
        return store.get_directory(resource_id, target_dir)
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_training_dataset(dataset_path: str, tokenizer, max_seq_length: int):
    from datasets import load_dataset

    logger.info("Loading dataset from %s", dataset_path)
    if os.path.exists(dataset_path):
        raw = load_dataset("json", data_files=dataset_path, split="train")
    else:
        raw = load_dataset(dataset_path, split="train")

    text_field = "text" if "text" in raw.column_names else raw.column_names[0]
    logger.info("Dataset loaded: %d examples, text field=%r, max_seq_length=%d",
                len(raw), text_field, max_seq_length)

    def tokenize(batch):
        return tokenizer(
            batch[text_field],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )

    tokenized = raw.map(tokenize, batched=True, remove_columns=raw.column_names)
    logger.info("Tokenization complete: %d examples ready for training", len(tokenized))
    return tokenized


def split_train_eval(tokenized, eval_split_ratio: float, seed: int):
    """Holds out `eval_split_ratio` of `tokenized` for evaluation, deterministically
    (same seed as training). Returns (train_dataset, eval_dataset_or_None).

    A ratio <= 0, or too few examples to hold any out, disables the split --
    evaluation is then a no-op regardless of --evaluate (see main()).
    """
    if eval_split_ratio and eval_split_ratio > 0 and len(tokenized) > 1:
        split = tokenized.train_test_split(test_size=eval_split_ratio, seed=seed)
        return split["train"], split["test"]
    return tokenized, None


def run_evaluation(model, eval_loader, device) -> Dict[str, float]:
    """Forward-only pass over `eval_loader`; mean loss is the only metric --
    real, not estimated. Restores model.train() before returning so callers
    don't need to remember to.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            num_batches += 1
    model.train()
    return {"eval_loss": total_loss / max(1, num_batches), "num_eval_batches": num_batches}


def make_collate_fn(tokenizer):
    def collate(batch):
        batch = tokenizer.pad(
            batch,
            padding=True,
            return_tensors="pt"
        )

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels

        return batch

    return collate

# ---------------------------------------------------------------------------
# Model / LoRA
# ---------------------------------------------------------------------------


def resolve_train_dtype(dtype_arg: str, device: str) -> torch.dtype:
    """Picks the model/training dtype for --dtype auto.

    Defaults to fp32 on CPU. On CUDA, defaults to fp32 for Pascal chips
    with compute capability 6.1 (Quadro P6000, GTX 1080 Ti, Titan Xp,
    etc.) -- unlike P100 (compute capability 6.0), those chips run fp16
    math at roughly 1/64th of fp32 throughput, so fp16 buys no speedup
    (and can be slower) while still costing precision. Everything else
    defaults to fp16.
    """
    if dtype_arg == "fp16":
        return torch.float16
    if dtype_arg == "fp32":
        return torch.float32
    if device != "cuda":
        return torch.float32
    if torch.cuda.get_device_capability(0) == (6, 1):
        logger.info(
            "Detected compute capability 6.1 (e.g. Quadro P6000, GTX 1080 "
            "Ti, Titan Xp) -- these Pascal chips lack fast native fp16 "
            "throughput, so --dtype auto selects fp32"
        )
        return torch.float32
    return torch.float16


def build_model_and_tokenizer(args: argparse.Namespace):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading tokenizer from %s", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer has no pad token; using eos token (%r) as pad token", tokenizer.eos_token)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        logger.info("CUDA device: %s | capability=%s",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_capability(0))
    dtype = resolve_train_dtype(args.dtype, device)
    logger.info("Loading base model %s (device=%s, dtype=%s)", args.model_path, device, dtype)
    base_model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype).to(device)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules.split(","),
        bias=args.lora_bias,
        task_type=TaskType.CAUSAL_LM,
    )
    logger.info(
        "Applying LoRA adapters: r=%d, alpha=%d, dropout=%s, target_modules=%s, bias=%s",
        lora_config.r, lora_config.lora_alpha, lora_config.lora_dropout,
        list(lora_config.target_modules), lora_config.bias,
    )
    model = get_peft_model(base_model, lora_config)

    try:
        trainable_params, all_params = model.get_nb_trainable_parameters()
        logger.info(
            "trainable params: %d || all params: %d || trainable%%: %.4f",
            trainable_params, all_params, 100 * trainable_params / all_params,
        )
    except AttributeError:
        model.print_trainable_parameters()

    return model, tokenizer, lora_config, device, dtype


def lora_state_dict(model) -> Dict[str, torch.Tensor]:
    """Only the trainable LoRA adapter weights -- the frozen base model is
    never checkpointed (constraint #18 / checkpoint/manager.py docstring).
    """
    from peft import get_peft_model_state_dict

    return {k: v for k, v in get_peft_model_state_dict(model).items()}


def load_lora_state_dict(model, state_dict: Dict[str, torch.Tensor]) -> None:
    from peft import set_peft_model_state_dict

    set_peft_model_state_dict(model, state_dict)


# ---------------------------------------------------------------------------
# Checkpoint <-> training-loop plumbing
# ---------------------------------------------------------------------------


@dataclass
class TrainingState:
    epoch: int = 0
    global_step: int = 0
    best_metric: Optional[float] = None
    grad_scaler: Optional[Dict[str, Any]] = field(default_factory=dict)


def build_checkpoint_manager(args: argparse.Namespace) -> BlobStoreCheckpointManager:
    run_id = args.run_id or os.path.basename(os.path.normpath(args.output_dir))
    chunk_size_mb, num_workers = resolve_checkpoint_sizing(
        args.checkpoint_storage, args.checkpoint_chunk_size_mb, args.checkpoint_workers
    )
    if args.checkpoint_storage == "local":
        if not args.checkpoint_local_dir:
            raise ValueError("--checkpoint-storage local requires --checkpoint-local-dir")
        storage = LocalFsBackend(args.checkpoint_local_dir)
    elif args.checkpoint_storage == "ddl":
        if not args.ddl_server:
            raise ValueError("--checkpoint-storage ddl requires --ddl-server")
        from pipeline.checkpoint.ddl_backend import DdlBackend

        storage = DdlBackend(
            server=args.ddl_server,
            run_id=run_id,
            port=args.ddl_port,
            tenant=args.ddl_tenant,
            max_chunk_bytes=chunk_size_mb * 1024 * 1024,
            ddl_cpu_base=args.ddl_cpu_base,
        )
    else:
        storage = BlobStoreBackend()
    return BlobStoreCheckpointManager(
        run_id=run_id,
        storage=storage,
        num_workers=num_workers,
        chunk_size_bytes=chunk_size_mb * 1024 * 1024,
        verify_mode=args.checkpoint_verify_mode,
        pack_size_bytes=(
            None if args.checkpoint_pack_size_mb is None else args.checkpoint_pack_size_mb * 1024 * 1024
        ),
    )


def save_training_checkpoint(
    manager: BlobStoreCheckpointManager,
    model,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[torch.cuda.amp.GradScaler],
    state: TrainingState,
    args: argparse.Namespace,
    lora_config,
    emitter: JsonlEventEmitter,
) -> Tuple[str, CheckpointMetrics]:
    checkpoint_id = f"epoch-{state.epoch}-step-{state.global_step}"
    logger.info("Saving checkpoint %r (epoch=%d, global_step=%d) ...",
                checkpoint_id, state.epoch, state.global_step)
    emitter.emit(
        "CHECKPOINTING", "checkpoint_start",
        checkpoint_id=checkpoint_id, epoch=state.epoch, global_step=state.global_step,
    )

    training_state: Dict[str, Any] = {
        "epoch": state.epoch,
        "global_step": state.global_step,
        "best_metric": state.best_metric,
        "rng": seed_utils.capture_rng_state(),
    }
    if scaler is not None:
        training_state["grad_scaler"] = scaler.state_dict()

    full_model_state = None
    if args.save_full_model:
        full_model_state = {k: v for k, v in model.state_dict().items()}

    manifest, metrics = manager.save_checkpoint(
        checkpoint_id,
        lora_state_dict(model),
        optimizer.state_dict(),
        scheduler.state_dict() if scheduler is not None else None,
        training_state,
        epoch=state.epoch,
        global_step=state.global_step,
        model_name_or_path=args.model_id,
        base_model_id=args.model_id,
        dataset_id=args.dataset_id_original,
        lora_config=dict(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            target_modules=list(lora_config.target_modules),
            bias=lora_config.bias,
        ),
        training_config=dict(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            num_epochs=args.epochs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            max_seq_length=args.max_seq_length,
        ),
        best_metric=state.best_metric,
        full_model_state=full_model_state,
    )
    logger.info("Checkpoint %r saved\n%s", checkpoint_id, metrics.format_report())
    emitter.emit(
        "CHECKPOINTING", "checkpoint_saved",
        checkpoint_id=checkpoint_id, epoch=state.epoch, global_step=state.global_step,
        size_bytes=metrics.size_bytes, num_tensors=metrics.num_tensors, num_chunks=metrics.num_chunks,
        num_blobs=metrics.num_blobs,
        chunk_size_bytes=metrics.chunk_size_bytes, num_workers=metrics.num_workers,
        gpu_to_cpu_s=metrics.gpu_to_cpu_s, chunking_s=metrics.chunking_s, write_s=metrics.write_s,
        commit_s=metrics.commit_s, total_s=metrics.total_s, throughput_mb_s=metrics.throughput_mb_s,
    )
    return checkpoint_id, metrics


def resume_training_state(
    manager: BlobStoreCheckpointManager,
    resume: str,
    model,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[torch.cuda.amp.GradScaler],
) -> TrainingState:
    checkpoint_id = manager.get_latest_checkpoint() if resume == "latest" else resume
    if checkpoint_id is None:
        raise RuntimeError(f"--resume {resume!r} requested but no matching committed checkpoint exists")

    logger.info("Resuming from checkpoint %r ...", checkpoint_id)
    loaded = manager.load_checkpoint(checkpoint_id)

    lora_weights = {
        name: tensor_adapter.reconstruct_torch_tensor(dtype, shape, raw)
        for name, (dtype, shape, raw) in loaded.model_tensors_raw.items()
    }
    load_lora_state_dict(model, lora_weights)

    from pipeline.checkpoint import state_flatten

    if loaded.optimizer_skeleton is not None:
        opt_tensors = {
            name: tensor_adapter.reconstruct_torch_tensor(dtype, shape, raw)
            for name, (dtype, shape, raw) in loaded.optimizer_tensors_raw.items()
        }
        optimizer.load_state_dict(state_flatten.unflatten_state_dict(loaded.optimizer_skeleton, opt_tensors))

    if scheduler is not None and loaded.scheduler_state is not None:
        scheduler.load_state_dict(state_flatten.unflatten_state_dict(loaded.scheduler_state, {}))

    ts_tensors = {
        name: tensor_adapter.reconstruct_torch_tensor(dtype, shape, raw)
        for name, (dtype, shape, raw) in loaded.training_state_tensors_raw.items()
    }
    training_state = state_flatten.unflatten_state_dict(loaded.training_state_skeleton, ts_tensors)

    seed_utils.restore_rng_state(training_state["rng"])
    if scaler is not None and "grad_scaler" in training_state:
        scaler.load_state_dict(training_state["grad_scaler"])

    logger.info("Resumed from checkpoint %r (epoch=%d, global_step=%d)",
                checkpoint_id, training_state["epoch"], training_state["global_step"])
    return TrainingState(
        epoch=training_state["epoch"],
        global_step=training_state["global_step"],
        best_metric=training_state.get("best_metric"),
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PhaseTimer:
    """Accumulates wall-clock time spent in one phase (H2D copy, forward,
    backward, optimizer step) across many calls, without forcing a CUDA
    sync on every call.

    CUDA kernels launched inside start()/stop() run asynchronously, so a
    naive perf_counter() delta around them only measures how long it took
    to *enqueue* the work, not how long the GPU actually spent on it.
    Instead, each start()/stop() pair records a torch.cuda.Event; drain()
    reads back their elapsed_time() to get the true GPU duration. That
    read is only valid once the events have fired, so the caller must
    only call drain() right after a torch.cuda.synchronize() -- querying
    an event that hasn't completed yet raises.

    On CPU (no CUDA), phases already run synchronously, so this just
    falls back to plain perf_counter() timing.
    """

    def __init__(self, use_cuda: bool, enabled: bool = True):
        self.use_cuda = use_cuda
        self.enabled = enabled
        self._pending: List[List[Optional["torch.cuda.Event"]]] = []
        self._cpu_total = 0.0
        self._cpu_start: Optional[float] = None

    def start(self) -> None:
        if not self.enabled:
            return
        if self.use_cuda:
            start_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            self._pending.append([start_evt, None])
        else:
            self._cpu_start = time.perf_counter()

    def stop(self) -> None:
        if not self.enabled:
            return
        if self.use_cuda:
            end_evt = torch.cuda.Event(enable_timing=True)
            end_evt.record()
            self._pending[-1][1] = end_evt
        else:
            self._cpu_total += time.perf_counter() - self._cpu_start
            self._cpu_start = None

    def drain(self) -> float:
        """Return accumulated seconds since the last drain(), and reset.
        Only call this right after a torch.cuda.synchronize()."""
        if not self.enabled:
            return 0.0
        if self.use_cuda:
            total_ms = sum(s.elapsed_time(e) for s, e in self._pending)
            self._pending.clear()
            return total_ms / 1000.0
        total = self._cpu_total
        self._cpu_total = 0.0
        return total


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.log_file, args.log_level)

    args.run_id = args.run_id or os.path.basename(os.path.normpath(args.output_dir))
    job_id = args.job_id or args.run_id
    emitter = JsonlEventEmitter(args.events_file, job_id=job_id)
    try:
        _run_training(args, emitter)
    except Exception as exc:
        emitter.emit("FAILED", "job_failed", error=str(exc), error_type=type(exc).__name__)
        raise
    finally:
        emitter.close()


def _run_training(args: argparse.Namespace, emitter: JsonlEventEmitter) -> None:
    logger.info("=" * 70)
    logger.info("LoRA fine-tuning run starting")
    logger.info("=" * 70)
    logger.info("Arguments:\n%s", json.dumps(vars(args), indent=2, default=str))
    emitter.emit("RUNNING", "worker_started", model=args.model_path, dataset=args.dataset,
                 epochs=args.epochs, run_id=args.run_id)

    set_all_seeds(args.seed)
    logger.info("Seeded RNGs with seed=%d", args.seed)

    # Preserve the original resource id for checkpoint manifest provenance
    # (save_training_checkpoint's model_name_or_path/base_model_id/dataset_id)
    # -- args.model_path/args.dataset get overwritten below with a local
    # staging path once materialized, which wouldn't otherwise be traceable
    # back to which Nebula resource a checkpoint was trained from.
    args.model_id = args.model_path
    args.dataset_id_original = args.dataset

    if args.model_source == "ddl":
        emitter.emit("RUNNING", "model_materializing_start", model=args.model_path, source="ddl")
        args.model_path = str(materialize_ddl_resource(args, "models", args.model_path, "model"))
        emitter.emit("RUNNING", "model_materialized", model=args.model_path)
    if args.dataset_source == "ddl":
        emitter.emit("RUNNING", "dataset_materializing_start", dataset=args.dataset, source="ddl")
        materialized_dir = materialize_ddl_resource(args, "datasets", args.dataset, "dataset")
        args.dataset = str(materialized_dir / "dataset.jsonl")
        emitter.emit("RUNNING", "dataset_materialized", dataset=args.dataset)

    emitter.emit("RUNNING", "model_loading_start", model=args.model_path)
    model, tokenizer, lora_config, device, dtype = build_model_and_tokenizer(args)
    emitter.emit("RUNNING", "model_loaded", model=args.model_path, device=device, dtype=str(dtype))

    emitter.emit("RUNNING", "dataset_loading_start", dataset=args.dataset)
    full_dataset = load_training_dataset(args.dataset, tokenizer, args.max_seq_length)
    train_dataset, eval_dataset = split_train_eval(full_dataset, args.eval_split_ratio, args.seed)
    evaluation_active = args.evaluate and eval_dataset is not None
    emitter.emit(
        "RUNNING", "dataset_loaded",
        num_train_examples=len(train_dataset),
        num_eval_examples=len(eval_dataset) if eval_dataset is not None else 0,
        evaluation_enabled=evaluation_active,
    )

    loader_kwargs = dict(
        dataset=train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer),
        num_workers=args.dataloader_workers,
        pin_memory=args.pin_memory and torch.cuda.is_available(),
    )
    if args.dataloader_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(**loader_kwargs)

    eval_loader = None
    if evaluation_active:
        eval_loader = DataLoader(
            dataset=eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=make_collate_fn(tokenizer),
        )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # The training loop performs optimizer.step() only after a complete
    # accumulation window. Use floor here so scheduler.total_steps matches
    # the number of optimizer updates actually performed.
    steps_per_epoch = len(loader) // args.gradient_accumulation_steps
    remainder_batches = len(loader) % args.gradient_accumulation_steps
    usable_batches = steps_per_epoch * args.gradient_accumulation_steps
    if steps_per_epoch == 0:
        raise ValueError(
            "Number of batches must be >= gradient accumulation steps "
            "for this training loop"
        )
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    from transformers import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    # GradScaler is only meaningful in fp16 -- fp32 training has no
    # underflow risk to guard against, so skip it there.
    scaler = (
        torch.cuda.amp.GradScaler()
        if dtype == torch.float16 and torch.cuda.is_available()
        else None
    )

    manager = build_checkpoint_manager(args)
    state = TrainingState()
    if args.resume == "latest" and manager.get_latest_checkpoint() is None:
        # "latest" is a best-effort request (e.g. an unconditional --resume
        # latest on every launch) -- fall back to a fresh run instead of
        # erroring when nothing has been committed yet. An explicit
        # checkpoint id that doesn't exist still fails loudly below.
        logger.info("--resume latest requested but no committed checkpoint exists yet; starting fresh")
    elif args.resume:
        state = resume_training_state(manager, args.resume, model, optimizer, scheduler, scaler)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num epochs = %d (starting at epoch %d)", args.epochs, state.epoch)
    logger.info("  Batches per epoch = %d", len(loader))
    logger.info("  Per-device train batch size = %d", args.per_device_train_batch_size)
    logger.info("  Gradient accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Optimizer steps per epoch = %d", steps_per_epoch)
    logger.info("  Total optimization steps = %d (starting at global_step %d)", total_steps, state.global_step)
    logger.info("  Warmup steps = %d", warmup_steps)
    logger.info("  Learning rate = %s", args.learning_rate)
    logger.info("  DataLoader workers = %d", args.dataloader_workers)
    logger.info("  Pin memory = %s", args.pin_memory and torch.cuda.is_available())
    logger.info("  Detailed performance monitor interval = %d optimizer steps", args.monitor_steps)
    if remainder_batches:
        logger.warning(
            "  WARNING: %d final microbatches per epoch do not form a complete "
            "gradient-accumulation window and will be dropped", remainder_batches
        )

    def gpu_memory_stats():
        if not torch.cuda.is_available():
            return None
        return {
            "allocated_gb": torch.cuda.memory_allocated() / (1024 ** 3),
            "reserved_gb": torch.cuda.memory_reserved() / (1024 ** 3),
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / (1024 ** 3),
        }

    last_checkpoint_id: Optional[str] = None
    try:
        model.train()
        train_start = time.time()
        for epoch in range(state.epoch, args.epochs):
            state.epoch = epoch
            epoch_start = time.time()
            running_loss = torch.zeros((), device=device)
            interval_loss = torch.zeros((), device=device)
            interval_batches = 0
            optimizer.zero_grad()

            logger.info("----- Epoch %d/%d starting -----", epoch + 1, args.epochs)
            emitter.emit("RUNNING", "epoch_start", epoch=epoch + 1, total_epochs=args.epochs,
                         global_step=state.global_step)

            # CUDA kernels launched below run asynchronously; per-phase
            # durations are measured with PhaseTimer (cuda Events, drained
            # only once synchronized at the monitoring boundary) rather than
            # a synchronous wall-clock delta around each call.
            monitor_start = time.perf_counter()
            monitor_batches = 0
            monitor_data_wait = 0.0
            use_cuda = torch.cuda.is_available()
            monitor_enabled = args.monitor_steps > 0
            h2d_timer = PhaseTimer(use_cuda, monitor_enabled)
            forward_timer = PhaseTimer(use_cuda, monitor_enabled)
            backward_timer = PhaseTimer(use_cuda, monitor_enabled)
            optimizer_timer = PhaseTimer(use_cuda, monitor_enabled)

            data_iter = iter(loader)
            step = 0
            while step < usable_batches:
                data_wait_start = time.perf_counter()
                batch = next(data_iter)
                data_wait = time.perf_counter() - data_wait_start

                h2d_timer.start()
                batch = {
                    k: v.to(device, non_blocking=True)
                    for k, v in batch.items()
                }
                h2d_timer.stop()

                forward_timer.start()
                outputs = model(**batch)
                forward_timer.stop()

                backward_timer.start()
                loss = outputs.loss / args.gradient_accumulation_steps
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                backward_timer.stop()

                # Accumulate on-device and defer the GPU->CPU sync (.item())
                # to the logging boundary below -- syncing every microbatch
                # (16x/optimizer-step here) serializes the pipeline and
                # stalls the async H2D/forward/backward overlap.
                batch_loss = outputs.loss.detach()
                running_loss += batch_loss
                interval_loss += batch_loss
                interval_batches += 1
                monitor_batches += 1
                monitor_data_wait += data_wait

                is_accumulation_boundary = (
                    (step + 1) % args.gradient_accumulation_steps == 0
                )

                if is_accumulation_boundary:
                    optimizer_timer.start()
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    optimizer_timer.stop()
                    state.global_step += 1

                    if state.global_step % args.logging_steps == 0:
                        avg_loss = (interval_loss / max(1, interval_batches)).item()
                        current_lr = scheduler.get_last_lr()[0]
                        logger.info(
                            "epoch %d/%d | batch %d/%d | global_step %d/%d | "
                            "loss %.4f | lr %.3e",
                            epoch + 1, args.epochs, step + 1, usable_batches,
                            state.global_step, total_steps, avg_loss, current_lr,
                        )
                        emitter.emit(
                            "RUNNING", "step", epoch=epoch + 1, global_step=state.global_step,
                            total_steps=total_steps, loss=avg_loss, learning_rate=current_lr,
                        )
                        interval_loss = torch.zeros((), device=device)
                        interval_batches = 0

                    if (
                        args.monitor_steps
                        and state.global_step % args.monitor_steps == 0
                    ):
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        monitor_elapsed = time.perf_counter() - monitor_start
                        # Safe to drain now: the synchronize() above
                        # guarantees every recorded cuda Event has fired.
                        h2d_elapsed = h2d_timer.drain()
                        forward_elapsed = forward_timer.drain()
                        backward_elapsed = backward_timer.drain()
                        optimizer_elapsed = optimizer_timer.drain()
                        mem = gpu_memory_stats()
                        batches_per_sec = (
                            monitor_batches / monitor_elapsed
                            if monitor_elapsed > 0 else 0.0
                        )
                        optimizer_steps_per_sec = (
                            args.monitor_steps / monitor_elapsed
                            if monitor_elapsed > 0 else 0.0
                        )
                        nominal_tokens = (
                            monitor_batches
                            * args.per_device_train_batch_size
                            * args.max_seq_length
                        )
                        tokens_per_sec = (
                            nominal_tokens / monitor_elapsed
                            if monitor_elapsed > 0 else 0.0
                        )
                        if mem is not None:
                            logger.info(
                                "PERF | step=%d | window=%.2fs | "
                                "batch/s=%.2f | opt-step/s=%.2f | "
                                "nominal_tokens/s=%.0f | data_wait=%.3fs | "
                                "h2d=%.3fs | forward=%.3fs | backward=%.3fs | "
                                "optimizer=%.3fs | GPU mem allocated=%.2fGB | "
                                "reserved=%.2fGB | peak_allocated=%.2fGB | "
                                "peak_reserved=%.2fGB",
                                state.global_step, monitor_elapsed,
                                batches_per_sec, optimizer_steps_per_sec,
                                tokens_per_sec, monitor_data_wait, h2d_elapsed,
                                forward_elapsed, backward_elapsed,
                                optimizer_elapsed, mem["allocated_gb"],
                                mem["reserved_gb"], mem["peak_allocated_gb"],
                                mem["peak_reserved_gb"],
                            )
                        else:
                            logger.info(
                                "PERF | step=%d | window=%.2fs | batch/s=%.2f | "
                                "opt-step/s=%.2f | nominal_tokens/s=%.0f | "
                                "data_wait=%.3fs | h2d=%.3fs | forward=%.3fs | "
                                "backward=%.3fs | optimizer=%.3fs",
                                state.global_step, monitor_elapsed,
                                batches_per_sec, optimizer_steps_per_sec,
                                tokens_per_sec, monitor_data_wait, h2d_elapsed,
                                forward_elapsed, backward_elapsed,
                                optimizer_elapsed,
                            )
                        if torch.cuda.is_available():
                            torch.cuda.reset_peak_memory_stats()
                        monitor_start = time.perf_counter()
                        monitor_batches = 0
                        monitor_data_wait = 0.0

                    if args.checkpoint_every_steps and state.global_step % args.checkpoint_every_steps == 0:
                        checkpoint_start = time.perf_counter()
                        last_checkpoint_id, _ = save_training_checkpoint(
                            manager, model, optimizer, scheduler, scaler,
                            state, args, lora_config, emitter,
                        )
                        checkpoint_elapsed = time.perf_counter() - checkpoint_start
                        logger.info(
                            "CHECKPOINT PERF | step=%d | elapsed=%.3fs",
                            state.global_step, checkpoint_elapsed
                        )

                step += 1

            epoch_elapsed = time.time() - epoch_start
            epoch_mean_loss = (running_loss / max(1, usable_batches)).item()
            logger.info(
                "----- Epoch %d/%d complete: mean loss %.4f | elapsed %.1fs | global_step %d -----",
                epoch + 1, args.epochs, epoch_mean_loss, epoch_elapsed, state.global_step,
            )
            emitter.emit(
                "RUNNING", "epoch_complete", epoch=epoch + 1, total_epochs=args.epochs,
                mean_loss=epoch_mean_loss, elapsed_s=epoch_elapsed, global_step=state.global_step,
            )

            if args.checkpoint_every_epoch:
                checkpoint_start = time.perf_counter()
                last_checkpoint_id, _ = save_training_checkpoint(
                    manager, model, optimizer, scheduler, scaler,
                    state, args, lora_config, emitter,
                )
                logger.info(
                    "CHECKPOINT PERF | epoch=%d | elapsed=%.3fs",
                    epoch + 1, time.perf_counter() - checkpoint_start
                )

            if evaluation_active:
                emitter.emit("EVALUATING", "eval_start", epoch=epoch + 1)
                eval_start = time.perf_counter()
                eval_metrics = run_evaluation(model, eval_loader, device)
                eval_elapsed = time.perf_counter() - eval_start
                logger.info(
                    "EVAL | epoch=%d | eval_loss=%.4f | num_batches=%d | elapsed=%.3fs",
                    epoch + 1, eval_metrics["eval_loss"], eval_metrics["num_eval_batches"], eval_elapsed,
                )
                emitter.emit(
                    "EVALUATING", "eval_complete", epoch=epoch + 1,
                    eval_loss=eval_metrics["eval_loss"],
                    num_eval_batches=eval_metrics["num_eval_batches"],
                    elapsed_s=eval_elapsed,
                )
                if state.best_metric is None or eval_metrics["eval_loss"] < state.best_metric:
                    state.best_metric = eval_metrics["eval_loss"]

        total_elapsed = time.time() - train_start
        logger.info(
            "***** Training complete: %d epochs | %d optimizer steps | elapsed %.1fs *****",
            args.epochs, state.global_step, total_elapsed,
        )

        artifact_dir = os.path.join(args.output_dir, "artifacts", "final_adapter")
        os.makedirs(artifact_dir, exist_ok=True)
        model.save_pretrained(artifact_dir)
        tokenizer.save_pretrained(artifact_dir)
        logger.info("Final adapter artifact saved to %s", artifact_dir)

        emitter.emit(
            "COMPLETED", "job_complete",
            total_epochs=args.epochs, total_steps=state.global_step, elapsed_s=total_elapsed,
            final_checkpoint_id=last_checkpoint_id, artifact_path=artifact_dir,
            best_metric=state.best_metric,
        )
    finally:
        logger.info("Shutting down checkpoint manager")
        manager.shutdown()


if __name__ == "__main__":
    main()
