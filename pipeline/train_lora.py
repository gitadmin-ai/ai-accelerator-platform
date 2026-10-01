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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from pipeline.checkpoint import tensor_adapter
from pipeline.checkpoint.blobstore_backend import BlobStoreBackend
from pipeline.checkpoint.localfs_backend import LocalFsBackend
from pipeline.checkpoint.manager import BlobStoreCheckpointManager
from pipeline.utils import seed as seed_utils
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

    p.add_argument("--model-path", required=True, help="Base Qwen model path or HF hub id")
    p.add_argument("--dataset", required=True, help="Path to a JSONL file, or a HF datasets id")
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
    p.add_argument("--checkpoint-workers", type=int, default=4)
    p.add_argument("--checkpoint-chunk-size-mb", type=int, default=256)
    p.add_argument(
        "--checkpoint-verify-mode",
        default="checksum",
        choices=["checksum", "head", "none"],
        help="Inline post-write validation strength (see checkpoint/validation.py)",
    )
    p.add_argument(
        "--checkpoint-storage",
        default="blobstore",
        choices=["blobstore", "local"],
        help="Where checkpoint chunks are persisted: 'blobstore' (DDL "
        "eclib::BlobStore, default) or 'local' (plain files under "
        "--checkpoint-local-dir, see checkpoint/localfs_backend.py)",
    )
    p.add_argument(
        "--checkpoint-local-dir",
        default=None,
        help="Root directory for --checkpoint-storage local (required in that mode)",
    )
    p.add_argument("--resume", default=None, help="'latest' or an explicit checkpoint id")
    p.add_argument(
        "--save-full-model",
        action="store_true",
        help="Additionally save the merged full model (off by default -- LoRA training "
        "only needs the adapter weights, see checkpoint/manager.py)",
    )

    return p.parse_args(argv)


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


def make_collate_fn(pad_token_id: int):
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
        attention_mask = torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate


# ---------------------------------------------------------------------------
# Model / LoRA
# ---------------------------------------------------------------------------


def build_model_and_tokenizer(args: argparse.Namespace):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading tokenizer from %s", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer has no pad token; using eos token (%r) as pad token", tokenizer.eos_token)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
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

    return model, tokenizer, lora_config, device


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
    if args.checkpoint_storage == "local":
        if not args.checkpoint_local_dir:
            raise ValueError("--checkpoint-storage local requires --checkpoint-local-dir")
        storage = LocalFsBackend(args.checkpoint_local_dir)
    else:
        storage = BlobStoreBackend()
    return BlobStoreCheckpointManager(
        run_id=run_id,
        storage=storage,
        num_workers=args.checkpoint_workers,
        chunk_size_bytes=args.checkpoint_chunk_size_mb * 1024 * 1024,
        verify_mode=args.checkpoint_verify_mode,
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
) -> CheckpointMetrics:
    checkpoint_id = f"epoch-{state.epoch}-step-{state.global_step}"
    logger.info("Saving checkpoint %r (epoch=%d, global_step=%d) ...",
                checkpoint_id, state.epoch, state.global_step)

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
        model_name_or_path=args.model_path,
        base_model_id=args.model_path,
        dataset_id=args.dataset,
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
    return metrics


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
        raise RuntimeError("--resume latest requested but no committed checkpoint exists")

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


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.log_file, args.log_level)

    logger.info("=" * 70)
    logger.info("LoRA fine-tuning run starting")
    logger.info("=" * 70)
    logger.info("Arguments:\n%s", json.dumps(vars(args), indent=2, default=str))

    set_all_seeds(args.seed)
    logger.info("Seeded RNGs with seed=%d", args.seed)

    model, tokenizer, lora_config, device = build_model_and_tokenizer(args)
    dataset = load_training_dataset(args.dataset, tokenizer, args.max_seq_length)
    loader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    from transformers import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    manager = build_checkpoint_manager(args)
    state = TrainingState()
    if args.resume:
        state = resume_training_state(manager, args.resume, model, optimizer, scheduler, scaler)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Num epochs = %d (starting at epoch %d)", args.epochs, state.epoch)
    logger.info("  Batches per epoch = %d", len(loader))
    logger.info("  Per-device train batch size = %d", args.per_device_train_batch_size)
    logger.info("  Gradient accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Optimizer steps per epoch = %d", steps_per_epoch)
    logger.info("  Total optimization steps = %d (starting at global_step %d)", total_steps, state.global_step)
    logger.info("  Warmup steps = %d", warmup_steps)
    logger.info("  Learning rate = %s", args.learning_rate)

    try:
        model.train()
        train_start = time.time()
        for epoch in range(state.epoch, args.epochs):
            state.epoch = epoch
            epoch_start = time.time()
            running_loss = 0.0
            interval_loss = 0.0
            interval_batches = 0
            optimizer.zero_grad()

            logger.info("----- Epoch %d/%d starting -----", epoch + 1, args.epochs)

            for step, batch in enumerate(loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation_steps

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                running_loss += loss.item()
                interval_loss += outputs.loss.item()
                interval_batches += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    state.global_step += 1

                    if state.global_step % args.logging_steps == 0:
                        avg_loss = interval_loss / max(1, interval_batches)
                        current_lr = scheduler.get_last_lr()[0]
                        logger.info(
                            "epoch %d/%d | batch %d/%d | global_step %d/%d | loss %.4f | lr %.3e",
                            epoch + 1, args.epochs, step + 1, len(loader),
                            state.global_step, total_steps, avg_loss, current_lr,
                        )
                        interval_loss = 0.0
                        interval_batches = 0

                    if args.checkpoint_every_steps and state.global_step % args.checkpoint_every_steps == 0:
                        save_training_checkpoint(manager, model, optimizer, scheduler, scaler, state, args, lora_config)

            epoch_elapsed = time.time() - epoch_start
            logger.info(
                "----- Epoch %d/%d complete: mean loss %.4f | elapsed %.1fs | global_step %d -----",
                epoch + 1, args.epochs, running_loss / max(1, len(loader)), epoch_elapsed, state.global_step,
            )

            if args.checkpoint_every_epoch:
                save_training_checkpoint(manager, model, optimizer, scheduler, scaler, state, args, lora_config)

        total_elapsed = time.time() - train_start
        logger.info(
            "***** Training complete: %d epochs | %d optimizer steps | elapsed %.1fs *****",
            args.epochs, state.global_step, total_elapsed,
        )
    finally:
        logger.info("Shutting down checkpoint manager")
        manager.shutdown()


if __name__ == "__main__":
    main()
