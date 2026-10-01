"""Job configuration schema (see Section 6 of the architecture plan).

This is the one place a job's model/dataset/training/storage/checkpoint/
evaluation settings are defined. `training/`, the Job Manager, and the API
all consume this shape rather than hardcoding any of these values, so
changing which model or dataset a job uses -- including swapping the
storage backend later -- never requires a pipeline code change.

`storage.backend` governs checkpoint storage specifically; `model.source`/
`dataset.source` govern where the training worker reads the model/dataset
from. All three are `Literal["local", "ddl"]` -- "ddl" routes through
Nebula (see backend/settings_store.py, pipeline/checkpoint/ddl_backend.py,
shared/nebula-ddl-storage/). This Literal was deliberately narrow
(`Literal["local"]`) until that second backend existed; it's been widened
now that it does.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    # Any Qwen-family causal LM path/hub id works unmodified -- see
    # build_model_and_tokenizer() in train_lora_with_gpu_stats.py. Defaults
    # to a small model so Phase 1 jobs run in minutes on CPU; the production
    # target (Qwen2.5-Coder-3B-Instruct) is a config change, not a code one.
    # When source="ddl", `name` is a model resource id (not a path/hub id)
    # that the worker materializes from Nebula before loading.
    name: str = "Qwen/Qwen2.5-0.5B"
    source: Literal["local", "ddl"] = "local"


class DatasetConfig(BaseModel):
    # A local JSONL path or a HF datasets id -- see load_training_dataset().
    # When source="ddl", `name` is a dataset resource id, same as ModelConfig.
    name: str = "data/demo_onboarding_v1.jsonl"
    source: Literal["local", "ddl"] = "local"


class TrainingConfig(BaseModel):
    method: Literal["lora"] = "lora"
    epochs: int = Field(default=1, ge=1, le=100)
    batch_size: int = Field(default=2, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0)
    max_seq_length: int = Field(default=256, ge=8, le=8192)
    # Real, already-wired CLI flags on train_lora_with_gpu_stats.py
    # (--lora-r/--lora-alpha/--lora-dropout); defaults match that script's
    # own argparse defaults exactly so this is purely additive -- a config
    # predating these fields still trains identically to before.
    lora_r: int = Field(default=8, ge=1, le=256)
    lora_alpha: int = Field(default=16, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, lt=1.0)


class StorageConfig(BaseModel):
    backend: Literal["local", "ddl"] = "local"


class CheckpointConfig(BaseModel):
    frequency: Literal["epoch"] = "epoch"


class EvaluationConfig(BaseModel):
    enabled: bool = True
    split_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)


class JobConfig(BaseModel):
    name: str = "example-finetuning-job"
    model: ModelConfig = Field(default_factory=ModelConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
