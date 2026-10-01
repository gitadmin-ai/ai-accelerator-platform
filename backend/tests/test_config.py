import pytest
from pydantic import ValidationError

from backend.config import JobConfig


def test_defaults_are_valid():
    config = JobConfig()
    assert config.model.name == "Qwen/Qwen2.5-0.5B"
    assert config.storage.backend == "local"
    assert config.training.method == "lora"


def test_lora_hyperparameter_defaults_match_training_script():
    # Must match train_lora_with_gpu_stats.py's own --lora-r/--lora-alpha/
    # --lora-dropout argparse defaults, so a config predating these fields
    # trains identically to before.
    config = JobConfig()
    assert config.training.lora_r == 8
    assert config.training.lora_alpha == 16
    assert config.training.lora_dropout == 0.05


def test_rejects_out_of_range_lora_r():
    with pytest.raises(ValidationError):
        JobConfig(training={"lora_r": 0})


def test_round_trips_through_json():
    config = JobConfig(name="my-job")
    restored = JobConfig.model_validate_json(config.model_dump_json())
    assert restored == config


def test_accepts_ddl_storage_backend():
    config = JobConfig(storage={"backend": "ddl"})
    assert config.storage.backend == "ddl"


def test_rejects_unknown_storage_backend():
    # Only "local" and "ddl" (Nebula) are implemented; anything else must
    # fail validation, not silently fall through.
    with pytest.raises(ValidationError):
        JobConfig(storage={"backend": "s3"})


def test_rejects_zero_epochs():
    with pytest.raises(ValidationError):
        JobConfig(training={"epochs": 0})


def test_accepts_full_yaml_shaped_payload():
    config = JobConfig(
        name="example-finetuning-job",
        model={"name": "Qwen/Qwen2.5-Coder-3B-Instruct"},
        dataset={"name": "onboarding-v1"},
        training={"method": "lora", "epochs": 3, "batch_size": 4, "learning_rate": 0.0002},
        storage={"backend": "local"},
        checkpoint={"frequency": "epoch"},
        evaluation={"enabled": True},
    )
    assert config.model.name == "Qwen/Qwen2.5-Coder-3B-Instruct"
    assert config.training.epochs == 3
