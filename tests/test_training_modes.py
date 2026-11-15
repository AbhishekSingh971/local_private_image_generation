import pytest

from training_modes import (
    FULL_FINETUNE_MIN_VRAM_GB,
    TRAINING_MODE_FULL_FINETUNE,
    TRAINING_MODE_LORA,
    resolve_training_mode,
    validate_training_settings,
    validate_training_vram,
)


def test_legacy_or_missing_training_mode_defaults_to_lora():
    assert resolve_training_mode(None) == TRAINING_MODE_LORA
    assert resolve_training_mode("") == TRAINING_MODE_LORA


def test_training_mode_accepts_lora_and_full_finetune():
    assert resolve_training_mode("lora") == TRAINING_MODE_LORA
    assert resolve_training_mode(" FULL_FINETUNE ") == TRAINING_MODE_FULL_FINETUNE


def test_unknown_training_mode_is_rejected():
    with pytest.raises(ValueError, match="Training method must be"):
        resolve_training_mode("dreambooth-ish")


def test_lora_training_keeps_existing_step_range():
    assert validate_training_settings("lora", "800", "512") == (800, 512)
    assert validate_training_settings("lora", 3000, 384) == (3000, 384)


def test_full_finetune_has_its_own_step_range():
    assert validate_training_settings("full_finetune", "400", "512") == (400, 512)
    assert validate_training_settings("full_finetune", 2000, 384) == (2000, 384)
    with pytest.raises(ValueError, match="between 100 and 2000"):
        validate_training_settings("full_finetune", 2500, 512)


def test_both_training_modes_reject_unsupported_resolutions():
    with pytest.raises(ValueError, match="resolution must be 384 or 512"):
        validate_training_settings("lora", 800, 768)
    with pytest.raises(ValueError, match="resolution must be 384 or 512"):
        validate_training_settings("full_finetune", 400, 1024)


def test_full_finetune_requires_at_least_20_gb_vram():
    validate_training_vram("full_finetune", FULL_FINETUNE_MIN_VRAM_GB)
    validate_training_vram("full_finetune", 24)
    with pytest.raises(ValueError, match="requires at least 24 GB"):
        validate_training_vram("full_finetune", 16)


def test_lora_remains_available_on_smaller_cuda_gpus():
    validate_training_vram("lora", 8)
