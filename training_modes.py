"""Validated training-mode choices shared by the web app and GPU worker."""

from __future__ import annotations

TRAINING_MODE_LORA = "lora"
TRAINING_MODE_FULL_FINETUNE = "full_finetune"
SUPPORTED_TRAINING_MODES = (TRAINING_MODE_LORA, TRAINING_MODE_FULL_FINETUNE)
FULL_FINETUNE_MIN_VRAM_GB = 24.0


def resolve_training_mode(value: str | None = None) -> str:
    mode = str(value or TRAINING_MODE_LORA).strip().lower()
    if mode not in SUPPORTED_TRAINING_MODES:
        raise ValueError("Training method must be LoRA or Full U-Net fine-tuning.")
    return mode


def validate_training_settings(mode: str, steps: int, resolution: int) -> tuple[int, int]:
    training_mode = resolve_training_mode(mode)
    try:
        step_count = int(steps)
        image_resolution = int(resolution)
    except (TypeError, ValueError) as exc:
        raise ValueError("Training steps and resolution must be whole numbers.") from exc

    maximum_steps = 3000 if training_mode == TRAINING_MODE_LORA else 2000
    if not 100 <= step_count <= maximum_steps:
        raise ValueError(
            f"{training_mode.replace('_', ' ').title()} training steps must be between 100 and {maximum_steps}."
        )
    if image_resolution not in (384, 512):
        raise ValueError("Training resolution must be 384 or 512.")
    return step_count, image_resolution


def validate_training_vram(mode: str, vram_gb: float) -> None:
    training_mode = resolve_training_mode(mode)
    try:
        available = float(vram_gb)
    except (TypeError, ValueError) as exc:
        raise ValueError("Could not determine the selected GPU's VRAM.") from exc
    if training_mode == TRAINING_MODE_FULL_FINETUNE and available < FULL_FINETUNE_MIN_VRAM_GB:
        raise ValueError(
            "Full U-Net fine-tuning requires at least "
            f"{FULL_FINETUNE_MIN_VRAM_GB:.0f} GB of GPU VRAM. "
            f"The selected GPU reports {available:.1f} GB. Choose LoRA or rent a larger GPU."
        )
