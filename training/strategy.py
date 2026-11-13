"""Explicit training-method selection and capability-aware planning.

The existing train_face_lora.py remains the production LoRA entry point.
Full fine-tuning is represented as a distinct strategy and is intentionally
blocked until a separately validated full-finetuning trainer is installed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


TRAINING_METHODS = {"lora", "full_finetune"}


@dataclass(frozen=True)
class TrainingPlan:
    method: str
    base_model: str
    profile_id: str | None
    dataset_path: str
    trainer_entrypoint: str | None
    trainable_scope: str
    estimated_resource_class: str
    executable: bool
    blockers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_training_plan(
    method: str,
    *,
    base_model: str,
    profile_id: str | None,
    dataset_path: str,
    cuda_available: bool,
    dataset_image_count: int,
    full_finetune_trainer_available: bool = False,
) -> TrainingPlan:
    choice = (method or "").strip().lower()
    if choice not in TRAINING_METHODS:
        raise ValueError("Training method must be 'lora' or 'full_finetune'.")
    if not base_model or not dataset_path:
        raise ValueError("base_model and dataset_path are required.")

    blockers: list[str] = []
    if not cuda_available:
        blockers.append("An NVIDIA CUDA GPU is required by the current training pipeline.")
    if dataset_image_count < 5:
        blockers.append("At least five validated training images are required.")

    if choice == "lora":
        if not profile_id:
            blockers.append("A face profile ID is required for isolated identity LoRA training.")
        return TrainingPlan(
            method=choice,
            base_model=base_model,
            profile_id=profile_id,
            dataset_path=dataset_path,
            trainer_entrypoint="train_face_lora.py",
            trainable_scope="UNet attention LoRA adapters; base weights frozen",
            estimated_resource_class="lower_than_full_finetune",
            executable=not blockers,
            blockers=tuple(blockers),
        )

    if not full_finetune_trainer_available:
        blockers.append(
            "Full fine-tuning trainer is not implemented/registered yet; this plan "
            "must not launch the LoRA trainer as a substitute."
        )
    return TrainingPlan(
        method=choice,
        base_model=base_model,
        profile_id=profile_id,
        dataset_path=dataset_path,
        trainer_entrypoint="training/trainers/full_finetune_trainer.py"
        if full_finetune_trainer_available else None,
        trainable_scope="Explicitly selected full UNet and optional text encoder weights",
        estimated_resource_class="high_vram_and_storage",
        executable=not blockers,
        blockers=tuple(blockers),
    )
