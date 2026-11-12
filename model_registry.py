"""Explicit image-model family registry.

Only model families with a matching Diffusers pipeline are selectable. Identity
LoRAs are family-specific; the existing face trainer currently produces SD 1.5
adapters, so SDXL is offered for base-model generation until an SDXL trainer and
validated SDXL identity adapters are added.
"""
from __future__ import annotations

import os
from pathlib import Path

SUPPORTED_MODEL_FAMILIES = ("sd15", "sdxl", "flux3_klein")

_MODEL_CONFIG = {
    "sd15": {
        "label": "Stable Diffusion 1.5",
        "default_model_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "model_id_env": "SD_MODEL_ID",
        "model_path_env": "SD_MODEL_PATH",
        "default_local_path": Path(__file__).resolve().parent / "models" / "stable-diffusion-v1-5",
        "supports_face_lora": True,
        "supports_pose_control": True,
        "minimum_recommended_vram_gb": 6,
    },
    "sdxl": {
        "label": "Stable Diffusion XL 1.0 (base)",
        "default_model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "model_id_env": "SDXL_MODEL_ID",
        "model_path_env": "SDXL_MODEL_PATH",
        "default_local_path": Path(__file__).resolve().parent / "models" / "stable-diffusion-xl-base-1.0",
        "supports_face_lora": False,
        "supports_pose_control": False,
        "minimum_recommended_vram_gb": 12,
    },
    "flux3_klein": {
        "label": "FLUX 3 Image Klein (distilled)",
        "default_model_id": "black-forest-labs/flux-3-image-klein",
        "model_id_env": "FLUX3_KLEIN_MODEL_ID",
        "model_path_env": "FLUX3_KLEIN_MODEL_PATH",
        "default_local_path": Path(__file__).resolve().parent / "models" / "flux-3-image-klein",
        "supports_face_lora": False,
        "supports_body_lora": False,
        "supports_pose_control": False,
        "supports_negative_prompt": False,
        "supports_image_strength": False,
        "supports_prompt_upsampling": True,
        "minimum_recommended_vram_gb": 24,
        "license": "FLUX Kommunity License v1.0; commercial use is conditional",
    },
}


def resolve_model_family(value: str | None = None) -> str:
    """Resolve and validate a model family without accepting arbitrary model IDs."""
    family = (value or os.environ.get("IMAGE_MODEL_FAMILY", "sd15")).strip().lower()
    if family not in SUPPORTED_MODEL_FAMILIES:
        choices = ", ".join(SUPPORTED_MODEL_FAMILIES)
        raise ValueError(f"Unsupported image model family '{family}'. Choose one of: {choices}.")
    return family


def model_configuration(family: str | None = None) -> dict:
    """Return a copy of the trusted configuration for one supported family."""
    key = resolve_model_family(family)
    config = dict(_MODEL_CONFIG[key])
    model_id = os.environ.get(config["model_id_env"], "").strip()
    config["model_id"] = model_id or config["default_model_id"]
    path_override = os.environ.get(config["model_path_env"], "").strip()
    config["path_override"] = path_override
    if path_override:
        config["source"] = Path(path_override).expanduser()
    elif config["default_local_path"].is_dir() and (config["default_local_path"] / "model_index.json").is_file():
        config["source"] = config["default_local_path"]
    elif key == "sd15" and os.environ.get("SD_MODEL_ID", "").strip():
        config["source"] = config["model_id"]
    else:
        config["source"] = config["model_id"]
    return config


def supports_face_lora(family: str | None = None) -> bool:
    return bool(model_configuration(family)["supports_face_lora"])


def model_family_label(family: str | None = None) -> str:
    return str(model_configuration(family)["label"])
