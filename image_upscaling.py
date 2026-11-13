"""Optional GPU-backed x4 image super-resolution using Diffusers.

This is a real generative upscaler, not a Lanczos resize. It is opt-in because
it downloads a separate checkpoint, uses additional GPU memory and may alter
fine details. Keep the original generation as the fallback if enhancement fails.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

DEFAULT_UPSCALER_MODEL = "stabilityai/stable-diffusion-x4-upscaler"
MAX_INPUT_EDGE = 768
DEFAULT_STEPS = 20
MIN_STEPS = 10
MAX_STEPS = 30

_PIPELINE = None
_PIPELINE_SOURCE: str | None = None
_LOAD_LOCK = threading.RLock()
_INFERENCE_LOCK = threading.Lock()


def resolve_upscale_options(enabled: object, steps: object = DEFAULT_STEPS) -> dict[str, Any]:
    """Validate enhancement options without importing torch or downloading models."""
    active = str(enabled or "").strip().lower() in {"1", "true", "yes", "on"}
    if not active:
        count = DEFAULT_STEPS
    else:
        try:
            count = int(steps if steps not in (None, "") else DEFAULT_STEPS)
        except (TypeError, ValueError) as exc:
            raise ValueError("AI upscaling steps must be an integer.") from exc
        if not MIN_STEPS <= count <= MAX_STEPS:
            raise ValueError(f"AI upscaling steps must be between {MIN_STEPS} and {MAX_STEPS}.")
    model_id = (str(__import__("os").environ.get("IMAGE_UPSCALER_MODEL_ID") or "").strip()
                or DEFAULT_UPSCALER_MODEL)
    return {"enabled": active, "steps": count, "model_id": model_id, "scale_factor": 4}


def _load_pipeline(source: str | Path | None = None):
    global _PIPELINE, _PIPELINE_SOURCE
    import os
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("AI x4 upscaling requires CUDA. Use a configured NVIDIA GPU worker.")
    model_source = str(source or os.environ.get("IMAGE_UPSCALER_MODEL_ID") or DEFAULT_UPSCALER_MODEL)
    with _LOAD_LOCK:
        if _PIPELINE is not None and _PIPELINE_SOURCE == model_source:
            return _PIPELINE
        _PIPELINE = None
        _PIPELINE_SOURCE = None
        torch.cuda.empty_cache()
        try:
            from diffusers import StableDiffusionUpscalePipeline
        except ImportError as exc:
            raise RuntimeError("Install a Diffusers version that provides StableDiffusionUpscalePipeline.") from exc
        dtype = torch.float16
        try:
            pipeline = StableDiffusionUpscalePipeline.from_pretrained(
                model_source, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
            )
            pipeline.enable_model_cpu_offload()
            if hasattr(pipeline, "enable_vae_slicing"):
                pipeline.enable_vae_slicing()
        except Exception as exc:
            message = str(exc)
            if "401" in message or "403" in message or "gated" in message.lower():
                raise RuntimeError(
                    "Access to the configured upscaler checkpoint was denied. Accept its model terms and configure HF_TOKEN."
                ) from exc
            raise
        _PIPELINE = pipeline
        _PIPELINE_SOURCE = model_source
        return pipeline


def upscale_image(
    image,
    prompt: str,
    *,
    negative_prompt: str = "",
    seed: int = 0,
    steps: int = DEFAULT_STEPS,
    source: str | Path | None = None,
    cancelled=None,
    progress_callback=None,
):
    """Generatively upscale an image by approximately 4x on a CUDA device."""
    from PIL import Image

    if not isinstance(image, Image.Image):
        raise TypeError("AI upscaling requires a PIL image.")
    if not str(prompt or "").strip():
        raise ValueError("A descriptive prompt is required for AI upscaling.")
    options = resolve_upscale_options(True, steps)
    rgb = image.convert("RGB")
    width, height = rgb.size
    edge = max(width, height)
    if edge > MAX_INPUT_EDGE:
        ratio = MAX_INPUT_EDGE / edge
        rgb = rgb.resize(
            (max(64, round(width * ratio)), max(64, round(height * ratio))),
            Image.Resampling.LANCZOS,
        )
    import torch
    pipeline = _load_pipeline(source or options["model_id"])
    if not _INFERENCE_LOCK.acquire(blocking=False):
        raise RuntimeError("The AI upscaler is busy with another job. Retry shortly.")
    try:
        if cancelled and cancelled():
            raise RuntimeError("AI upscaling was cancelled before inference.")
        generator = torch.Generator(device="cuda").manual_seed(int(seed))
        def on_step_end(active_pipeline, step_index, _timestep, callback_kwargs):
            if cancelled and cancelled():
                raise RuntimeError("AI upscaling was cancelled by the user.")
            if progress_callback:
                progress_callback(step_index + 1, options["steps"])
            return callback_kwargs
        with torch.inference_mode():
            result = pipeline(
                prompt=str(prompt),
                image=rgb,
                num_inference_steps=options["steps"],
                guidance_scale=7.5,
                negative_prompt=str(negative_prompt or "") or None,
                generator=generator,
                callback_on_step_end=on_step_end,
            )
        if cancelled and cancelled():
            raise RuntimeError("AI upscaling was cancelled after inference.")
        images = getattr(result, "images", None)
        if not images or not isinstance(images[0], Image.Image):
            raise RuntimeError("The AI upscaler returned no valid image.")
        return images[0]
    finally:
        _INFERENCE_LOCK.release()
