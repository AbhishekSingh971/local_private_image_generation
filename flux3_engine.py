"""FLUX 3 Image Klein inference adapter.

This is intentionally separate from the SD 1.5/SDXL Diffusers pipelines:
FLUX 3 Klein is a modular flow-matching pipeline with its own prompt upsampler.
It uses the distilled checkpoint for generation and image-conditioned editing.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)
_PIPELINE = None
_COMPONENTS_MANAGER = None
_PIPELINE_SOURCE = None
_LOAD_LOCK = threading.RLock()
_INFERENCE_LOCK = threading.Lock()


def _resolve_dtype(torch):
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except (AttributeError, RuntimeError):
        pass
    return torch.float16


def load_flux3_pipeline(source: str | Path | None = None, *, device: str = "cuda"):
    """Load FLUX 3 Klein with automatic component CPU offload."""
    global _PIPELINE, _COMPONENTS_MANAGER, _PIPELINE_SOURCE

    if device != "cuda":
        raise RuntimeError("FLUX 3 Klein requires a CUDA GPU in this application. Select a configured GPU worker.")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Connect a supported NVIDIA GPU worker to run FLUX 3 Klein.")

    if source is None:
        from model_registry import model_configuration
        source = model_configuration("flux3_klein")["source"]

    cache_key = str(source)
    with _LOAD_LOCK:
        if _PIPELINE is not None and _PIPELINE_SOURCE == cache_key:
            return _PIPELINE

        # Release a previously loaded FLUX model before loading another source.
        _PIPELINE = None
        _COMPONENTS_MANAGER = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        try:
            from diffusers import ComponentsManager, ModularPipeline
        except ImportError as exc:
            raise RuntimeError(
                "FLUX 3 Klein requires Diffusers 0.43.0 or newer with ModularPipeline support."
            ) from exc

        manager = ComponentsManager()
        pipeline = ModularPipeline.from_pretrained(
            cache_key,
            components_manager=manager,
        )
        dtype = _resolve_dtype(torch)
        pipeline.load_components(dtype={"vae": torch.float32, "default": dtype})
        manager.enable_auto_cpu_offload(device="cuda")

        _PIPELINE = pipeline
        _COMPONENTS_MANAGER = manager
        _PIPELINE_SOURCE = cache_key
        return _PIPELINE


def generate_flux3_image(
    prompt: str,
    *,
    image=None,
    seed: int = 0,
    steps: int = 16,
    width: int | None = None,
    height: int | None = None,
    source: str | Path | None = None,
) -> dict[str, Any]:
    """Generate or edit an image and return the exact caption produced by FLUX PUP."""
    clean_prompt = str(prompt or "").strip()
    if not clean_prompt:
        raise ValueError("Prompt is required.")
    if not 4 <= int(steps) <= 16:
        raise ValueError("FLUX 3 Klein distilled generation supports 4–16 inference steps.")

    import torch

    pipeline = load_flux3_pipeline(source, device="cuda")
    kwargs: dict[str, Any] = {
        "prompt": clean_prompt,
        "generator": torch.Generator(device="cuda").manual_seed(int(seed)),
        "num_inference_steps": int(steps),
        "output": ["images", "prompt"],
        # The text encoder supports long captions; do not use the short CLIP
        # context defaults from SD pipelines. Oversized requests should be
        # rejected by the model/tokenizer rather than clipped by this app.
        "max_sequence_length": 8192,
    }
    if image is not None:
        kwargs["image"] = image
    else:
        if width is not None:
            kwargs["width"] = int(width)
        if height is not None:
            kwargs["height"] = int(height)

    # FLUX 3 Klein distilled has no negative_prompt or per-image strength input.
    # A separate lock prevents two HTTP threads from concurrently moving its
    # large modular components between CPU and GPU.
    if not _INFERENCE_LOCK.acquire(blocking=False):
        raise RuntimeError("FLUX 3 Klein is busy with another generation. Retry shortly.")
    try:
        with torch.inference_mode():
            output = pipeline(**kwargs)
    finally:
        _INFERENCE_LOCK.release()

    images = output["images"]
    if not images:
        raise RuntimeError("FLUX 3 Klein returned no image.")
    expanded_prompt = output.get("prompt") or clean_prompt
    return {
        "image": images[0],
        "upsampled_prompt": str(expanded_prompt),
        "model_family": "flux3_klein",
    }
