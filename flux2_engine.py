"""FLUX.2 Klein 4B inference adapter.

The distilled FLUX.2 Klein 4B checkpoint supports text-to-image and reference
image editing in one pipeline. This adapter sends the compiled prompt directly
to the model; automatic caption rewriting is deliberately disabled so the
application does not silently change user instructions.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from generation_contracts import ensure_token_capacity

_PIPELINE = None
_PIPELINE_SOURCE = None
_LOAD_LOCK = threading.RLock()
_INFERENCE_LOCK = threading.Lock()
MAX_PROMPT_TOKENS = 512


def load_flux2_pipeline(source: str | Path | None = None, *, device: str = "cuda"):
    """Load the 4B distilled pipeline with CPU offload for supported CUDA cards."""
    global _PIPELINE, _PIPELINE_SOURCE

    if device != "cuda":
        raise RuntimeError("FLUX.2 Klein requires a CUDA GPU. Select a configured local or remote GPU worker.")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Connect a supported NVIDIA GPU worker to run FLUX.2 Klein.")

    if source is None:
        from model_registry import model_configuration
        source = model_configuration("flux2_klein")["source"]

    cache_key = str(source)
    with _LOAD_LOCK:
        if _PIPELINE is not None and _PIPELINE_SOURCE == cache_key:
            return _PIPELINE

        _PIPELINE = None
        _PIPELINE_SOURCE = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        try:
            from diffusers import Flux2KleinPipeline
        except ImportError as exc:
            raise RuntimeError(
                "FLUX.2 Klein requires a Diffusers release that includes Flux2KleinPipeline."
            ) from exc

        dtype = torch.bfloat16 if getattr(torch.cuda, "is_bf16_supported", lambda: False)() else torch.float16
        try:
            pipeline = Flux2KleinPipeline.from_pretrained(cache_key, torch_dtype=dtype)
            pipeline.enable_model_cpu_offload()
        except Exception as exc:
            message = str(exc)
            if "gated" in message.lower() or "401" in message or "403" in message:
                raise RuntimeError(
                    "Hugging Face access to FLUX.2 Klein was denied. Accept the model terms and configure HF_TOKEN "
                    "in the application/worker environment; never store the token in Git."
                ) from exc
            raise

        _PIPELINE = pipeline
        _PIPELINE_SOURCE = cache_key
        return _PIPELINE


def _prompt_token_count(pipeline, prompt: str) -> int:
    """Count with the model tokenizer before inference; never let it silently clip."""
    tokenizer = getattr(pipeline, "tokenizer", None)
    if tokenizer is None or not callable(tokenizer):
        raise RuntimeError(
            "Could not inspect the FLUX.2 prompt tokenizer on this Diffusers build. "
            "Generation was stopped to avoid silently truncating your prompt."
        )
    encoded = tokenizer(prompt, add_special_tokens=True, truncation=False)
    input_ids = encoded.get("input_ids") if hasattr(encoded, "get") else None
    if input_ids is None:
        raise RuntimeError("FLUX.2 tokenizer did not return input_ids; prompt length could not be verified.")
    if input_ids and isinstance(input_ids[0], list):
        count = len(input_ids[0])
    else:
        count = len(input_ids)
    return ensure_token_capacity(count, MAX_PROMPT_TOKENS, "FLUX.2 prompt")


def generate_flux2_image(
    prompt: str,
    *,
    image=None,
    seed: int = 0,
    width: int | None = None,
    height: int | None = None,
    source: str | Path | None = None,
) -> dict[str, Any]:
    """Generate or edit an image with the original compiled prompt unchanged."""
    clean_prompt = str(prompt or "")
    if not clean_prompt.strip():
        raise ValueError("Prompt is required.")

    import torch

    pipeline = load_flux2_pipeline(source, device="cuda")
    token_count = _prompt_token_count(pipeline, clean_prompt)
    kwargs: dict[str, Any] = {
        "prompt": clean_prompt,
        "generator": torch.Generator(device="cuda").manual_seed(int(seed)),
        "num_inference_steps": 4,
        "guidance_scale": 1.0,
        "max_sequence_length": MAX_PROMPT_TOKENS,
    }
    if image is not None:
        kwargs["image"] = image
    else:
        if width is not None:
            kwargs["width"] = int(width)
        if height is not None:
            kwargs["height"] = int(height)

    # The distilled pipeline has no negative_prompt or strength input. Keep the
    # exact prompt path and make concurrent access explicit.
    if not _INFERENCE_LOCK.acquire(blocking=False):
        raise RuntimeError("FLUX.2 Klein is busy with another generation. Retry shortly.")
    try:
        with torch.inference_mode():
            output = pipeline(**kwargs)
    finally:
        _INFERENCE_LOCK.release()

    images = getattr(output, "images", None)
    if not images:
        raise RuntimeError("FLUX.2 Klein returned no image.")
    return {
        "image": images[0],
        "model_family": "flux2_klein",
        "prompt_token_count": token_count,
    }
