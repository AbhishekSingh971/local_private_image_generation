"""Conservative SD 1.5 regional inpainting helpers.

The user paints the mask. White pixels are editable, black pixels are protected.
The inpainting model sees a padded crop for context; the final composite blends
only inside a slightly expanded/feathered mask, leaving distant pixels byte-for-
byte identical to the source image.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Callable

from PIL import Image, ImageFilter


DEFAULT_INPAINT_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"
MAX_INPAINT_SOURCE_PIXELS = 40_000_000
MAX_WORK_EDGE = 768
MIN_WORK_EDGE = 512


def normalize_edit_mask(source_image: Image.Image, mask_image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int], float]:
    """Validate dimensions/coverage and return a hard binary white-means-edit mask."""
    if not isinstance(source_image, Image.Image) or not isinstance(mask_image, Image.Image):
        raise TypeError("Source and mask must be valid images.")
    source = source_image.convert("RGB")
    if source.width < 128 or source.height < 128:
        raise ValueError("Source image must be at least 128 × 128 pixels.")
    if source.width * source.height > MAX_INPAINT_SOURCE_PIXELS:
        raise ValueError("Source image is too large for regional repair (maximum 40 megapixels).")
    if mask_image.size != source.size:
        raise ValueError("Repair mask dimensions must exactly match the source image.")
    mask = mask_image.convert("L").point(lambda value: 255 if value >= 32 else 0)
    bbox = mask.getbbox()
    if not bbox:
        raise ValueError("Paint the area you want to repair before submitting.")
    histogram = mask.histogram()
    selected_pixels = int(histogram[255])
    coverage = selected_pixels / float(source.width * source.height)
    if coverage < 0.0001:
        raise ValueError("The selected repair area is too small. Paint a slightly larger area.")
    if coverage > 0.65:
        raise ValueError("Select a smaller area (no more than 65% of the image) to preserve the rest of the scene.")
    return mask, bbox, round(coverage, 6)


def prepare_inpaint_crop(
    source_image: Image.Image,
    mask_image: Image.Image,
    *,
    padding: int = 128,
    max_edge: int = MAX_WORK_EDGE,
) -> dict:
    """Build a padded, size-bounded crop for the inpainting model."""
    source = source_image.convert("RGB")
    mask, bbox, coverage = normalize_edit_mask(source, mask_image)
    left, top, right, bottom = bbox
    region_width, region_height = right - left, bottom - top
    padding = max(32, min(int(padding), 512), int(max(region_width, region_height) * 0.35))
    crop_box = (
        max(0, left - padding),
        max(0, top - padding),
        min(source.width, right + padding),
        min(source.height, bottom + padding),
    )
    crop_source = source.crop(crop_box)
    crop_mask = mask.crop(crop_box)
    crop_width, crop_height = crop_source.size
    longest_edge = max(crop_width, crop_height)
    max_edge = max(512, min(int(max_edge), MAX_WORK_EDGE))
    if longest_edge < MIN_WORK_EDGE:
        scale = MIN_WORK_EDGE / float(longest_edge)
    elif longest_edge > max_edge:
        scale = max_edge / float(longest_edge)
    else:
        scale = 1.0
    work_width = max(64, int(round(crop_width * scale / 8.0)) * 8)
    work_height = max(64, int(round(crop_height * scale / 8.0)) * 8)
    work_width = min(work_width, max_edge)
    work_height = min(work_height, max_edge)
    work_source = crop_source.resize((work_width, work_height), Image.Resampling.LANCZOS)
    work_mask = crop_mask.resize((work_width, work_height), Image.Resampling.NEAREST)
    return {
        "source": source,
        "mask": mask,
        "coverage": coverage,
        "bbox": crop_box,
        "crop_box": crop_box,
        "mask_bbox": bbox,
        "crop_source": crop_source,
        "work_source": work_source,
        "work_mask": work_mask,
        "work_size": (work_width, work_height),
    }


def merge_inpainted_crop(
    source_image: Image.Image,
    generated_crop: Image.Image,
    *,
    crop_box: tuple[int, int, int, int],
    crop_mask: Image.Image,
    grow_px: int = 8,
    blur_px: float = 2.0,
) -> Image.Image:
    """Blend a generated patch into the painted region and preserve other pixels."""
    source = source_image.convert("RGB")
    left, top, right, bottom = crop_box
    original_crop = source.crop(crop_box)
    generated = generated_crop.convert("RGB")
    if generated.size != original_crop.size:
        generated = generated.resize(original_crop.size, Image.Resampling.LANCZOS)
    binary = crop_mask.convert("L").point(lambda value: 255 if value >= 32 else 0)
    grow_px = max(0, min(int(grow_px), 48))
    if grow_px:
        kernel = grow_px * 2 + 1
        binary = binary.filter(ImageFilter.MaxFilter(kernel))
    blur_px = max(0.0, min(float(blur_px), 16.0))
    if blur_px:
        binary = binary.filter(ImageFilter.GaussianBlur(radius=blur_px))
    merged_crop = Image.composite(generated, original_crop, binary)
    output = source.copy()
    output.paste(merged_crop, (left, top))
    return output


def build_regional_inpaint_prompts(prompt: str, negative_prompt: str = "") -> tuple[str, str]:
    """Compose local edit guidance without changing the source image's visual style."""
    prompt = str(prompt or "").strip()
    negative_prompt = str(negative_prompt or "").strip()
    safe_prompt = (
        "A localized image edit that matches the source image's existing visual style, "
        "materials, camera perspective, lighting and color grading. Replace only the white "
        "masked area according to the requested correction. Keep the unmasked surroundings "
        "and all unrelated scene details consistent with the source. Do not add unrelated objects. "
        + prompt
    )
    default_negative = (
        "unintended extra objects, duplicate parts, broken geometry, malformed anatomy, "
        "detached components, blur, low detail, visual artifacts"
    )
    safe_negative = ", ".join(part for part in (default_negative, negative_prompt) if part)
    return safe_prompt, safe_negative


def _inpaint_source():
    local_path = (os.environ.get("INPAINT_MODEL_PATH") or "").strip()
    if local_path:
        path = Path(local_path).expanduser()
        if not (path / "model_index.json").is_file():
            raise RuntimeError("INPAINT_MODEL_PATH must point to a Diffusers model directory containing model_index.json.")
        return path, True
    model_id = (os.environ.get("INPAINT_MODEL_ID") or DEFAULT_INPAINT_MODEL_ID).strip()
    if not model_id:
        raise RuntimeError("Set INPAINT_MODEL_ID to a Stable Diffusion 1.5 inpainting checkpoint.")
    return model_id, False


def run_sd15_regional_inpainting(
    source_image: Image.Image,
    mask_image: Image.Image,
    *,
    prompt: str,
    negative_prompt: str = "",
    identity_lora_path: str | Path | None = None,
    seed: int = 0,
    steps: int = 24,
    strength: float = 0.88,
    device: str = "cuda",
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[Image.Image, dict]:
    """Run one SD 1.5 inpainting pass and composite it back through the mask."""
    if device not in {"cuda", "cpu"}:
        raise ValueError("Inpainting device must be cuda or cpu.")
    prompt = str(prompt or "").strip()
    if not prompt:
        raise ValueError("Describe what should replace the painted region.")
    if len(prompt) > 1200:
        raise ValueError("Repair instruction must be 1,200 characters or fewer.")
    if len(str(negative_prompt or "")) > 1200:
        raise ValueError("Negative repair instruction must be 1,200 characters or fewer.")
    try:
        steps = int(steps)
        seed = int(seed)
        strength = float(strength)
    except (TypeError, ValueError) as exc:
        raise ValueError("Repair steps, seed and strength must be valid numbers.") from exc
    if not 8 <= steps <= 40:
        raise ValueError("Regional repair steps must be between 8 and 40.")
    if not 0 <= seed <= (2 ** 31 - 1):
        raise ValueError("Repair seed must be between 0 and 2147483647.")
    if not 0.55 <= strength <= 1.0:
        raise ValueError("Regional repair strength must be between 0.55 and 1.00.")

    import torch
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for regional inpainting. Select a connected GPU worker.")
    crop_padding = int(os.environ.get("INPAINT_CROP_PADDING_PX", "128"))
    grow_px = int(os.environ.get("INPAINT_MASK_GROW_PX", "8"))
    blur_px = float(os.environ.get("INPAINT_MASK_BLUR_PX", "2"))
    work = prepare_inpaint_crop(source_image, mask_image, padding=crop_padding)
    model_source, local_only = _inpaint_source()
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipeline = None
    try:
        from diffusers import StableDiffusionInpaintPipeline
        kwargs = {
            "torch_dtype": dtype,
            "safety_checker": None,
            "requires_safety_checker": False,
        }
        if local_only:
            kwargs["local_files_only"] = True
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(model_source, **kwargs)
        pipeline = pipeline.to(device)
        if device == "cpu":
            pipeline.enable_attention_slicing("max")
        try:
            pipeline.enable_vae_slicing()
        except Exception:
            pass

        if identity_lora_path:
            adapter_dir = Path(identity_lora_path)
            if not adapter_dir.is_dir() or not any(adapter_dir.iterdir()):
                raise RuntimeError("The selected face LoRA folder is missing or empty; refusing an identity-blind repair.")
            try:
                pipeline.load_lora_weights(str(adapter_dir), adapter_name="face_identity")
                pipeline.set_adapters(["face_identity"], adapter_weights=[1.0])
            except Exception as exc:
                raise RuntimeError(
                    "The selected face LoRA could not be loaded by the inpainting checkpoint. "
                    "The repair was stopped rather than silently dropping identity conditioning: " + str(exc)[:240]
                ) from exc

        safe_prompt, safe_negative = build_regional_inpaint_prompts(prompt, negative_prompt)
        from generation_contracts import validate_diffusion_prompt_capacity
        prompt_token_counts = validate_diffusion_prompt_capacity(pipeline, safe_prompt, safe_negative)
        total_steps = steps
        def on_step_end(_active_pipeline, step_index, _timestep, callback_kwargs):
            if progress_callback:
                progress_callback(min(step_index + 1, total_steps), total_steps)
            return callback_kwargs

        generator = torch.Generator(device=device).manual_seed(seed)
        output = pipeline(
            prompt=safe_prompt,
            negative_prompt=safe_negative,
            image=work["work_source"],
            mask_image=work["work_mask"],
            width=work["work_size"][0],
            height=work["work_size"][1],
            num_inference_steps=steps,
            guidance_scale=7.5,
            strength=strength,
            generator=generator,
            callback_on_step_end=on_step_end,
        )
        if not getattr(output, "images", None):
            raise RuntimeError("Inpainting pipeline returned no image.")
        merged = merge_inpainted_crop(
            work["source"],
            output.images[0],
            crop_box=work["crop_box"],
            crop_mask=work["mask"].crop(work["crop_box"]),
            grow_px=grow_px,
            blur_px=blur_px,
        )
        return merged, {
            "model_id": str(model_source),
            "seed": seed,
            "steps": steps,
            "strength": round(strength, 3),
            "prompt_token_counts": prompt_token_counts,
            "mask_coverage": work["coverage"],
            "mask_bbox": list(work["mask_bbox"]),
            "crop_bbox": list(work["crop_box"]),
            "crop_size": list(work["crop_source"].size),
            "inference_size": list(work["work_size"]),
            "identity_lora_applied": bool(identity_lora_path),
            "composite_mask_grow_px": grow_px,
            "composite_mask_blur_px": blur_px,
            "preservation": "Pixels outside the expanded/feathered mask are copied from the original source.",
        }
    finally:
        if pipeline is not None:
            del pipeline
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
