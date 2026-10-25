"""Prompt enhancement and size presets for Stable Diffusion 1.5 image generation.

Stable Diffusion 1.5 has a short CLIP text context, so key composition terms are
placed first. The original user prompt is retained after the small preset prefix.
"""
from __future__ import annotations

import re

IMAGE_DIMENSIONS = {
    "portrait": (512, 768),
    "square": (512, 512),
    "landscape": (768, 512),
}

IMAGE_PRESETS = {"standard", "photorealistic", "full_body_photo"}
NUDE_TERMS = re.compile(
    r"\b(?:nude|nudity|naked|unclothed|undressed|fine[ -]?art nude|without clothes|no clothes)\b",
    re.IGNORECASE,
)

_PHOTO_PREFIX = (
    "photorealistic real-life photograph, natural skin texture, visible fine detail, "
    "lifelike facial features, realistic lighting and shadows, natural skin tones, "
)
_FULL_BODY_PREFIX = (
    "photorealistic full-body fine-art photograph of an adult subject, "
    "entire subject visible from head to feet, both feet fully inside the frame, "
    "subject centered with space above the head and below the feet, camera pulled back, "
    "natural anatomy and realistic skin texture, "
)
_NUDE_FULL_BODY_PREFIX = (
    "photorealistic non-sexual fine-art nude photograph of an adult woman, completely nude, "
    "full-body head-to-toe composition with feet visible, centered subject, camera pulled back, "
    "natural anatomy and realistic skin texture, "
)
_PHOTO_NEGATIVE = (
    "cartoon, anime, illustration, CGI, 3D render, plastic skin, airbrushed skin, "
    "doll-like face, blurry face, distorted face, deformed anatomy, malformed hands, "
    "extra fingers, extra arms, extra limbs, duplicate body parts"
)
_FULL_BODY_NEGATIVE = (
    "cropped body, cut off feet, cut off head, close-up, headshot, waist-up, half-body, "
    "portrait crop, out of frame"
)
_CLOTHING_NEGATIVE = (
    "clothes, clothing, dress, shirt, trousers, lingerie, underwear, bra, panties, "
    "bikini, swimsuit, robe"
)


def resolve_image_dimensions(aspect_ratio: str = "portrait") -> tuple[int, int]:
    """Return a supported SD 1.5 canvas size for the requested composition."""
    key = (aspect_ratio or "portrait").strip().lower()
    try:
        return IMAGE_DIMENSIONS[key]
    except KeyError as exc:
        raise ValueError("Image aspect ratio must be portrait, square or landscape.") from exc


def validate_image_preset(preset: str = "full_body_photo") -> str:
    key = (preset or "full_body_photo").strip().lower()
    if key not in IMAGE_PRESETS:
        raise ValueError("Image preset must be standard, photorealistic or full_body_photo.")
    return key


def build_generation_prompts(
    prompt: str,
    negative_prompt: str = "",
    preset: str = "full_body_photo",
) -> tuple[str, str]:
    """Add concise prompt terms that prioritize photo realism and framing.

    If the user's prompt requests nudity, front-load adult fine-art nude wording
    and explicitly discourage accidental clothing. No text is added in standard mode.
    """
    key = validate_image_preset(preset)
    clean_prompt = (prompt or "").strip()
    clean_negative = (negative_prompt or "").strip()
    if key == "standard":
        return clean_prompt, clean_negative

    if key == "photorealistic":
        prefix = _PHOTO_PREFIX
        negative_prefix = _PHOTO_NEGATIVE
    else:
        is_nude_request = bool(NUDE_TERMS.search(clean_prompt))
        prefix = _NUDE_FULL_BODY_PREFIX if is_nude_request else _FULL_BODY_PREFIX
        negative_prefix = _FULL_BODY_NEGATIVE + ", " + _PHOTO_NEGATIVE
        if is_nude_request:
            # Put clothing terms first so the short SD 1.5 negative context sees them.
            negative_prefix = _CLOTHING_NEGATIVE + ", " + negative_prefix

    enhanced_prompt = prefix + clean_prompt if clean_prompt else prefix.rstrip(", ")
    enhanced_negative = ", ".join(part for part in (negative_prefix, clean_negative) if part)
    return enhanced_prompt, enhanced_negative
