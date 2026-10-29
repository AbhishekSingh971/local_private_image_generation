"""Prompt enhancement and size presets for Stable Diffusion 1.5.

The user's prompt is authoritative: presets may add realism and framing hints, but
must not silently change the requested visual style or override a supplied reference
photo's composition.
"""
from __future__ import annotations

import re
import secrets

IMAGE_DIMENSIONS = {
    "portrait": (512, 768),
    "square": (512, 512),
    "landscape": (768, 512),
}

IMAGE_PRESETS = {"standard", "photorealistic", "full_body_photo"}
MAX_GENERATION_SEED = (2 ** 31) - 1


def resolve_generation_seed(seed: object = None) -> int:
    """Return a safe 31-bit seed, creating one when the caller leaves it blank."""
    if seed is None or str(seed).strip() == "":
        return secrets.randbelow(MAX_GENERATION_SEED + 1)
    try:
        value = int(str(seed).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError("Seed must be an integer between 0 and 2147483647.") from exc
    if not 0 <= value <= MAX_GENERATION_SEED:
        raise ValueError("Seed must be an integer between 0 and 2147483647.")
    return value


# Explicit style requests always win over a photo-oriented preset.
_NON_PHOTO_STYLE = re.compile(
    r"\b(?:anime|manga|cartoon|comic(?:[ -]?book)?|illustration|illustrated|"
    r"drawing|sketch|watercolor|watercolour|oil painting|acrylic painting|"
    r"painting|digital art|pixel art|3d render|cgi|claymation|stop motion)\b",
    re.IGNORECASE,
)
_EXPLICIT_FRAMING = re.compile(
    r"\b(?:extreme close[ -]?up|close[ -]?up|headshot|bust shot|waist[ -]?up|"
    r"half[ -]?body|full[ -]?body|head[ -]?to[ -]?toe|wide shot|long shot|"
    r"macro shot|aerial view|overhead view|landscape composition|portrait crop)\b",
    re.IGNORECASE,
)
_HUMAN_SUBJECT = re.compile(
    r"\b(?:person|people|woman|women|man|men|girl|boy|model|portrait|face|"
    r"character|subject|bride|groom|actor|actress|dancer|athlete)\b",
    re.IGNORECASE,
)

_PHOTO_PREFIX = (
    "photorealistic photograph, natural skin and material texture, lifelike details, "
    "realistic lighting and shadows, natural colors, "
)
_FULL_BODY_PREFIX = (
    "photorealistic full-body photograph, entire person visible from head to feet, "
    "feet inside the frame, centered subject with space above the head and below the feet, "
    "camera pulled back, natural anatomy and realistic textures, "
)
_PHOTO_NEGATIVE = (
    "blurry, low detail, distorted face, asymmetrical eyes, deformed anatomy, "
    "missing limbs, missing arms, missing shoulders, extra limbs, duplicate body parts, "
    "malformed hands, fused fingers, extra fingers, missing fingers, twisted joints, "
    "disconnected limbs, unnatural torso, distorted hips, unnatural body proportions"
)
_FULL_BODY_NEGATIVE = (
    "cropped body, cut off feet, cut off head, out of frame, incomplete body, "
    "missing arm, missing leg, missing shoulder, extra arm, extra leg"
)
_ANATOMY_DETAIL = (
    "anatomically coherent adult human body, complete shoulders and limbs, "
    "natural joint placement, balanced shoulders and hips, coherent torso and pelvis, "
    "realistic facial structure, symmetrical eyes, natural skin texture, "
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
    has_reference_image: bool = False,
) -> tuple[str, str]:
    """Enhance realism without overriding explicit prompt intent or reference framing.

    The full-body preset adds head-to-toe framing only when the prompt describes a
    person, specifies no other composition, and no reference photo is supplied.
    Explicit art styles are left as requested. Nudity wording is never added
    automatically; the user's prompt and negative prompt remain authoritative.
    """
    key = validate_image_preset(preset)
    clean_prompt = (prompt or "").strip()
    clean_negative = (negative_prompt or "").strip()
    if key == "standard" or not clean_prompt:
        return clean_prompt, clean_negative

    if _NON_PHOTO_STYLE.search(clean_prompt):
        return clean_prompt, clean_negative

    is_human = bool(_HUMAN_SUBJECT.search(clean_prompt))
    use_full_body_framing = (
        key == "full_body_photo"
        and not has_reference_image
        and is_human
        and not _EXPLICIT_FRAMING.search(clean_prompt)
    )
    explicit_full_body = bool(re.search(
        r"\\b(?:full[ -]?body|head[ -]?to[ -]?toe|entire body|full length)\\b",
        clean_prompt, re.IGNORECASE,
    ))
    use_full_body_constraints = (
        key == "full_body_photo" and not has_reference_image and is_human
        and (use_full_body_framing or explicit_full_body)
    )

    prefix = _FULL_BODY_PREFIX if use_full_body_framing else _PHOTO_PREFIX
    if key in {"photorealistic", "full_body_photo"} and is_human:
        prefix += _ANATOMY_DETAIL
    negative_prefix = _PHOTO_NEGATIVE
    if use_full_body_constraints:
        negative_prefix = _FULL_BODY_NEGATIVE + ", " + negative_prefix

    enhanced_prompt = clean_prompt if clean_prompt.lower().startswith(prefix.rstrip(", ").lower()) else prefix + clean_prompt
    enhanced_negative = ", ".join(part for part in (negative_prefix, clean_negative) if part)
    return enhanced_prompt, enhanced_negative
