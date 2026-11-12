"""Prompt enhancement and size presets for supported diffusion model families.

The user's prompt is authoritative. Explicit full-body framing requests must remain
active for reference-image generation; img2img strength still controls how strongly
the source composition is preserved.
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


def resolve_image_dimensions(aspect_ratio: str = "portrait", model_family: str = "sd15") -> tuple[int, int]:
    """Return model-family-appropriate dimensions for the requested composition."""
    key = (aspect_ratio or "portrait").strip().lower()
    family = (model_family or "sd15").strip().lower()
    dimensions = {
        "sd15": IMAGE_DIMENSIONS,
        # SDXL was trained around a 1024px image area. These aspect-ratio
        # buckets are multiples of 8 and avoid the old SD 1.5-sized canvas.
        "sdxl": {
            "portrait": (832, 1216),
            "square": (1024, 1024),
            "landscape": (1216, 832),
        },
        "flux2_klein": {
            "portrait": (1024, 1536),
            "square": (1024, 1024),
            "landscape": (1536, 1024),
        },
    }
    if family not in dimensions:
        raise ValueError("Image model family must be sd15, sdxl or flux2_klein.")
    try:
        return dimensions[family][key]
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
    """Enhance realism while honoring explicit composition requests.

    A supplied reference normally keeps its original framing. However, an explicit
    full-body request is a deliberate request to reframe and must receive full-body
    framing and anti-cropping constraints even in img2img mode. The caller's
    denoising strength determines how much the source composition can change.
    Explicit art styles and user prompt/negative prompt wording remain authoritative.
    """
    key = validate_image_preset(preset)
    clean_prompt = (prompt or "").strip()
    clean_negative = (negative_prompt or "").strip()
    if key == "standard" or not clean_prompt:
        return clean_prompt, clean_negative

    if _NON_PHOTO_STYLE.search(clean_prompt):
        return clean_prompt, clean_negative

    is_human = bool(_HUMAN_SUBJECT.search(clean_prompt))
    explicit_full_body = bool(re.search(
        r"\b(?:full[ -]?body|head[ -]?to[ -]?toe|entire body|full length)\b",
        clean_prompt, re.IGNORECASE,
    ))
    # An explicit full-body instruction takes precedence over the source photo's
    # original crop. Without an explicit instruction, keep reference framing intact.
    use_full_body_framing = (
        key == "full_body_photo"
        and is_human
        and (
            explicit_full_body
            or (not has_reference_image and not _EXPLICIT_FRAMING.search(clean_prompt))
        )
    )
    use_full_body_constraints = use_full_body_framing

    prefix = _FULL_BODY_PREFIX if use_full_body_framing else _PHOTO_PREFIX
    if key in {"photorealistic", "full_body_photo"} and is_human:
        prefix += _ANATOMY_DETAIL
    # Give multi-person beach scenes explicit composition guidance. SD 1.5 often
    # merges nearby people or drops limbs when a group scene is described only
    # in natural language, so make the spatial arrangement and framing explicit.
    beach_group_scene = (
        bool(re.search(r"\bbeach\b|\bseaside\b|\bshoreline\b", clean_prompt, re.IGNORECASE))
        and bool(re.search(r"\bwoman\b|\bwomen\b|\bfemale\b", clean_prompt, re.IGNORECASE))
        and bool(re.search(r"\b(?:two|2)\s+(?:(?:adult|naked|shirtless)\s+)*(?:men|males)\b", clean_prompt, re.IGNORECASE))
    )
    if beach_group_scene:
        prefix = (
            "photorealistic wide full-body beach photograph, three separate adult people "
            "visible head to toe, woman centered with two adult men positioned separately "
            "one on each side, clear space between each person, all faces and limbs visible, "
            "tropical trees clearly visible in the background, natural daylight, "
            "realistic camera perspective, "
        ) + prefix

    negative_prefix = _PHOTO_NEGATIVE
    if use_full_body_constraints:
        negative_prefix = _FULL_BODY_NEGATIVE + ", " + negative_prefix
    if beach_group_scene:
        negative_prefix += (
            ", extra people, duplicated people, merged bodies, fused torsos, overlapping "
            "limbs, hidden arms, hidden legs, cropped feet, cropped heads, out of frame, "
            "extra hands, malformed fingers, distorted faces, text, watermark"
        )

    enhanced_prompt = clean_prompt if clean_prompt.lower().startswith(prefix.rstrip(", ").lower()) else prefix + clean_prompt
    enhanced_negative = ", ".join(part for part in (negative_prefix, clean_negative) if part)
    return enhanced_prompt, enhanced_negative
