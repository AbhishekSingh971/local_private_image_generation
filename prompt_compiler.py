"""Deterministic structured prompt composition with instruction-aware scene guidance."""
from __future__ import annotations

from collections.abc import Mapping

from instruction_policy import build_instruction_guidance

FIELD_LABELS = {
    "style": "Visual style",
    "composition": "Composition",
    "camera": "Camera and lens",
    "lighting": "Lighting",
    "pose": "Pose",
    "expression": "Expression",
    "body_shape": "Body shape",
    "environment": "Environment",
}

ALLOWED_VALUES = {
    "style": {"photorealistic", "cinematic", "illustration", "anime", "3d cgi", "oil painting", "watercolor", "pencil sketch"},
    "composition": {"close-up", "headshot", "waist-up", "three-quarter body", "full-body", "wide scene"},
    "camera": {"portrait 85mm", "50mm natural perspective", "35mm environmental portrait", "wide-angle", "macro detail", "low-angle camera", "eye-level camera"},
    "lighting": {"natural daylight", "soft studio lighting", "golden hour", "dramatic chiaroscuro", "neon lighting", "overcast soft light", "rim lighting"},
    "expression": {"neutral expression", "gentle smile", "broad smile", "laughing", "serious expression", "surprised expression", "confident expression"},
}


def build_structured_prompt(
    prompt: str,
    controls: Mapping | None = None,
    *,
    has_reference_image: bool = False,
    has_trained_identity: bool = False,
) -> str:
    """Append explicit controls and safe defaults without overriding the user's request."""
    original = str(prompt or "")
    if not original.strip():
        return original
    controls = controls or {}
    parts = [original]
    for key, label in FIELD_LABELS.items():
        value = str(controls.get(key) or "").strip()
        if not value:
            continue
        value = " ".join(value.split())
        allowed = ALLOWED_VALUES.get(key)
        if allowed is not None and value.casefold() not in allowed:
            raise ValueError(f"Unsupported {label.lower()} selection.")
        if key == "pose":
            if len(value) > 240:
                raise ValueError("Pose description must be 240 characters or fewer.")
        elif key == "body_shape":
            if len(value) > 180:
                raise ValueError("Body-shape description must be 180 characters or fewer.")
        elif key == "environment":
            if len(value) > 240:
                raise ValueError("Environment description must be 240 characters or fewer.")
        parts.append(f"{label}: {value}.")

    # These are guidance, not hard constraints: explicit user edits take priority.
    guidance = build_instruction_guidance(
        original,
        has_reference_image=has_reference_image,
        has_trained_identity=has_trained_identity,
    )
    if guidance:
        parts.append(guidance)
    return " ".join(parts)
