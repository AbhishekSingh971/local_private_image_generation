"""Conservative technical quality gates for generated photographs.

This module detects obvious technical failures; it does not certify identity,
anatomical correctness, or professional suitability. Those require separate
evaluators and, for important work, human review.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ImageQualityReport:
    status: str
    reasons: tuple[str, ...]
    width: int
    height: int
    sharpness_score: float
    mean_luminance: float
    clipped_dark_fraction: float
    clipped_bright_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_image_quality(
    image: Image.Image,
    *,
    min_width: int = 384,
    min_height: int = 384,
    min_sharpness: float = 45.0,
    min_mean_luminance: float = 18.0,
    max_mean_luminance: float = 238.0,
    max_clipped_fraction: float = 0.30,
) -> ImageQualityReport:
    """Flag low-resolution, blurry, or badly exposed images.

    The sharpness score is a Laplacian-response variance heuristic, not a
    universal perceptual metric. Calibrate thresholds against real outputs.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image")
    rgb = image.convert("RGB")
    width, height = rgb.size
    gray = np.asarray(rgb.convert("L"), dtype=np.float32)
    if gray.size == 0:
        raise ValueError("image is empty")

    if min(gray.shape) >= 3:
        center = gray[1:-1, 1:-1]
        laplacian = (
            gray[:-2, 1:-1] + gray[2:, 1:-1]
            + gray[1:-1, :-2] + gray[1:-1, 2:]
            - 4.0 * center
        )
        sharpness = float(np.var(laplacian))
    else:
        sharpness = 0.0
    mean_luminance = float(np.mean(gray))
    clipped_dark = float(np.mean(gray <= 5.0))
    clipped_bright = float(np.mean(gray >= 250.0))

    reasons: list[str] = []
    if width < min_width or height < min_height:
        reasons.append(f"Resolution is {width}×{height}; minimum is {min_width}×{min_height}.")
    if sharpness < min_sharpness:
        reasons.append("Image may be blurry or overly smooth; inspect sharpness at 100% zoom.")
    if mean_luminance < min_mean_luminance:
        reasons.append("Image is very dark; inspect exposure and facial detail.")
    elif mean_luminance > max_mean_luminance:
        reasons.append("Image is very bright; inspect exposure and highlight detail.")
    if clipped_dark > max_clipped_fraction:
        reasons.append("A large portion of the image is nearly black.")
    if clipped_bright > max_clipped_fraction:
        reasons.append("A large portion of the image is nearly white.")

    return ImageQualityReport(
        status="needs_review" if reasons else "technical_checks_passed",
        reasons=tuple(reasons),
        width=width,
        height=height,
        sharpness_score=round(sharpness, 3),
        mean_luminance=round(mean_luminance, 3),
        clipped_dark_fraction=round(clipped_dark, 4),
        clipped_bright_fraction=round(clipped_bright, 4),
    )


def assess_image_bytes(data: bytes, **kwargs: Any) -> ImageQualityReport:
    """Decode and assess image bytes, rejecting corrupt or unsupported data."""
    if not data:
        raise ValueError("image data is empty")
    try:
        with Image.open(BytesIO(data)) as opened:
            opened.load()
            return assess_image_quality(opened, **kwargs)
    except (OSError, ValueError) as exc:
        raise ValueError("image data is not a valid supported image") from exc
