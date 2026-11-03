"""Validation helpers for optional pre-rendered SD 1.5 OpenPose ControlNet maps."""
from __future__ import annotations

from PIL import Image, ImageStat


def prepare_openpose_control_map(image: Image.Image, width: int, height: int) -> Image.Image:
    """Fit a user-supplied OpenPose skeleton map to the requested generation canvas.

    This deliberately accepts a rendered control map rather than guessing a body
    skeleton from a normal photograph. A compatible pose preprocessor can be
    added later without changing the generation request contract.
    """
    if image is None:
        raise ValueError("Upload an OpenPose skeleton map to use full-body pose control.")
    source = image.convert("RGB")
    if source.width < 64 or source.height < 64:
        raise ValueError("OpenPose control maps must be at least 64 × 64 pixels.")
    extrema = source.getextrema()
    if all(low == high for low, high in extrema):
        raise ValueError("The uploaded pose map is blank; choose a visible OpenPose skeleton map.")
    fitted = Image.new("RGB", (int(width), int(height)), (0, 0, 0))
    source.thumbnail((int(width), int(height)), Image.Resampling.LANCZOS)
    fitted.paste(source, ((int(width) - source.width) // 2, (int(height) - source.height) // 2))
    if sum(ImageStat.Stat(fitted).mean) < 2:
        raise ValueError("The uploaded pose map appears blank after resizing.")
    return fitted
