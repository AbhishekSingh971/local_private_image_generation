"""Shared validation for AnimateDiff request settings."""
from __future__ import annotations


def validate_video_settings(*, steps: int, frames: int, fps: int, seed: int):
    try:
        steps, frames, fps, seed = int(steps), int(frames), int(fps), int(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("Video steps, frames, FPS and seed must be integers.") from exc
    if not 5 <= steps <= 20:
        raise ValueError("Video inference steps must be between 5 and 20.")
    if frames not in (8, 16, 24):
        raise ValueError("Video frame count must be 8, 16 or 24.")
    if fps not in (6, 8, 12):
        raise ValueError("Video FPS must be 6, 8 or 12.")
    if not 0 <= seed <= 2147483647:
        raise ValueError("Video seed must be between 0 and 2147483647.")
    return {"steps": steps, "frames": frames, "fps": fps, "seed": seed}
