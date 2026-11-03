"""Duration-aware AnimateDiff request validation and segmented MP4 assembly."""
from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path
from typing import Callable


VIDEO_QUALITY_PRESETS = {
    "fast": {"steps": 8, "guidance_scale": 6.8, "label": "Fast"},
    "balanced": {"steps": 12, "guidance_scale": 7.0, "label": "Balanced"},
    "high": {"steps": 16, "guidance_scale": 7.0, "label": "High"},
    "maximum": {"steps": 20, "guidance_scale": 7.0, "label": "Maximum"},
}
MAX_VIDEO_DURATION_SECONDS = 60
MAX_ANIMATEDIFF_CHUNK_FRAMES = 24
MIN_ANIMATEDIFF_CHUNK_FRAMES = 8


def resolve_video_request(*, duration_seconds=3, fps=8, seed=0, quality="balanced"):
    """Validate user-facing settings and resolve a supported quality preset."""
    try:
        duration_seconds = int(duration_seconds)
        fps = int(fps)
        seed = int(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("Video duration, FPS and seed must be whole numbers.") from exc
    quality = str(quality or "balanced").strip().lower()
    if not 1 <= duration_seconds <= MAX_VIDEO_DURATION_SECONDS:
        raise ValueError("Video duration must be between 1 and 60 seconds.")
    if fps not in (6, 8, 12):
        raise ValueError("Video FPS must be 6, 8 or 12.")
    if not 0 <= seed <= 2147483647:
        raise ValueError("Video seed must be between 0 and 2147483647.")
    if quality not in VIDEO_QUALITY_PRESETS:
        raise ValueError("Choose Fast, Balanced, High or Maximum video quality.")
    preset = VIDEO_QUALITY_PRESETS[quality]
    return {
        "duration_seconds": duration_seconds,
        "fps": fps,
        "seed": seed,
        "quality": quality,
        "steps": preset["steps"],
        "guidance_scale": preset["guidance_scale"],
        "total_frames": duration_seconds * fps,
        "max_chunk_frames": MAX_ANIMATEDIFF_CHUNK_FRAMES,
        "segments": max(1, math.ceil(duration_seconds * fps / MAX_ANIMATEDIFF_CHUNK_FRAMES)),
    }


def validate_video_settings(*, steps: int, frames: int, fps: int, seed: int):
    """Backward-compatible validation for older callers and saved settings."""
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


def _chunk_frame_counts(total_frames: int) -> list[int]:
    """Split work into balanced 8–24 frame chunks supported by AnimateDiff."""
    total_frames = int(total_frames)
    if total_frames < 1:
        raise ValueError("At least one output frame is required.")
    if total_frames <= MAX_ANIMATEDIFF_CHUNK_FRAMES:
        return [max(MIN_ANIMATEDIFF_CHUNK_FRAMES, total_frames)]
    chunk_count = math.ceil(total_frames / MAX_ANIMATEDIFF_CHUNK_FRAMES)
    base, remainder = divmod(total_frames, chunk_count)
    chunks = [base + (1 if index < remainder else 0) for index in range(chunk_count)]
    if any(size < MIN_ANIMATEDIFF_CHUNK_FRAMES or size > MAX_ANIMATEDIFF_CHUNK_FRAMES for size in chunks):
        raise ValueError("Requested duration cannot be split into supported AnimateDiff chunks.")
    return chunks


def generate_video_segments(
    pipeline,
    *,
    prompt: str,
    negative_prompt: str | None,
    duration_seconds: int,
    fps: int,
    steps: int,
    guidance_scale: float,
    seed: int,
    output_path: str | Path,
    export_to_video: Callable,
    generator_factory: Callable,
    progress_callback: Callable | None = None,
) -> dict:
    """Generate up to 60 seconds as 24-frame-or-shorter clips, then assemble MP4.

    AnimateDiff does not carry latent motion state between these independent chunks.
    The same prompt and adapters are reused; deterministic, distinct seeds are used
    for each chunk. A final trim removes any extra frames required for 1-second clips
    at 6 FPS, where the model's practical minimum is 8 frames.
    """
    request = resolve_video_request(
        duration_seconds=duration_seconds, fps=fps, seed=seed,
        quality=next((key for key, value in VIDEO_QUALITY_PRESETS.items()
                      if value["steps"] == int(steps) and value["guidance_scale"] == float(guidance_scale)), "balanced"),
    )
    chunks = _chunk_frame_counts(request["total_frames"])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("Video assembly requires imageio-ffmpeg. Install the project's requirements.") from exc

    with tempfile.TemporaryDirectory(prefix="video_segments_", dir=str(output_path.parent)) as temp_dir:
        temp_root = Path(temp_dir)
        segment_paths = []
        total_steps = max(1, len(chunks) * int(steps))
        for index, frame_count in enumerate(chunks):
            segment_path = temp_root / f"segment_{index:03d}.mp4"

            def on_step_end(_pipeline, step_index, _timestep, callback_kwargs):
                if progress_callback:
                    progress_callback(index, len(chunks), min(int(step_index) + 1, int(steps)), int(steps))
                return callback_kwargs

            chunk_seed = (int(seed) + index * 104729) % 2147483647
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                num_frames=frame_count,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance_scale),
                generator=generator_factory(chunk_seed),
                callback_on_step_end=on_step_end,
            )
            export_to_video(result.frames[0], str(segment_path), fps=int(fps))
            if not segment_path.is_file() or segment_path.stat().st_size <= 0:
                raise RuntimeError(f"AnimateDiff did not export segment {index + 1}.")
            segment_paths.append(segment_path)

        concat_file = temp_root / "segments.txt"
        with concat_file.open("w", encoding="utf-8") as handle:
            for segment_path in segment_paths:
                escaped = str(segment_path.resolve()).replace("'", "'\\''")
                handle.write(f"file '{escaped}'\n")

        command = [
            ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-t", str(int(duration_seconds)),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-r", str(int(fps)),
            "-movflags", "+faststart", str(output_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=3600, check=False)
        if completed.returncode != 0 or not output_path.is_file() or output_path.stat().st_size <= 0:
            detail = (completed.stderr or "FFmpeg did not create the assembled MP4.")[-1600:]
            raise RuntimeError("Video segment assembly failed: " + detail)

    if progress_callback:
        progress_callback(len(chunks), len(chunks), int(steps), int(steps))
    return {
        "duration_seconds": int(duration_seconds),
        "fps": int(fps),
        "frames": int(request["total_frames"]),
        "segments": len(chunks),
        "segment_frames": chunks,
        "steps": int(steps),
        "quality": request["quality"],
        "output_path": str(output_path),
    }
