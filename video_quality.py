"""Advisory frame-level video checks; these do not verify identity or anatomy."""
from __future__ import annotations

from pathlib import Path
from statistics import median
from typing import Any


def summarize_temporal_metrics(*, frame_count: int, sharpness: list[float],
                               brightness: list[float], luminance_deltas: list[float],
                               motion_medians: list[float], face_detection_ratio: float | None = None) -> dict[str, Any]:
    """Summarize sampled technical signals without claiming face identity consistency."""
    if frame_count < 2 or len(sharpness) < 2:
        return {
            "status": "needs_review",
            "reason": "Fewer than two readable frames were available for temporal checks.",
            "identity_consistency": "not_assessed",
            "human_review_required": True,
        }
    sharp_median = float(median(sharpness))
    brightness_mean = sum(brightness) / max(1, len(brightness))
    delta_median = float(median(luminance_deltas)) if luminance_deltas else 0.0
    delta_max = max(luminance_deltas, default=0.0)
    motion_median = float(median(motion_medians)) if motion_medians else 0.0
    motion_max = max(motion_medians, default=0.0)
    reasons = []
    if sharp_median < 18:
        reasons.append("Sampled frames appear soft or blurred.")
    if brightness_mean < 22 or brightness_mean > 238:
        reasons.append("Sampled frame exposure may be too dark or too bright.")
    if delta_max > max(42.0, delta_median * 3.5):
        reasons.append("Large frame-to-frame luminance changes may indicate flicker.")
    if motion_max > max(18.0, motion_median * 4.0):
        reasons.append("An unusually large motion jump was detected between sampled frames.")
    return {
        "status": "needs_review" if reasons else "technical_checks_passed",
        "frames_sampled": len(sharpness),
        "sharpness_median": round(sharp_median, 3),
        "brightness_mean": round(brightness_mean, 3),
        "luminance_delta_median": round(delta_median, 3),
        "luminance_delta_max": round(delta_max, 3),
        "motion_median": round(motion_median, 3),
        "motion_max": round(motion_max, 3),
        "face_detection_ratio": None if face_detection_ratio is None else round(float(face_detection_ratio), 3),
        "identity_consistency": "not_assessed",
        "reasons": reasons,
        "human_review_required": True,
        "note": "These are sampled technical/temporal signals only; they do not verify face identity, anatomy, or prompt adherence.",
    }


def assess_video_temporal_quality(path: str | Path, *, max_frames: int = 16) -> dict[str, Any]:
    """Sample a video with OpenCV when available; fail open and preserve the asset."""
    try:
        import cv2
        import numpy as np
    except Exception as exc:
        return {
            "status": "not_assessed",
            "reason": "Optional OpenCV/NumPy video analysis dependencies are unavailable.",
            "detail": type(exc).__name__,
            "identity_consistency": "not_assessed",
            "human_review_required": True,
        }
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return {
            "status": "not_assessed", "reason": "Video could not be opened for analysis.",
            "identity_consistency": "not_assessed", "human_review_required": True,
        }
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total < 2:
            return summarize_temporal_metrics(frame_count=total, sharpness=[], brightness=[],
                                               luminance_deltas=[], motion_medians=[])
        count = max(2, min(int(max_frames), total))
        indices = sorted(set(int(round(i * (total - 1) / (count - 1))) for i in range(count)))
        grays = []
        sharpness = []
        brightness = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grays.append(gray)
            sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            brightness.append(float(gray.mean()))
        if len(grays) < 2:
            return summarize_temporal_metrics(frame_count=len(grays), sharpness=sharpness,
                                               brightness=brightness, luminance_deltas=[],
                                               motion_medians=[])
        deltas = []
        motions = []
        previous = grays[0]
        for current in grays[1:]:
            deltas.append(float(np.abs(current.astype(np.float32) - previous.astype(np.float32)).mean()))
            flow = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 2, 15, 2, 5, 1.1, 0)
            magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
            motions.append(float(np.median(magnitude)))
            previous = current
        return summarize_temporal_metrics(
            frame_count=total, sharpness=sharpness, brightness=brightness,
            luminance_deltas=deltas, motion_medians=motions,
        )
    except Exception as exc:
        return {
            "status": "not_assessed",
            "reason": "Video temporal analysis failed; the generated asset was retained.",
            "detail": str(exc)[:240],
            "identity_consistency": "not_assessed",
            "human_review_required": True,
        }
    finally:
        capture.release()
