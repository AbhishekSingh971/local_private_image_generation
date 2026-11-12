"""Optional GFPGAN face restoration using a user-supplied model checkpoint.

Face restoration is opt-in because it can alter identity details. No weights are
bundled or downloaded. Keep the original generated image when this stage fails.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

_PIPELINE = None
_PIPELINE_KEY = None
_LOCK = threading.RLock()
_INFERENCE_LOCK = threading.Lock()


def resolve_face_restoration_options(enabled, weight=0.5):
    active = str(enabled or "").strip().lower() in {"1", "true", "yes", "on"}
    if not active:
        return {"enabled": False, "weight": 0.5, "model_path": None}
    try:
        fidelity = float(weight if weight not in (None, "") else 0.5)
    except (TypeError, ValueError) as exc:
        raise ValueError("Face restoration fidelity must be a number between 0 and 1.") from exc
    if not 0.0 <= fidelity <= 1.0:
        raise ValueError("Face restoration fidelity must be between 0 and 1.")
    raw_path = (os.environ.get("GFPGAN_MODEL_PATH") or "").strip()
    return {
        "enabled": True, "weight": fidelity,
        "model_path": str(Path(raw_path).expanduser()) if raw_path else None,
    }


def restore_faces(image, *, weight=0.5, model_path=None):
    """Restore detected faces while preserving the original canvas dimensions."""
    from PIL import Image
    import numpy as np
    import cv2
    import torch

    if not isinstance(image, Image.Image):
        raise TypeError("Face restoration requires a PIL image.")
    source = Path(model_path or os.environ.get("GFPGAN_MODEL_PATH") or "").expanduser()
    if not str(source) or not source.is_file():
        raise RuntimeError("Set GFPGAN_MODEL_PATH to a licensed GFPGAN checkpoint before enabling face restoration.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = (str(source.resolve()), source.stat().st_mtime_ns, device)
    global _PIPELINE, _PIPELINE_KEY
    with _LOCK:
        if _PIPELINE is None or _PIPELINE_KEY != key:
            try:
                from gfpgan import GFPGANer
            except ImportError as exc:
                raise RuntimeError("Install the optional gfpgan package to enable face restoration.") from exc
            _PIPELINE = GFPGANer(
                model_path=str(source), upscale=1, arch="clean",
                channel_multiplier=2, bg_upsampler=None, device=device,
            )
            _PIPELINE_KEY = key
        restorer = _PIPELINE
    if not _INFERENCE_LOCK.acquire(blocking=False):
        raise RuntimeError("The face-restoration engine is busy. Retry shortly.")
    try:
        rgb = np.asarray(image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        _cropped, restored_faces, restored_image = restorer.enhance(
            bgr, has_aligned=False, only_center_face=False,
            paste_back=True, weight=max(0.0, min(1.0, float(weight))),
        )
        if restored_image is None or not restored_faces:
            return {
                "image": image.copy(), "status": "no_face",
                "face_count": 0, "device": device,
            }
        restored_rgb = cv2.cvtColor(restored_image, cv2.COLOR_BGR2RGB)
        result = Image.fromarray(restored_rgb)
        if result.size != image.size:
            result = result.resize(image.size, Image.Resampling.LANCZOS)
        return {
            "image": result, "status": "completed",
            "face_count": len(restored_faces), "device": device,
        }
    finally:
        _INFERENCE_LOCK.release()
