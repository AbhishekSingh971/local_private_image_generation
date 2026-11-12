"""Sampled-frame quality and identity-consistency checks for generated videos."""
from __future__ import annotations

import os
from pathlib import Path


def evaluate_video_identity(video_path, reference_images, *, model_path=None, max_frames=8) -> dict:
    """Compare sampled video frames against approved references using the optional ONNX evaluator."""
    if not (model_path or os.environ.get("FACE_IDENTITY_ONNX_MODEL")):
        return {"status": "not_configured", "reason": "No licensed identity ONNX model is configured."}
    path = Path(video_path)
    if not path.is_file():
        return {"status": "not_assessed", "reason": "Generated video file is missing."}
    try:
        import imageio.v2 as imageio
        import numpy as np
        from PIL import Image
        from identity_evaluation import evaluate_identity_similarity

        reader = imageio.get_reader(str(path), format="ffmpeg")
        try:
            count = int(reader.count_frames())
            if count < 1:
                return {"status": "not_assessed", "reason": "Video contains no readable frames."}
            indices = np.linspace(0, count - 1, num=min(max(2, int(max_frames)), count), dtype=int)
            reports = []
            for index in sorted(set(int(value) for value in indices)):
                frame = Image.fromarray(reader.get_data(index)).convert("RGB")
                reports.append(evaluate_identity_similarity(reference_images, frame, model_path=model_path))
        finally:
            reader.close()
        scores = [
            float(report["cosine_similarity"])
            for report in reports if report.get("status") == "scored"
        ]
        if len(scores) < 2:
            return {
                "status": "needs_review",
                "sampled_frame_count": len(reports),
                "scored_frame_count": len(scores),
                "reason": "Fewer than two sampled frames had a measurable face identity score.",
                "frames": reports,
            }
        return {
            "status": "scored",
            "sampled_frame_count": len(reports),
            "scored_frame_count": len(scores),
            "mean_cosine_similarity": round(float(np.mean(scores)), 6),
            "minimum_cosine_similarity": round(float(np.min(scores)), 6),
            "maximum_cosine_similarity": round(float(np.max(scores)), 6),
            "standard_deviation": round(float(np.std(scores)), 6),
            "threshold": None,
            "interpretation": "Sampled-frame consistency signal only. Calibrate a release threshold on a private benchmark.",
            "frames": reports,
        }
    except Exception as exc:
        return {
            "status": "not_assessed",
            "reason": "Video identity evaluation failed: " + str(exc)[:240],
        }
