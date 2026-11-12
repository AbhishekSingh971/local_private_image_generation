"""Optional identity-consistency scoring with a user-supplied ArcFace-compatible ONNX model.

No biometric model weights are bundled or downloaded. The evaluator compares a
generated face to consented reference images; it is a quality signal, not proof
of identity. Calibrate thresholds on a private benchmark before using scores to
make automated decisions.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

_SESSION = None
_SESSION_KEY = None
_SESSION_LOCK = threading.RLock()
_CANONICAL_5 = None


def _model_path(value=None):
    raw = str(value or os.environ.get("FACE_IDENTITY_ONNX_MODEL") or "").strip()
    return Path(raw).expanduser() if raw else None


def _load_session(model_path):
    global _SESSION, _SESSION_KEY
    path = Path(model_path).resolve()
    if not path.is_file():
        raise RuntimeError("Configured face-identity ONNX model file does not exist.")
    key = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
    with _SESSION_LOCK:
        if _SESSION is not None and _SESSION_KEY == key:
            return _SESSION
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Install optional onnxruntime to enable identity similarity scoring.") from exc
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()
        if not inputs:
            raise RuntimeError("The configured identity model has no ONNX input.")
        shape = inputs[0].shape
        if len(shape) != 4 or shape[-2:] != [112, 112] and tuple(shape[-2:]) != (112, 112):
            raise RuntimeError("Identity model must accept a 112 × 112 face tensor in NCHW format.")
        _SESSION, _SESSION_KEY = session, key
        return session


def _aligned_face(image):
    import cv2
    import numpy as np
    import mediapipe as mp
    from PIL import ImageOps

    rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    if rgb.ndim != 3 or min(rgb.shape[:2]) < 64:
        raise ValueError("Reference or generated face image is too small.")
    from pose_validation import _face_mesh_api
    mesh_api = _face_mesh_api(mp)
    if mesh_api is None:
        raise RuntimeError("MediaPipe Face Mesh is unavailable in this environment.")
    with mesh_api.FaceMesh(
        static_image_mode=True, max_num_faces=2, refine_landmarks=False,
        min_detection_confidence=0.55,
    ) as detector:
        result = detector.process(rgb)
    faces = result.multi_face_landmarks or []
    if len(faces) != 1:
        raise ValueError(f"Expected exactly one detectable face; found {len(faces)}.")
    points = faces[0].landmark
    height, width = rgb.shape[:2]
    def point(index):
        return [points[index].x * width, points[index].y * height]
    left_eye = np.mean([point(33), point(133)], axis=0)
    right_eye = np.mean([point(362), point(263)], axis=0)
    source = np.asarray([
        left_eye, right_eye, point(1), point(61), point(291)
    ], dtype=np.float32)
    destination = np.asarray([
        [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
        [41.5493, 92.3655], [70.7299, 92.2041],
    ], dtype=np.float32)
    transform, _ = cv2.estimateAffinePartial2D(source, destination, method=cv2.LMEDS)
    if transform is None:
        raise ValueError("Could not align the detected face for identity comparison.")
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.warpAffine(bgr, transform, (112, 112), borderValue=0)


def _embedding(image, session):
    import cv2
    import numpy as np

    aligned = _aligned_face(image)
    blob = cv2.dnn.blobFromImage(
        aligned, scalefactor=1.0 / 127.5, size=(112, 112),
        mean=(127.5, 127.5, 127.5), swapRB=True,
    )
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: blob})
    if not output:
        raise RuntimeError("Identity model returned no embedding.")
    vector = np.asarray(output[0], dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not norm:
        raise RuntimeError("Identity model returned an empty embedding.")
    return vector / norm


def evaluate_identity_similarity(reference_images, generated_image, *, model_path=None) -> dict:
    """Compare a generated face to a centroid of at least two consented references."""
    configured = _model_path(model_path)
    if configured is None:
        return {
            "status": "not_configured",
            "reason": "Set FACE_IDENTITY_ONNX_MODEL to a licensed ArcFace-compatible 112 × 112 ONNX model.",
        }
    references = list(reference_images or [])[:20]
    if len(references) < 2:
        return {
            "status": "insufficient_references",
            "reference_count": len(references),
            "reason": "At least two approved reference photos are required for identity comparison.",
        }
    try:
        import numpy as np
        session = _load_session(configured)
        embeddings = []
        rejected = 0
        for image in references:
            try:
                embeddings.append(_embedding(image, session))
            except (ValueError, RuntimeError):
                rejected += 1
        if len(embeddings) < 2:
            return {
                "status": "needs_review",
                "reference_count": len(references),
                "usable_reference_count": len(embeddings),
                "reason": "Fewer than two reference photos contain one alignable face.",
            }
        centroid = np.mean(np.stack(embeddings), axis=0)
        centroid_norm = float(np.linalg.norm(centroid))
        if not centroid_norm:
            raise RuntimeError("Reference identity embeddings cancel out; review the reference set.")
        centroid = centroid / centroid_norm
        generated = _embedding(generated_image, session)
        similarity = float(np.dot(centroid, generated))
        model_version = (os.environ.get("FACE_IDENTITY_MODEL_VERSION") or "user-supplied").strip()[:80]
        return {
            "status": "scored",
            "cosine_similarity": round(similarity, 6),
            "reference_count": len(references),
            "usable_reference_count": len(embeddings),
            "rejected_reference_count": rejected,
            "model_version": model_version,
            "threshold": None,
            "interpretation": "Score only. Calibrate a release threshold on a consented benchmark; this is not proof of identity.",
        }
    except Exception as exc:
        return {
            "status": "not_assessed",
            "reason": "Identity similarity evaluation failed: " + str(exc)[:240],
        }
