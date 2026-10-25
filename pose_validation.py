"""Optional geometric head-pose estimation and conservative target validation.

The generation service never labels a view as pose-verified when the estimator
is unavailable or cannot detect a face. Those cases are explicitly marked for
manual review.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class HeadPose:
    yaw: float
    pitch: float
    roll: float = 0.0


def estimate_head_pose(image) -> tuple[HeadPose | None, str]:
    """Estimate yaw/pitch from MediaPipe Face Mesh plus OpenCV solvePnP.

    Returns (None, reason) instead of guessing when optional runtime dependencies,
    the face detector, or PnP are unavailable.
    """
    try:
        import cv2
        import mediapipe as mp
        import numpy as np
    except Exception:
        return None, "Pose validation unavailable: install the optional MediaPipe/OpenCV dependencies and restart the app."

    try:
        rgb = np.asarray(image.convert("RGB"))
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] < 64 or rgb.shape[1] < 64:
            return None, "Pose validation could not inspect this image size."
        height, width = rgb.shape[:2]
        face_mesh_api = getattr(getattr(mp, "solutions", None), "face_mesh", None)
        if face_mesh_api is None:
            return None, "MediaPipe Face Mesh API is unavailable in this installation."
        with face_mesh_api.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.55,
        ) as face_mesh:
            detection = face_mesh.process(rgb)
        if not detection.multi_face_landmarks:
            return None, "Pose validation could not find a clear face; review the image manually."
        landmarks = detection.multi_face_landmarks[0].landmark
        indices = (1, 152, 33, 263, 61, 291)
        image_points = np.array(
            [(landmarks[index].x * width, landmarks[index].y * height) for index in indices],
            dtype=np.float64,
        )
        # Generic 3-D face points paired to the six stable Face Mesh landmarks.
        model_points = np.array(
            [
                (0.0, 0.0, 0.0),          # nose tip
                (0.0, -330.0, -65.0),     # chin
                (-225.0, 170.0, -135.0),  # left eye outer corner
                (225.0, 170.0, -135.0),   # right eye outer corner
                (-150.0, -150.0, -125.0), # left mouth corner
                (150.0, -150.0, -125.0),  # right mouth corner
            ],
            dtype=np.float64,
        )
        focal_length = float(width)
        camera_matrix = np.array(
            [[focal_length, 0.0, width / 2.0],
             [0.0, focal_length, height / 2.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        distortion = np.zeros((4, 1), dtype=np.float64)
        success, rotation_vector, _translation = cv2.solvePnP(
            model_points, image_points, camera_matrix, distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None, "Pose validation could not estimate a stable head rotation."
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        angles, *_ = cv2.RQDecomp3x3(rotation_matrix)
        pitch, yaw, roll = (float(angles[0]), float(angles[1]), float(angles[2]))
        if not all(isfinite(value) for value in (yaw, pitch, roll)):
            return None, "Pose validation returned invalid angle estimates."
        # The chosen 3-D template and solvePnP axes have opposite signs from the
        # UI's subject-relative yaw convention, so normalize them here.
        pose = HeadPose(yaw=-yaw, pitch=pitch, roll=roll)
        return pose, ""
    except Exception:
        return None, "Pose validation failed; review the generated image manually."


def validate_pose_target(
    actual: HeadPose | None,
    target_yaw: float | None,
    target_pitch: float | None,
    *,
    yaw_tolerance: float = 24.0,
    pitch_tolerance: float = 17.0,
) -> tuple[str, str]:
    """Return passed only for a measurable, approximately matching pose."""
    if actual is None:
        return "needs_review", "The head pose could not be verified automatically. Inspect this image before training."
    if target_yaw is None or target_pitch is None:
        return "not_applicable", "No exact head pose target was specified."
    yaw_error = abs(actual.yaw - float(target_yaw))
    pitch_error = abs(actual.pitch - float(target_pitch))
    if yaw_error <= yaw_tolerance and pitch_error <= pitch_tolerance:
        return "passed", (
            f"Estimated pose yaw {actual.yaw:.0f}°, pitch {actual.pitch:.0f}°; "
            f"within target tolerances (yaw ±{yaw_tolerance:.0f}°, pitch ±{pitch_tolerance:.0f}°)."
        )
    return "needs_review", (
        f"Estimated pose yaw {actual.yaw:.0f}°, pitch {actual.pitch:.0f}°; "
        f"target yaw {float(target_yaw):.0f}°, pitch {float(target_pitch):.0f}°. "
        "This output does not closely match the requested pose."
    )
