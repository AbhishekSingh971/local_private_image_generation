"""Optional geometric head-pose estimation and conservative target validation.

The generation service never labels a view as pose-verified when the estimator
is unavailable or cannot detect a face. Those cases are explicitly marked for
manual review.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


def _face_mesh_api(mp_module=None):
    """Return Face Mesh API across MediaPipe package layouts."""
    try:
        module = mp_module
        if module is None:
            import mediapipe as module
        api = getattr(getattr(module, "solutions", None), "face_mesh", None)
        if api is not None and hasattr(api, "FaceMesh"):
            return api
    except Exception:
        pass
    try:
        from mediapipe.python.solutions import face_mesh
        return face_mesh if hasattr(face_mesh, "FaceMesh") else None
    except Exception:
        return None


def _face_mesh_tessellation(mp_module=None):
    """Return tessellation connections across supported MediaPipe layouts."""
    api = _face_mesh_api(mp_module)
    connections = getattr(api, "FACEMESH_TESSELATION", None) if api else None
    if connections:
        return connections
    try:
        from mediapipe.python.solutions import face_mesh_connections
        return getattr(face_mesh_connections, "FACEMESH_TESSELATION", None)
    except Exception:
        return None


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
        face_mesh_api = _face_mesh_api(mp)
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


def build_pose_control_image(
    target_yaw: float,
    target_pitch: float,
    width: int = 512,
    height: int = 512,
    source_image=None,
):
    """Build a MediaPipe Face Mesh control map compatible with face-landmark ControlNet.

    A real source-face mesh is rotated in 3-D to the requested yaw/pitch before
    projection. This preserves the source's landmark proportions better than the
    previous hand-drawn generic face. If a source mesh cannot be detected, fail
    closed instead of sending a fabricated control map to the generator.
    """
    from math import cos, radians, sin
    import numpy as np
    from PIL import Image, ImageDraw
    import mediapipe as mp

    if source_image is None:
        raise ValueError("A source face image is required for pose-conditioned generation.")
    width = max(128, int(width))
    height = max(128, int(height))
    rgb = np.asarray(source_image.convert("RGB"))
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 64:
        raise ValueError("The source face image is too small for face-pose conditioning.")
    mesh_api = _face_mesh_api(mp)
    if mesh_api is None:
        raise RuntimeError(
            "MediaPipe is installed, but its Face Mesh API could not be imported. "
            "Reinstall the pinned mediapipe version in the active virtual environment."
        )
    with mesh_api.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.6,
    ) as detector:
        result = detector.process(rgb)
    if not result.multi_face_landmarks:
        raise ValueError("Could not detect a clear source face for pose conditioning.")

    landmarks = result.multi_face_landmarks[0].landmark
    source_pose, reason = estimate_head_pose(source_image)
    if source_pose is None:
        raise ValueError("Could not estimate source head pose; refusing to guess the pose-control rotation. " + reason)

    # MediaPipe coordinates are normalized image coordinates with relative depth.
    # Convert to a centered pseudo-3D point cloud, rotate by the target-source
    # angle delta, then project. This is geometric conditioning, not a guarantee
    # that diffusion will obey the target; output validation remains mandatory.
    source_h, source_w = rgb.shape[:2]
    face_scale = max(source_w, source_h)
    points = np.asarray([
        ((float(p.x) * source_w - source_w / 2.0),
         (float(p.y) * source_h - source_h / 2.0),
         (float(p.z) * face_scale))
        for p in landmarks
    ], dtype=np.float64)
    delta_yaw = radians(float(target_yaw) - source_pose.yaw)
    delta_pitch = radians(float(target_pitch) - source_pose.pitch)
    cy, sy = cos(delta_yaw), sin(delta_yaw)
    cp, sp = cos(delta_pitch), sin(delta_pitch)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    xr = x * cy + z * sy
    zr = -x * sy + z * cy
    yr = y * cp - zr * sp
    depth = y * sp + zr * cp

    # Normalize the projected mesh into a consistent square canvas without
    # changing the x/y ratio. Far-side landmarks are attenuated rather than
    # deleted so the conditioning image stays connected at profile angles.
    projected_x = xr
    projected_y = yr
    span_x = max(float(np.ptp(projected_x)), 1.0)
    span_y = max(float(np.ptp(projected_y)), 1.0)
    scale = min((width * 0.72) / span_x, (height * 0.72) / span_y)
    center_x, center_y = width / 2.0, height / 2.0
    px = center_x + projected_x * scale
    py = center_y + projected_y * scale
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    # MediaPipe face-landmark ControlNet is trained on mesh/landmark maps,
    # not the OpenPose body-skeleton palette used by the previous implementation.
    tessellation = _face_mesh_tessellation(mp)
    if not tessellation:
        raise RuntimeError(
            "MediaPipe face-mesh tessellation connections are unavailable in this installation."
        )
    for first, second in tessellation:
        if first >= len(px) or second >= len(px):
            continue
        # Suppress connections that project outside the image; retain a sparse
        # near-side mesh in strong profile views to avoid a solid scribble.
        if depth[first] < -face_scale * 0.12 and depth[second] < -face_scale * 0.12:
            continue
        x1, y1, x2, y2 = float(px[first]), float(py[first]), float(px[second]), float(py[second])
        if not all(isfinite(v) for v in (x1, y1, x2, y2)):
            continue
        if max(x1, x2) < 0 or min(x1, x2) >= width or max(y1, y2) < 0 or min(y1, y2) >= height:
            continue
        draw.line((x1, y1, x2, y2), fill=(180, 220, 200), width=1)
    # Draw eye, brow, and mouth landmarks more prominently, matching the face
    # condition's semantic emphasis while avoiding a fake generic face outline.
    for index, (xv, yv) in enumerate(zip(px, py)):
        if index in (33, 133, 159, 145, 362, 263, 386, 374, 1, 4, 13, 14, 61, 291, 70, 105, 334, 300):
            radius = 2
            draw.ellipse((xv-radius, yv-radius, xv+radius, yv+radius), fill=(240, 245, 230))
    return canvas
