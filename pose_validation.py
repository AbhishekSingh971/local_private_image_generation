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


def build_pose_control_image(target_yaw: float, target_pitch: float, width: int = 512, height: int = 512):
    """Render a synthetic, target-specific face keypoint map for an SD 1.5 OpenPose ControlNet.

    This is structural guidance, not a source-identity image. Identity comes from the
    source img2img image and, when available, the trained face LoRA/IP-Adapter.
    """
    from math import cos, radians, sin
    from PIL import Image, ImageDraw

    width = max(128, int(width))
    height = max(128, int(height))
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    # Approximate, generic 3-D facial landmarks. z > 0 is the front of the face.
    outlines = [
        (-68, -82, -12), (-42, -104, -10), (0, -112, -8), (42, -104, -10),
        (68, -82, -12), (80, -48, -9), (77, -8, -2), (63, 34, 5),
        (34, 66, 12), (0, 82, 15), (-34, 66, 12), (-63, 34, 5),
        (-77, -8, -2), (-80, -48, -9),
    ]
    left_eye = [(-49, -31, 9), (-39, -39, 13), (-27, -38, 16), (-18, -30, 13), (-29, -24, 12), (-41, -24, 10)]
    right_eye = [(18, -30, 13), (27, -38, 16), (39, -39, 13), (49, -31, 9), (41, -24, 10), (29, -24, 12)]
    left_brow = [(-52, -48, 4), (-40, -54, 5), (-28, -53, 7), (-17, -47, 9)]
    right_brow = [(17, -47, 9), (28, -53, 7), (40, -54, 5), (52, -48, 4)]
    nose = [(0, -28, 20), (0, -8, 30), (0, 5, 39), (-10, 11, 25), (0, 14, 29), (10, 11, 25)]
    mouth_outer = [(-27, 28, 12), (-15, 23, 15), (0, 25, 17), (15, 23, 15), (27, 28, 12), (16, 39, 13), (0, 43, 14), (-16, 39, 13)]
    mouth_inner = [(-16, 29, 15), (0, 31, 17), (16, 29, 15), (0, 38, 16)]
    left_ear = [(-77, -30, -4), (-88, -20, -3), (-88, 1, 0), (-76, 12, 2)]
    right_ear = [(77, -30, -4), (88, -20, -3), (88, 1, 0), (76, 12, 2)]

    groups = [
        (outlines, True), (left_eye, True), (right_eye, True),
        (left_brow, False), (right_brow, False), (nose, False),
        (mouth_outer, True), (mouth_inner, True), (left_ear, False), (right_ear, False),
    ]
    yaw = radians(float(target_yaw))
    pitch = radians(float(target_pitch))
    cy, sy = cos(yaw), sin(yaw)
    cp, sp = cos(pitch), sin(pitch)
    scale = min(width / 240.0, height / 270.0)
    center_x, center_y = width * 0.5, height * 0.48

    projected_groups = []
    all_points = {}
    point_counter = 0
    for points, closed in groups:
        projected = []
        for x, y, z in points:
            # Rotate the generic face around the vertical (yaw), then pitch.
            xr = x * cy + z * sy
            zr = -x * sy + z * cy
            yr = y * cp - zr * sp
            depth = y * sp + zr * cp
            px = center_x + xr * scale
            py = center_y - yr * scale
            visible = depth >= -9.0
            projected.append((px, py, visible))
            all_points[point_counter] = (px, py, visible)
            point_counter += 1
        projected_groups.append((projected, closed))

    palette = [
        (255, 80, 80), (255, 150, 40), (255, 220, 30),
        (100, 235, 85), (30, 215, 150), (30, 205, 255),
        (65, 130, 255), (130, 90, 255), (220, 70, 255), (255, 80, 170),
    ]
    for group_index, (projected, closed) in enumerate(projected_groups):
        color = palette[group_index % len(palette)]
        pair_count = len(projected) if closed else len(projected) - 1
        for index in range(pair_count):
            first = projected[index]
            second = projected[(index + 1) % len(projected)]
            if first[2] and second[2]:
                draw.line((first[0], first[1], second[0], second[1]), fill=color, width=4)
        for px, py, visible in projected:
            if visible:
                radius = 3
                draw.ellipse((px-radius, py-radius, px+radius, py+radius), fill=color)
    return canvas
