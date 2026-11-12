"""Extract normalized MediaPipe face and body landmarks for geometry-aware controls.

The z values are relative model coordinates, not metric 3D reconstruction. This
module is a 2.5D landmark bridge, not a substitute for FLAME or SMPL-X.
"""
from __future__ import annotations


def analyze_geometry(image) -> dict:
    try:
        import numpy as np
        import mediapipe as mp
        from PIL import ImageOps
        from pose_validation import _face_mesh_api, estimate_head_pose
    except Exception as exc:
        return {"status": "not_available", "reason": "MediaPipe geometry dependencies are unavailable: " + str(exc)[:180]}
    try:
        rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
        if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 64:
            return {"status": "invalid_image", "reason": "Image must be at least 64 pixels in each dimension."}
        height, width = rgb.shape[:2]
        face_api = _face_mesh_api(mp)
        face_landmarks = []
        face_count = 0
        if face_api is not None:
            with face_api.FaceMesh(
                static_image_mode=True, max_num_faces=2, refine_landmarks=True,
                min_detection_confidence=0.55,
            ) as face_detector:
                face_result = face_detector.process(rgb)
            faces = face_result.multi_face_landmarks or []
            face_count = len(faces)
            if face_count == 1:
                face_landmarks = [
                    [round(float(point.x), 6), round(float(point.y), 6), round(float(point.z), 6)]
                    for point in faces[0].landmark
                ]
        body_landmarks = []
        body_count = 0
        pose_api = getattr(getattr(mp, "solutions", None), "pose", None)
        if pose_api is None:
            try:
                from mediapipe.python.solutions import pose as pose_api
            except Exception:
                pose_api = None
        if pose_api is not None:
            with pose_api.Pose(
                static_image_mode=True, model_complexity=1,
                enable_segmentation=False, min_detection_confidence=0.55,
            ) as pose_detector:
                pose_result = pose_detector.process(rgb)
            if pose_result.pose_landmarks:
                body_count = 1
                body_landmarks = [
                    [
                        round(float(point.x), 6), round(float(point.y), 6),
                        round(float(point.z), 6), round(float(point.visibility), 5),
                    ]
                    for point in pose_result.pose_landmarks.landmark
                ]
        head_pose, pose_reason = estimate_head_pose(image) if face_count == 1 else (None, "Exactly one face is required for head-pose estimation.")
        return {
            "status": "analyzed" if face_count or body_count else "no_landmarks",
            "image_dimensions": {"width": width, "height": height},
            "coordinate_system": {
                "x": "normalized image coordinate, left to right",
                "y": "normalized image coordinate, top to bottom",
                "z": "relative model depth, not metric distance",
                "face_landmark_count": len(face_landmarks),
                "body_landmark_count": len(body_landmarks),
            },
            "face_count": face_count,
            "face_landmarks_3d_relative": face_landmarks,
            "body_pose_landmarks_3d_relative": body_landmarks,
            "head_pose_degrees": (
                {"yaw": round(head_pose.yaw, 2), "pitch": round(head_pose.pitch, 2), "roll": round(head_pose.roll, 2)}
                if head_pose else None
            ),
            "head_pose_reason": pose_reason,
            "limitations": [
                "MediaPipe z coordinates are relative, not calibrated metric geometry.",
                "This does not reconstruct FLAME facial identity or SMPL-X body shape.",
                "Use human review; landmark confidence and occlusion can vary with lighting, crop and pose.",
            ],
        }
    except Exception as exc:
        return {"status": "not_assessed", "reason": "Geometry analysis failed: " + str(exc)[:240]}
