"""Explicit target manifests for face-angle and facial-expression dataset jobs.

Yaw convention used by this app: negative = subject turns to their own left,
positive = subject turns to their own right. Pitch is negative when the chin
raises / the face points upward, positive when the chin lowers / the face
points downward. Pose estimates are approximate and must be reviewed visually.
"""
from __future__ import annotations

FACE_ANGLE_TARGETS = [
    {
        "label": "Front view",
        "prompt": "front-facing head pose, yaw 0 degrees, pitch 0 degrees, face straight toward the camera, both eyes similarly visible",
        "yaw": 0.0, "pitch": 0.0, "strength": 0.24,
    },
    {
        "label": "Middle-left",
        "prompt": "head turned toward the subject's own left by 30 degrees, yaw -30 degrees, pitch 0 degrees, clear three-quarter asymmetry between the two sides of the face",
        "yaw": -30.0, "pitch": 0.0, "strength": 0.48,
    },
    {
        "label": "Middle-right",
        "prompt": "head turned toward the subject's own right by 30 degrees, yaw +30 degrees, pitch 0 degrees, clear three-quarter asymmetry between the two sides of the face",
        "yaw": 30.0, "pitch": 0.0, "strength": 0.48,
    },
    {
        "label": "Three-quarter left",
        "prompt": "head turned toward the subject's own left by 55 degrees, yaw -55 degrees, pitch 0 degrees, near three-quarter view with the far eye less visible",
        "yaw": -55.0, "pitch": 0.0, "strength": 0.60,
    },
    {
        "label": "Three-quarter right",
        "prompt": "head turned toward the subject's own right by 55 degrees, yaw +55 degrees, pitch 0 degrees, near three-quarter view with the far eye less visible",
        "yaw": 55.0, "pitch": 0.0, "strength": 0.60,
    },
    {
        "label": "Left profile",
        "prompt": "strict left-facing side profile of the subject, yaw -90 degrees, only the near eye visible, nose silhouette points to the subject's own left, not a frontal portrait",
        "yaw": -90.0, "pitch": 0.0, "strength": 0.68,
    },
    {
        "label": "Right profile",
        "prompt": "strict right-facing side profile of the subject, yaw +90 degrees, only the near eye visible, nose silhouette points to the subject's own right, not a frontal portrait",
        "yaw": 90.0, "pitch": 0.0, "strength": 0.68,
    },
    {
        "label": "Looking up",
        "prompt": "head tilted upward, chin raised, face angled upward by about 18 degrees, pitch -18 degrees, keep the neck natural",
        "yaw": 0.0, "pitch": -18.0, "strength": 0.45,
    },
    {
        "label": "Looking down",
        "prompt": "head tilted downward, chin lowered, face angled downward by about 18 degrees, pitch +18 degrees, keep the neck natural",
        "yaw": 0.0, "pitch": 18.0, "strength": 0.45,
    },
]

FACE_EXPRESSION_TARGETS = [
    {"label": "Neutral expression", "prompt": "relaxed neutral expression, closed relaxed lips, relaxed eyebrows", "strength": 0.24},
    {"label": "Gentle smile", "prompt": "gentle natural closed-mouth smile, subtle raised mouth corners, relaxed eyes", "strength": 0.28},
    {"label": "Broad smile", "prompt": "broad joyful smile, lifted cheeks, visible teeth, natural eye creases", "strength": 0.34},
    {"label": "Laughing", "prompt": "genuine laughing expression, mouth open naturally, smiling eyes, raised cheeks", "strength": 0.38},
    {"label": "Sad", "prompt": "sad expression, inner eyebrows slightly raised, mouth corners gently lowered, watery-looking but natural eyes", "strength": 0.34},
    {"label": "Angry", "prompt": "angry expression, brows drawn together, focused eyes, lips pressed naturally", "strength": 0.35},
    {"label": "Surprised", "prompt": "surprised expression, eyebrows raised, eyes open wider, lips gently parted", "strength": 0.36},
    {"label": "Fearful", "prompt": "fearful expression, eyebrows raised and drawn together, eyes widened, lips tense and slightly parted", "strength": 0.38},
    {"label": "Disgusted", "prompt": "disgusted expression, slight nose wrinkle, upper lip subtly raised, eyebrows tightened", "strength": 0.35},
    {"label": "Confused", "prompt": "confused expression, one eyebrow slightly raised, brows subtly pinched, lips unevenly pursed", "strength": 0.34},
]


def targets_for_dataset_type(dataset_type: str) -> list[dict]:
    """Return a defensive copy of the supported targets for one dataset type."""
    key = (dataset_type or "").strip().lower()
    if key == "face_angles":
        return [dict(item) for item in FACE_ANGLE_TARGETS]
    if key == "expressions":
        return [dict(item) for item in FACE_EXPRESSION_TARGETS]
    raise ValueError("Dataset type must be face_angles or expressions.")
