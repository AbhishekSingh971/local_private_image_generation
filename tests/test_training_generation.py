import pytest
from PIL import ImageChops

from pose_validation import HeadPose, _normalize_euler_angles, build_pose_control_image, validate_pose_target
from training_generation import FACE_ANGLE_TARGETS, FACE_EXPRESSION_TARGETS, targets_for_dataset_type


def test_face_angles_are_nine_explicit_targets():
    targets = targets_for_dataset_type("face_angles")
    assert len(targets) == 9
    assert [item["label"] for item in targets] == [
        "Front view", "Middle-left", "Middle-right", "Three-quarter left",
        "Three-quarter right", "Left profile", "Right profile", "Looking up", "Looking down",
    ]
    assert targets[0]["yaw"] == 0
    assert targets[1]["yaw"] == -30
    assert targets[2]["yaw"] == 30
    assert targets[5]["yaw"] == -90
    assert targets[6]["yaw"] == 90
    assert targets[7]["pitch"] < 0
    assert targets[8]["pitch"] > 0
    assert all("prompt" in item and item["strength"] > 0 for item in targets)


def test_face_expressions_are_ten_independent_targets():
    targets = targets_for_dataset_type("expressions")
    assert len(targets) == 10
    assert [item["label"] for item in targets] == [
        "Neutral expression", "Gentle smile", "Broad smile", "Laughing", "Sad",
        "Angry", "Surprised", "Fearful", "Disgusted", "Confused",
    ]
    assert all("prompt" in item and item["strength"] > 0 for item in targets)
    assert all("yaw" not in item and "pitch" not in item for item in targets)


def test_unknown_dataset_type_is_rejected():
    with pytest.raises(ValueError, match="Dataset type must be"):
        targets_for_dataset_type("body_type")


def test_pose_estimator_normalizes_ambiguous_near_180_pitch_without_flipping_yaw():
    pitch, yaw, roll = _normalize_euler_angles(-177.0, 1.0, 0.0)
    assert pitch == pytest.approx(3.0)
    assert yaw == pytest.approx(1.0)
    assert roll == pytest.approx(0.0)


def test_pose_validation_marks_matching_target_passed():
    status, message = validate_pose_target(HeadPose(yaw=-85.0, pitch=3.0), -90.0, 0.0)
    assert status == "passed"
    assert "within target tolerances" in message


def test_pose_validation_marks_front_view_as_wrong_for_profile():
    status, message = validate_pose_target(HeadPose(yaw=4.0, pitch=1.0), -90.0, 0.0)
    assert status == "needs_review"
    assert "does not closely match" in message


def test_pose_validation_never_passes_when_pose_is_unknown():
    status, message = validate_pose_target(None, 90.0, 0.0)
    assert status == "needs_review"
    assert "could not be verified" in message



def test_pose_control_image_refuses_to_guess_without_source_face():
    with pytest.raises(ValueError, match="source face image is required"):
        build_pose_control_image(0, 0, 256, 256)
