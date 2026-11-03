from pathlib import Path

import pytest

from dataset_policy import select_training_files


def test_identity_training_excludes_angle_expression_and_body_photos():
    files = [Path(name) for name in ("source.jpg", "manual.jpg", "angle.jpg", "expression.jpg", "body.jpg")]
    metadata = {
        "source.jpg": {"photo_kind": "source"},
        "manual.jpg": {"photo_kind": "manual"},
        "angle.jpg": {"photo_kind": "angle"},
        "expression.jpg": {"photo_kind": "expression"},
        "body.jpg": {"photo_kind": "body"},
    }
    selected, excluded = select_training_files(files, metadata, "identity")
    assert [item.name for item in selected] == ["manual.jpg", "source.jpg"]
    assert {item["filename"] for item in excluded} == {"angle.jpg", "expression.jpg", "body.jpg"}


def test_body_training_never_uses_unclassified_legacy_files():
    files = [Path("body.jpg"), Path("unknown.jpg")]
    selected, excluded = select_training_files(
        files, {"body.jpg": {"photo_kind": "body"}}, "body"
    )
    assert [item.name for item in selected] == ["body.jpg"]
    assert excluded == [{"filename": "unknown.jpg", "reason": "missing_purpose_metadata"}]


def test_needs_review_photos_are_not_used():
    selected, excluded = select_training_files(
        [Path("face.jpg")],
        {"face.jpg": {"photo_kind": "source", "validation_status": "needs_review"}},
        "identity",
    )
    assert not selected
    assert excluded[0]["reason"] == "requires_human_review"


def test_legacy_identity_photos_remain_backward_compatible():
    selected, _ = select_training_files([Path("old.jpg")], {}, "identity")
    assert [item.name for item in selected] == ["old.jpg"]


def test_invalid_training_purpose_is_rejected():
    with pytest.raises(ValueError):
        select_training_files([], {}, "everything")


def test_legacy_caption_classifier_separates_pose_expression_and_body():
    from dataset_policy import infer_photo_kind_from_caption

    assert infer_photo_kind_from_caption("Left profile, yaw -90 degrees") == "angle"
    assert infer_photo_kind_from_caption("Gentle natural closed-mouth smile") == "expression"
    assert infer_photo_kind_from_caption("Body type: athletic proportions") == "body"
    assert infer_photo_kind_from_caption("Original portrait in daylight") is None
