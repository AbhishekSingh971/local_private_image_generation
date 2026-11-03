import pytest

from video_generation import (
    _chunk_frame_counts,
    resolve_video_request,
    validate_video_settings,
)


def test_video_duration_and_quality_presets_resolve_to_frame_budget():
    request = resolve_video_request(duration_seconds="60", fps="8", seed="123", quality="high")
    assert request["duration_seconds"] == 60
    assert request["total_frames"] == 480
    assert request["segments"] == 20
    assert request["steps"] == 16
    assert request["quality"] == "high"


@pytest.mark.parametrize("duration", [0, 61, -1, "not-a-number"])
def test_video_duration_outside_supported_range_is_rejected(duration):
    with pytest.raises(ValueError):
        resolve_video_request(duration_seconds=duration, fps=8, seed=0, quality="balanced")


def test_video_quality_must_be_a_supported_preset():
    with pytest.raises(ValueError, match="quality"):
        resolve_video_request(duration_seconds=3, fps=8, seed=0, quality="lossless")


def test_one_second_at_six_fps_is_supported_by_eight_frame_model_minimum():
    assert _chunk_frame_counts(6) == [8]


def test_long_video_is_split_into_supported_balanced_chunks():
    chunks = _chunk_frame_counts(721)
    assert sum(chunks) == 721
    assert all(8 <= chunk <= 24 for chunk in chunks)
    assert max(chunks) - min(chunks) <= 1


def test_legacy_video_settings_remain_supported():
    assert validate_video_settings(steps=12, frames=24, fps=8, seed=123)["frames"] == 24
