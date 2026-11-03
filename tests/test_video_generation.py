import pytest
from pathlib import Path
from types import SimpleNamespace

from video_generation import (
    _chunk_frame_counts,
    resolve_video_request,
    validate_video_settings,
    generate_video_segments,
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
    assert sum(chunks) >= 721
    assert sum(chunks) - 721 < 24
    assert all(chunk in (8, 16, 24) for chunk in chunks)
    assert chunks[:-1] == [24] * (len(chunks) - 1)


def test_legacy_video_settings_remain_supported():
    assert validate_video_settings(steps=12, frames=24, fps=8, seed=123)["frames"] == 24



def test_segmented_video_generates_supported_windows_and_assembles_mp4(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", SimpleNamespace(get_ffmpeg_exe=lambda: "ffmpeg"))
    calls = []

    class FakePipeline:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(frames=[["frame"] * kwargs["num_frames"]])

    def fake_export(frames, path, fps):
        Path(path).write_bytes(b"segment")

    def fake_run(command, **kwargs):
        Path(command[-1]).write_bytes(b"assembled-mp4")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("video_generation.subprocess.run", fake_run)
    output = tmp_path / "result.mp4"
    result = generate_video_segments(
        FakePipeline(), prompt="A calm landscape with moving clouds",
        negative_prompt="flicker", duration_seconds=4, fps=8,
        steps=12, guidance_scale=7.0, seed=99, output_path=output,
        export_to_video=fake_export,
        generator_factory=lambda seed: seed,
    )
    assert [call["num_frames"] for call in calls] == [24, 8]
    assert [call["generator"] for call in calls] == [99, 104828]
    assert result["duration_seconds"] == 4
    assert result["frames"] == 32
    assert result["segments"] == 2
    assert output.read_bytes() == b"assembled-mp4"
