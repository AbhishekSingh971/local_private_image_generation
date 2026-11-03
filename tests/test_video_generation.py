import pytest

from video_generation import validate_video_settings


def test_video_settings_accept_reproducible_request():
    assert validate_video_settings(steps="12", frames="24", fps="8", seed="1234") == {
        "steps": 12, "frames": 24, "fps": 8, "seed": 1234
    }


@pytest.mark.parametrize("kwargs", [
    {"steps": 4, "frames": 16, "fps": 8, "seed": 1},
    {"steps": 12, "frames": 32, "fps": 8, "seed": 1},
    {"steps": 12, "frames": 16, "fps": 30, "seed": 1},
    {"steps": 12, "frames": 16, "fps": 8, "seed": 2147483648},
])
def test_video_settings_reject_unsupported_values(kwargs):
    with pytest.raises(ValueError):
        validate_video_settings(**kwargs)
