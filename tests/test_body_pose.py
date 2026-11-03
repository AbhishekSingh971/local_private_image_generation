from PIL import Image, ImageDraw
import pytest

from body_pose import prepare_openpose_control_map


def test_openpose_map_is_fitted_to_requested_canvas():
    source = Image.new("RGB", (320, 640), (0, 0, 0))
    ImageDraw.Draw(source).line((160, 80, 160, 560), fill=(255, 255, 255), width=8)
    result = prepare_openpose_control_map(source, 512, 768)
    assert result.size == (512, 768)
    assert result.getbbox() is not None


def test_blank_openpose_map_is_rejected():
    with pytest.raises(ValueError, match="blank"):
        prepare_openpose_control_map(Image.new("RGB", (256, 256), (0, 0, 0)), 512, 768)
