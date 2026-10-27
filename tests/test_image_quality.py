from PIL import Image, ImageDraw
import pytest

from image_quality import assess_image_quality, assess_image_bytes


def test_clear_detailed_image_passes_technical_checks():
    image = Image.new("RGB", (512, 512), "gray")
    draw = ImageDraw.Draw(image)
    for x in range(0, 512, 8):
        draw.line((x, 0, x, 511), fill="white", width=2)
        draw.line((0, x, 511, x), fill="black", width=2)
    report = assess_image_quality(image)
    assert report.status == "technical_checks_passed"
    assert report.sharpness_score > 45


def test_small_image_is_flagged_for_review():
    report = assess_image_quality(Image.new("RGB", (128, 128), "gray"))
    assert report.status == "needs_review"
    assert any("Resolution" in reason for reason in report.reasons)


def test_flat_image_is_flagged_as_potentially_blurry():
    report = assess_image_quality(Image.new("RGB", (512, 512), (128, 128, 128)))
    assert report.status == "needs_review"
    assert any("blurry" in reason for reason in report.reasons)


def test_dark_image_is_flagged():
    report = assess_image_quality(Image.new("RGB", (512, 512), (0, 0, 0)))
    assert report.status == "needs_review"
    assert any("dark" in reason for reason in report.reasons)


def test_bytes_api_rejects_empty_or_invalid_content():
    with pytest.raises(ValueError, match="empty"):
        assess_image_bytes(b"")
    with pytest.raises(ValueError, match="valid supported image"):
        assess_image_bytes(b"not an image")
