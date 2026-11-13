from PIL import Image, ImageDraw

import pytest

from regional_inpainting import (
    merge_inpainted_crop,
    normalize_edit_mask,
    prepare_inpaint_crop,
)


def _paint_mask(size=(640, 480), box=(250, 180, 310, 240)):
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle(box, fill=255)
    return mask


def test_normalize_mask_returns_white_selection_bbox_and_coverage():
    source = Image.new("RGB", (640, 480), "blue")
    mask = _paint_mask()
    normalized, bbox, coverage = normalize_edit_mask(source, mask)
    assert normalized.mode == "L"
    assert normalized.getpixel((270, 200)) == 255
    assert normalized.getpixel((10, 10)) == 0
    assert bbox == (250, 180, 311, 241)
    assert 0 < coverage < 0.02


def test_normalize_mask_rejects_empty_mask():
    source = Image.new("RGB", (512, 512), "white")
    mask = Image.new("L", source.size, 0)
    with pytest.raises(ValueError, match="Paint the area"):
        normalize_edit_mask(source, mask)


def test_normalize_mask_rejects_size_mismatch():
    source = Image.new("RGB", (512, 512), "white")
    mask = Image.new("L", (256, 256), 255)
    with pytest.raises(ValueError, match="dimensions must exactly match"):
        normalize_edit_mask(source, mask)


def test_normalize_mask_rejects_a_mask_that_covers_too_much_of_the_scene():
    source = Image.new("RGB", (512, 512), "white")
    mask = Image.new("L", source.size, 255)
    with pytest.raises(ValueError, match="no more than 65%"):
        normalize_edit_mask(source, mask)


def test_prepare_inpaint_crop_adds_scene_context_and_bounds_resolution():
    source = Image.new("RGB", (1200, 900), "navy")
    mask = _paint_mask((1200, 900), (580, 430, 620, 470))
    work = prepare_inpaint_crop(source, mask, padding=128, max_edge=768)
    crop_left, crop_top, crop_right, crop_bottom = work["crop_box"]
    mask_left, mask_top, mask_right, mask_bottom = work["mask_bbox"]
    assert crop_left < mask_left and crop_top < mask_top
    assert crop_right > mask_right and crop_bottom > mask_bottom
    assert max(work["work_size"]) <= 768
    assert min(work["work_size"]) >= 512
    assert work["work_source"].size == work["work_mask"].size


def test_merge_preserves_pixels_outside_the_expanded_mask_exactly():
    source = Image.new("RGB", (400, 320), (10, 20, 30))
    mask = Image.new("L", source.size, 0)
    ImageDraw.Draw(mask).rectangle((160, 120, 220, 180), fill=255)
    bbox = (32, 32, 352, 288)
    generated_crop = Image.new("RGB", (320, 256), (220, 30, 40))
    result = merge_inpainted_crop(
        source, generated_crop, crop_box=bbox,
        crop_mask=mask.crop(bbox), grow_px=8, blur_px=2,
    )
    assert result.getpixel((10, 10)) == source.getpixel((10, 10))
    assert result.getpixel((80, 80)) == source.getpixel((80, 80))
    assert result.getpixel((190, 150)) != source.getpixel((190, 150))


def test_merge_does_not_change_any_pixel_when_mask_is_empty():
    source = Image.new("RGB", (256, 256), (30, 40, 50))
    generated_crop = Image.new("RGB", source.size, (250, 0, 0))
    mask = Image.new("L", source.size, 0)
    result = merge_inpainted_crop(
        source, generated_crop, crop_box=(0, 0, 256, 256),
        crop_mask=mask, grow_px=8, blur_px=2,
    )
    assert result.tobytes() == source.tobytes()


def test_small_mask_near_image_edge_stays_inside_source_bounds():
    source = Image.new("RGB", (512, 512), "gray")
    mask = Image.new("L", source.size, 0)
    ImageDraw.Draw(mask).rectangle((0, 0, 20, 20), fill=255)
    work = prepare_inpaint_crop(source, mask)
    left, top, right, bottom = work["bbox"]
    assert (left, top) == (0, 0)
    assert right <= source.width and bottom <= source.height


def test_merge_rejects_bad_crop_box_by_pillow_validation():
    source = Image.new("RGB", (256, 256), "gray")
    mask = Image.new("L", (100, 100), 255)
    generated = Image.new("RGB", (100, 100), "red")
    with pytest.raises(ValueError):
        merge_inpainted_crop(
            source, generated, crop_box=(0, 0, 100, 100),
            crop_mask=Image.new("L", (64, 64), 255), grow_px=8, blur_px=2,
        )
