import pytest

from image_generation import (
    build_generation_prompts,
    resolve_image_dimensions,
    validate_image_preset,
    resolve_generation_seed,
    MAX_GENERATION_SEED,
)


def test_portrait_composition_uses_tall_sd15_canvas():
    assert resolve_image_dimensions("portrait") == (512, 768)


def test_square_and_landscape_canvas_sizes_are_supported():
    assert resolve_image_dimensions("square") == (512, 512)
    assert resolve_image_dimensions("landscape") == (768, 512)


def test_unknown_aspect_ratio_is_rejected():
    with pytest.raises(ValueError, match="portrait, square or landscape"):
        resolve_image_dimensions("panorama")


def test_standard_preset_does_not_change_user_prompts():
    prompt, negative = build_generation_prompts(
        "a lighthouse at sunset", "low quality", "standard"
    )
    assert prompt == "a lighthouse at sunset"
    assert negative == "low quality"


def test_full_body_preset_adds_framing_for_a_person_when_unspecified():
    prompt, negative = build_generation_prompts(
        "a realistic woman wearing a red dress", "", "full_body_photo"
    )
    assert prompt.startswith("photorealistic full-body photograph")
    assert "entire person visible from head to feet" in prompt
    assert "cropped body" in negative
    assert "clothes, clothing, dress" not in negative


def test_explicit_closeup_overrides_full_body_preset():
    prompt, negative = build_generation_prompts(
        "close-up portrait of a woman wearing a red dress", "", "full_body_photo"
    )
    assert prompt.startswith("photorealistic photograph")
    assert "full-body photograph" not in prompt
    assert "cropped body" not in negative


def test_reference_photo_prevents_forced_full_body_reframing():
    prompt, negative = build_generation_prompts(
        "woman in a red dress", "", "full_body_photo", has_reference_image=True
    )
    assert prompt.startswith("photorealistic photograph")
    assert "full-body photograph" not in prompt
    assert "cropped body" not in negative


def test_prompt_style_overrides_photorealistic_preset():
    prompt, negative = build_generation_prompts(
        "anime illustration of a woman wearing a red dress", "keep clean lines", "full_body_photo"
    )
    assert prompt == "anime illustration of a woman wearing a red dress"
    assert negative == "keep clean lines"


def test_prompt_does_not_get_unrequested_nudity_or_clothing_constraints():
    prompt, negative = build_generation_prompts(
        "photorealistic woman wearing a red dress", "", "full_body_photo"
    )
    assert "nude" not in prompt.lower()
    assert "clothes, clothing, dress" not in negative
    assert "bikini" not in negative


def test_non_person_prompt_does_not_get_forced_human_composition():
    prompt, negative = build_generation_prompts(
        "a red sports car on a mountain road", "", "full_body_photo"
    )
    assert prompt.startswith("photorealistic photograph")
    assert "full-body" not in prompt
    assert "cropped body" not in negative


def test_invalid_preset_is_rejected():
    with pytest.raises(ValueError, match="Image preset must be"):
        validate_image_preset("invented")


def test_generation_seed_keeps_valid_seed_unchanged():
    assert resolve_generation_seed("12345") == 12345
    assert resolve_generation_seed(0) == 0
    assert resolve_generation_seed(MAX_GENERATION_SEED) == MAX_GENERATION_SEED


def test_generation_seed_generates_a_valid_seed_when_missing():
    assert 0 <= resolve_generation_seed(None) <= MAX_GENERATION_SEED
    assert 0 <= resolve_generation_seed("") <= MAX_GENERATION_SEED


@pytest.mark.parametrize("seed", ["abc", "-1", "2147483648", "1.5"])
def test_invalid_generation_seed_is_rejected(seed):
    with pytest.raises(ValueError, match="Seed must be an integer"):
        resolve_generation_seed(seed)
