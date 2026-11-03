import pytest

from image_generation import (
    build_generation_prompts,
    resolve_image_dimensions,
    validate_image_preset,
    resolve_generation_seed,
    MAX_GENERATION_SEED,
    validate_remote_generation_prompts,
    build_face_identity_prompt,
    extract_face_prompt_matrix,
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
    assert prompt.startswith("a realistic woman wearing a red dress")
    assert "photorealistic full-body photograph" in prompt
    assert "entire person visible from head to feet" in prompt
    assert "cropped body" in negative
    assert "clothes, clothing, dress" not in negative


def test_explicit_full_body_prompt_gets_anatomy_and_complete_limb_constraints():
    prompt, negative = build_generation_prompts(
        "photorealistic full-body photograph of an adult woman", "", "full_body_photo"
    )
    assert "anatomically coherent adult human body" in prompt
    assert "missing shoulder" in negative
    assert "cropped body" in negative


def test_explicit_closeup_overrides_full_body_preset():
    prompt, negative = build_generation_prompts(
        "close-up portrait of a woman wearing a red dress", "", "full_body_photo"
    )
    assert prompt.startswith("close-up portrait of a woman wearing a red dress")
    assert "photorealistic photograph" in prompt
    assert "full-body photograph" not in prompt
    assert "cropped body" not in negative


def test_reference_photo_prevents_forced_full_body_reframing():
    prompt, negative = build_generation_prompts(
        "woman in a red dress", "", "full_body_photo", has_reference_image=True
    )
    assert prompt.startswith("woman in a red dress")
    assert "photorealistic photograph" in prompt
    assert "full-body photograph" not in prompt
    assert "cropped body" not in negative


def test_explicit_full_body_request_overrides_reference_crop():
    prompt, negative = build_generation_prompts(
        "full-body head-to-toe photograph of a woman seated on a chair",
        "",
        "full_body_photo",
        has_reference_image=True,
    )
    assert prompt.startswith("photorealistic full-body photograph")
    assert "entire person visible from head to feet" in prompt
    assert "cropped body" in negative
    assert "cut off feet" in negative


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
    assert prompt.startswith("a red sports car on a mountain road")
    assert "photorealistic photograph" in prompt
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


class _FakeClipTokenizer:
    model_max_length = 77

    def tokenize(self, text):
        import re
        return re.findall(r"[A-Za-z0-9]+|[^A-Za-z0-9 ]", text)

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)


def test_fit_sd15_prompt_keeps_short_prompt_unchanged():
    from image_generation import fit_sd15_prompt

    tokenizer = _FakeClipTokenizer()
    prompt = "Photorealistic portrait of the same person."
    assert fit_sd15_prompt(tokenizer, prompt) == prompt


def test_fit_sd15_prompt_trims_to_clip_context_and_preserves_leading_instructions():
    from image_generation import fit_sd15_prompt

    tokenizer = _FakeClipTokenizer()
    prompt = "same adult person preserve identity true right side profile " + " ".join(
        f"optional_detail_{index}" for index in range(100)
    )
    fitted = fit_sd15_prompt(tokenizer, prompt)

    assert fitted.startswith("same adult person preserve identity true right side profile")
    assert len(tokenizer.tokenize(fitted)) + tokenizer.num_special_tokens_to_add() <= 77


def test_remote_prompt_validation_accepts_app_expanded_prompts():
    positive = "x" * 1800
    negative = "y" * 900
    assert validate_remote_generation_prompts(positive, negative) == (positive, negative)


def test_remote_prompt_validation_reports_empty_positive_prompt():
    with pytest.raises(ValueError, match="received an empty prompt"):
        validate_remote_generation_prompts("   ", "")


def test_remote_prompt_validation_uses_separate_positive_and_negative_limits():
    with pytest.raises(ValueError, match="expanded prompt exceeds the remote limit"):
        validate_remote_generation_prompts("x" * 2401, "")
    with pytest.raises(ValueError, match="expanded negative prompt exceeds the remote limit"):
        validate_remote_generation_prompts("valid prompt", "y" * 1201)


def test_trained_face_prompt_injects_exact_training_trigger():
    prompt, matrix = build_face_identity_prompt(
        "A photorealistic full-body woman standing in a garden.",
        profile_token="mizo_7f3a",
        profile_description="Long dark hair, oval face, brown eyes, soft eyebrows",
    )
    assert prompt.startswith("a photo of mizo_7f3a person")
    assert "Long dark hair, oval face, brown eyes, soft eyebrows" in prompt
    assert "full-body woman standing in a garden" in prompt
    assert matrix["mode"] == "trained_lora"
    assert matrix["trigger_token"] == "mizo_7f3a"
    assert matrix["profile_description_used"] is True


def test_trained_face_prompt_does_not_duplicate_trigger_when_already_present():
    prompt, matrix = build_face_identity_prompt(
        "a photo of mizo_7f3a person, front-facing studio portrait",
        profile_token="mizo_7f3a",
    )
    assert prompt.count("mizo_7f3a") == 1
    assert prompt.startswith("a photo of mizo_7f3a person")
    assert matrix["mode"] == "trained_lora"


def test_prompt_matrix_uses_only_traits_explicitly_present_in_prompt():
    source = (
        "Photorealistic full-body portrait of an adult woman with an oval face, "
        "almond-shaped hazel eyes, a straight nose, full lips and long black hair."
    )
    prompt, matrix = build_face_identity_prompt(source)
    assert prompt.startswith(source)
    assert "clear, coherent natural human face" in prompt
    assert matrix["mode"] == "prompt_matrix"
    assert matrix["features"]["face_shape"] == "oval face"
    assert matrix["features"]["nose"] == "straight nose"
    assert "hazel eyes" in matrix["features"]["eyes"]
    assert matrix["features"]["hair"] == "long black hair"
    assert "skin_tone" not in matrix["features"]


def test_prompt_without_human_subject_is_not_given_face_constraints():
    source = "A cinematic photograph of an old wooden house beside a lake."
    prompt, matrix = build_face_identity_prompt(source)
    assert prompt == source
    assert matrix["mode"] == "base_prompt"
    assert matrix["face_requested"] is False


def test_invalid_profile_trigger_token_is_rejected():
    with pytest.raises(ValueError, match="invalid trigger token"):
        build_face_identity_prompt("Portrait of a person", profile_token="bad token")
