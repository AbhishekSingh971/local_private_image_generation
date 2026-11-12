import pytest

from generation_contracts import preserve_generation_text


def test_long_prompt_is_preserved_without_application_character_cap():
    prompt = ("A highly detailed scene with camera, light, fabric, anatomy and expression. " * 500)
    assert len(prompt) > 1200
    assert preserve_generation_text(prompt, "Prompt") == prompt


def test_unicode_and_outer_whitespace_are_preserved_exactly():
    prompt = "  一張完整的肖像 — natural skin, lighting and detail  \n"
    assert preserve_generation_text(prompt, "Prompt") == prompt


def test_null_characters_are_rejected_explicitly():
    with pytest.raises(ValueError, match="unsupported null character"):
        preserve_generation_text("before\x00after", "Prompt")



def test_model_token_capacity_accepts_limit_and_rejects_overflow():
    from generation_contracts import ensure_token_capacity

    assert ensure_token_capacity(512, 512, "FLUX.2 prompt") == 512
    with pytest.raises(ValueError, match="512-token input limit"):
        ensure_token_capacity(513, 512, "FLUX.2 prompt")
