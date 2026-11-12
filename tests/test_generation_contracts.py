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



class FakeTokenizer:
    def __init__(self, model_max_length=77):
        self.model_max_length = model_max_length

    def __call__(self, text, add_special_tokens=True, truncation=False):
        assert truncation is False
        return {"input_ids": list(range(len(text.split()) + int(add_special_tokens)))}

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is False
        assert kwargs["add_generation_prompt"] is True
        assert kwargs["enable_thinking"] is False
        return " ".join(messages[0]["content"].split()) + " assistant"


class FakePipeline:
    def __init__(self, tokenizer, tokenizer_2=None):
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2


def test_sd_prompt_capacity_accepts_both_sd15_tokenizers():
    from generation_contracts import validate_diffusion_prompt_capacity

    pipeline = FakePipeline(FakeTokenizer(77))
    result = validate_diffusion_prompt_capacity(pipeline, "short image prompt")
    assert result["tokenizer_prompt"] == 4


def test_sd_prompt_capacity_rejects_overflow_without_truncating():
    from generation_contracts import validate_diffusion_prompt_capacity

    pipeline = FakePipeline(FakeTokenizer(5))
    with pytest.raises(ValueError, match="exceeding this model's 5-token input limit"):
        validate_diffusion_prompt_capacity(pipeline, "this prompt has more than five words")


def test_sdxl_checks_both_tokenizers():
    from generation_contracts import validate_diffusion_prompt_capacity

    pipeline = FakePipeline(FakeTokenizer(77), FakeTokenizer(4))
    with pytest.raises(ValueError, match="tokenizer_2"):
        validate_diffusion_prompt_capacity(pipeline, "a prompt with several words")


def test_flux2_counts_prompt_after_chat_template_is_applied():
    from generation_contracts import count_flux2_prompt_tokens

    pipeline = FakePipeline(FakeTokenizer(512))
    pipeline.tokenizer_max_length = 512
    assert count_flux2_prompt_tokens(pipeline, "a cat") == 4


def test_flux2_chat_template_overflow_is_rejected():
    from generation_contracts import count_flux2_prompt_tokens

    pipeline = FakePipeline(FakeTokenizer(3))
    pipeline.tokenizer_max_length = 3
    with pytest.raises(ValueError, match="FLUX.2 prompt"):
        count_flux2_prompt_tokens(pipeline, "a cat")



def test_training_selector_includes_only_approved_photos(tmp_path):
    from training_approval import select_approved_training_files

    (tmp_path / "approved-a.jpg").write_bytes(b"a")
    (tmp_path / "approved-b.jpg").write_bytes(b"b")
    (tmp_path / "rejected.jpg").write_bytes(b"x")
    selected = select_approved_training_files(
        tmp_path, ["approved-b.jpg", "approved-a.jpg"], {".jpg", ".jpeg"}
    )
    assert [path.name for path in selected] == ["approved-a.jpg", "approved-b.jpg"]


def test_training_selector_rejects_path_traversal(tmp_path):
    from training_approval import select_approved_training_files

    with pytest.raises(ValueError, match="plain file basenames"):
        select_approved_training_files(tmp_path, ["../outside.jpg"], {".jpg"})


def test_training_selector_rejects_missing_approved_photo(tmp_path):
    from training_approval import select_approved_training_files

    with pytest.raises(ValueError, match="missing from the profile dataset"):
        select_approved_training_files(tmp_path, ["missing.jpg"], {".jpg"})


def test_training_selector_rejects_duplicate_names(tmp_path):
    from training_approval import select_approved_training_files

    with pytest.raises(ValueError, match="must be unique"):
        select_approved_training_files(tmp_path, ["same.jpg", "same.jpg"], {".jpg"})



def test_training_selector_rejects_windows_path_traversal(tmp_path):
    from training_approval import select_approved_training_files

    with pytest.raises(ValueError, match="plain file basenames"):
        select_approved_training_files(tmp_path, [r"..\outside.jpg"], {".jpg"})
