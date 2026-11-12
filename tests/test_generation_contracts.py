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



def test_upscale_options_are_opt_in_and_use_x4():
    from image_upscaling import resolve_upscale_options

    options = resolve_upscale_options("false")
    assert options["enabled"] is False
    assert options["scale_factor"] == 4
    assert options["steps"] == 20


def test_upscale_options_validate_step_range():
    from image_upscaling import resolve_upscale_options

    with pytest.raises(ValueError, match="between 10 and 30"):
        resolve_upscale_options(True, 31)


def test_upscale_options_reject_non_integer_steps():
    from image_upscaling import resolve_upscale_options

    with pytest.raises(ValueError, match="must be an integer"):
        resolve_upscale_options(True, "many")



def test_model_bundle_safe_extract_restores_lora(tmp_path):
    import io
    import zipfile
    from model_artifacts import safe_extract_lora_bundle, sha256_bytes

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("pytorch_lora_weights.safetensors", b"weights")
        archive.writestr("adapter_config.json", "{}")
    payload = buffer.getvalue()
    destination = tmp_path / "restored"
    names = safe_extract_lora_bundle(payload, destination)
    assert "pytorch_lora_weights.safetensors" in names
    assert (destination / "pytorch_lora_weights.safetensors").read_bytes() == b"weights"
    assert len(sha256_bytes(payload)) == 64


def test_model_bundle_safe_extract_rejects_path_traversal(tmp_path):
    import io
    import zipfile
    from model_artifacts import safe_extract_lora_bundle

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../outside.txt", "bad")
        archive.writestr("pytorch_lora_weights.safetensors", b"weights")
    with pytest.raises(ValueError, match="unsafe file path"):
        safe_extract_lora_bundle(buffer.getvalue(), tmp_path / "restored")


def test_model_bundle_safe_extract_rejects_missing_lora_weights(tmp_path):
    import io
    import zipfile
    from model_artifacts import safe_extract_lora_bundle

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("readme.txt", "not a model")
    with pytest.raises(ValueError, match="does not contain"):
        safe_extract_lora_bundle(buffer.getvalue(), tmp_path / "restored")



def test_face_geometry_reports_unavailable_without_mediapipe(monkeypatch):
    import builtins
    from PIL import Image
    from pose_validation import assess_face_geometry

    real_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name == "mediapipe":
            raise ImportError("optional dependency unavailable")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    report = assess_face_geometry(Image.new("RGB", (512, 512), "white"))
    assert report["status"] == "not_assessed"
    assert "could not be checked" in report["reason"]



def test_encrypted_backup_round_trip_and_authentication():
    from backup_postgres import decrypt_bytes, encrypt_bytes

    key = b"\x11" * 32
    nonce = b"\x22" * 12
    original = b"private database dump bytes" * 100
    encrypted = encrypt_bytes(original, key, nonce)
    assert encrypted != original
    assert decrypt_bytes(encrypted, key) == original
    tampered = encrypted[:-1] + bytes([encrypted[-1] ^ 1])
    with pytest.raises(ValueError, match="authentication failed"):
        decrypt_bytes(tampered, key)


def test_backup_encryption_key_requires_exact_32_bytes():
    from backup_postgres import load_encryption_key

    with pytest.raises(RuntimeError, match="exactly 32 bytes"):
        load_encryption_key("short")


def test_backup_encryption_key_accepts_64_hex_characters():
    from backup_postgres import load_encryption_key

    assert load_encryption_key("ab" * 32) == bytes.fromhex("ab" * 32)



def test_disabled_upscaling_ignores_irrelevant_step_value():
    from image_upscaling import resolve_upscale_options

    options = resolve_upscale_options(False, "not-used")
    assert options["enabled"] is False
    assert options["steps"] == 20



def test_private_data_encryption_round_trip_and_key_rotation(monkeypatch):
    from private_data_crypto import decrypt_sensitive_bytes, encrypt_sensitive_bytes, is_encrypted

    old_key = "11" * 32
    new_key = "22" * 32
    monkeypatch.setenv("FACE_DATA_ENCRYPTION_KEY", old_key)
    payload = b"private face photo bytes"
    encrypted = encrypt_sensitive_bytes(payload)
    assert is_encrypted(encrypted)
    assert decrypt_sensitive_bytes(encrypted) == payload
    monkeypatch.setenv("FACE_DATA_ENCRYPTION_KEY", new_key)
    monkeypatch.setenv("FACE_DATA_ENCRYPTION_KEY_PREVIOUS", old_key)
    assert decrypt_sensitive_bytes(encrypted) == payload
    reencrypted = encrypt_sensitive_bytes(payload)
    assert decrypt_sensitive_bytes(reencrypted) == payload
    assert reencrypted != encrypted


def test_private_data_encryption_requires_key_to_decrypt(monkeypatch):
    from private_data_crypto import decrypt_sensitive_bytes, encrypt_sensitive_bytes

    monkeypatch.setenv("FACE_DATA_ENCRYPTION_KEY", "33" * 32)
    encrypted = encrypt_sensitive_bytes(b"secret")
    monkeypatch.delenv("FACE_DATA_ENCRYPTION_KEY")
    with pytest.raises(RuntimeError, match="Could not decrypt private data"):
        decrypt_sensitive_bytes(encrypted)


def test_private_data_encryption_is_backward_compatible_without_key(monkeypatch):
    from private_data_crypto import decrypt_sensitive_bytes, encrypt_sensitive_bytes, is_encrypted

    monkeypatch.delenv("FACE_DATA_ENCRYPTION_KEY", raising=False)
    raw = b"legacy plaintext data"
    assert encrypt_sensitive_bytes(raw) == raw
    assert decrypt_sensitive_bytes(raw) == raw
    assert not is_encrypted(raw)
