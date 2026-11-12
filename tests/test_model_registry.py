from model_registry import (
    model_configuration,
    resolve_model_family,
    supports_face_lora,
)


def test_supported_model_families():
    assert resolve_model_family("sd15") == "sd15"
    assert resolve_model_family("sdxl") == "sdxl"
    assert resolve_model_family("flux2_klein") == "flux2_klein"


def test_invalid_model_family_is_rejected():
    try:
        resolve_model_family("unknown-model")
    except ValueError as exc:
        assert "Unsupported image model family" in str(exc)
    else:
        raise AssertionError("Unknown model family must be rejected")


def test_face_lora_compatibility_is_explicit():
    assert supports_face_lora("sd15") is True
    assert supports_face_lora("sdxl") is False
    assert supports_face_lora("flux2_klein") is False


def test_flux2_registry_points_to_apache_licensed_checkpoint(monkeypatch):
    monkeypatch.delenv("FLUX2_KLEIN_MODEL_ID", raising=False)
    config = model_configuration("flux2_klein")
    assert config["model_id"] == "black-forest-labs/FLUX.2-klein-4B"
    assert config["supports_prompt_upsampling"] is False
    assert config["recommended_vram_gb"] == 13
    assert config["license"] == "Apache-2.0"
    assert config["distilled_steps"] == 4
    assert config["supports_negative_prompt"] is False
    assert config["supports_image_strength"] is False
