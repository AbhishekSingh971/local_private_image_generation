from model_registry import (
    model_configuration,
    resolve_model_family,
    supports_face_lora,
)


def test_supported_model_families():
    assert resolve_model_family("sd15") == "sd15"
    assert resolve_model_family("sdxl") == "sdxl"
    assert resolve_model_family("flux3_klein") == "flux3_klein"


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
    assert supports_face_lora("flux3_klein") is False


def test_flux3_registry_points_to_distilled_checkpoint(monkeypatch):
    monkeypatch.delenv("FLUX3_KLEIN_MODEL_ID", raising=False)
    config = model_configuration("flux3_klein")
    assert config["model_id"] == "black-forest-labs/flux-3-image-klein"
    assert config["supports_prompt_upsampling"] is True
    assert config["supports_negative_prompt"] is False
    assert config["supports_image_strength"] is False
