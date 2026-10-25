from pathlib import Path

import pytest
from PIL import Image

import face_profiles as profiles


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    profile_dir = data_dir / "face_profiles"
    lora_dir = tmp_path / "models" / "face_loras"
    monkeypatch.setattr(profiles, "DATA_DIR", data_dir)
    monkeypatch.setattr(profiles, "PROFILES_FILE", data_dir / "face_profiles.json")
    monkeypatch.setattr(profiles, "PROFILES_DIR", profile_dir)
    monkeypatch.setattr(profiles, "LORA_DIR", lora_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)
    lora_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, profile_dir, lora_dir


def test_profiles_keep_datasets_separate(isolated_store):
    first = profiles.create_profile("Model A")
    second = profiles.create_profile("Model B")

    profiles.save_profile_image(first["id"], Image.new("RGB", (256, 256), "white"))

    assert profiles.get_profile(first["id"])["image_count"] == 1
    assert profiles.get_profile(second["id"])["image_count"] == 0
    assert profiles.dataset_dir(first["id"]) != profiles.dataset_dir(second["id"])
    assert len(profiles.list_profiles()) == 2


def test_duplicate_profile_name_and_prompt_token_are_rejected(isolated_store):
    profiles.create_profile("Model A", token="model_a_token")

    with pytest.raises(ValueError, match="name already exists"):
        profiles.create_profile("model a")

    with pytest.raises(ValueError, match="token is already assigned"):
        profiles.create_profile("Model B", token="model_a_token")


def test_local_and_runpod_readiness_are_independent(isolated_store):
    profile = profiles.create_profile("Model A")
    assert not profiles.profile_ready_for_backend(profile["id"], "local")
    assert not profiles.profile_ready_for_backend(profile["id"], "runpod")

    model_dir = profiles.model_path(profile["id"])
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "pytorch_lora_weights.safetensors").write_bytes(b"test adapter")

    assert profiles.profile_ready_for_backend(profile["id"], "local")
    assert not profiles.profile_ready_for_backend(profile["id"], "runpod")

    profiles.update_profile(profile["id"], runpod_model_ready=True)
    assert profiles.profile_ready_for_backend(profile["id"], "runpod")


def test_profile_id_validation_rejects_paths(isolated_store):
    with pytest.raises(ValueError):
        profiles.validate_profile_id("../../outside")
