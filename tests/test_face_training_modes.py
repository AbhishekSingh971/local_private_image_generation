from face_profiles import profile_model_training_mode


def test_local_and_rented_gpu_training_modes_are_independent(monkeypatch):
    import face_profiles

    monkeypatch.setattr(
        face_profiles,
        "get_profile",
        lambda profile_id: {
            "id": profile_id,
            "local_training_mode": "lora",
            "runpod_training_mode": "full_finetune",
        },
    )

    profile_id = "a" * 32
    assert profile_model_training_mode(profile_id, "local") == "lora"
    assert profile_model_training_mode(profile_id, "runpod") == "full_finetune"
