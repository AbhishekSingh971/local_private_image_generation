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



def test_identity_trigger_is_injected_for_local_and_rented_gpu_image_generation():
    app_source = open("app.py", encoding="utf-8").read()
    worker_source = open("gpu_worker.py", encoding="utf-8").read()
    assert "ensure_face_identity_trigger(prompt, profile[\"token\"])" in app_source
    assert "ensure_face_identity_trigger(prompt, profile[\"token\"])" in worker_source
