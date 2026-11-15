from identity_prompt import ensure_face_identity_trigger


def test_face_prompt_includes_same_trigger_phrase_used_during_training():
    result = ensure_face_identity_trigger(
        "standing outdoors in daylight, wearing a dark jacket",
        "face_ab12cd34",
    )
    assert result == (
        "a photo of face_ab12cd34 person, "
        "standing outdoors in daylight, wearing a dark jacket"
    )


def test_face_prompt_does_not_duplicate_existing_trigger():
    prompt = "a photo of face_ab12cd34 person, portrait photograph"
    assert ensure_face_identity_trigger(prompt, "face_ab12cd34") == prompt


def test_face_trigger_uses_token_boundaries_not_substrings():
    result = ensure_face_identity_trigger("face_ab12cd34extra person", "face_ab12cd34")
    assert result.startswith("a photo of face_ab12cd34 person, ")


def test_empty_identity_token_leaves_prompt_unchanged():
    assert ensure_face_identity_trigger("a portrait photo", "") == "a portrait photo"


def test_identity_trigger_rejects_invalid_profile_token():
    import pytest

    with pytest.raises(ValueError, match="invalid identity trigger token"):
        ensure_face_identity_trigger("a portrait photo", "bad token")
