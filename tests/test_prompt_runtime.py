from prompt_intelligence.runtime import prepare_generation_prompt


def test_prompt_runtime_preserves_user_prompt_and_negative_constraints(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    result = prepare_generation_prompt(
        "A red bicycle beside a stone bridge at sunrise",
        "no text, no watermark",
        task="image",
        context={"profile_id": "global", "full_body": False},
        preset="standard",
    )
    assert "A red bicycle beside a stone bridge at sunrise" in result["prompt"]
    assert "no text" in result["negative_prompt"]
    assert result["strategy_id"] == "baseline"
    assert result["provider"] == "fallback"


def test_identity_strategy_is_only_added_when_a_profile_is_selected(monkeypatch):
    monkeypatch.setattr("prompt_intelligence.runtime._strategy_for", lambda task, context: "identity_guard")
    result = prepare_generation_prompt(
        "An adult person standing in a garden",
        task="image",
        context={"profile_id": "profile-123"},
        preset="standard",
    )
    assert "same individual" in result["prompt"]
    assert "identity drift" in result["negative_prompt"]
    assert "An adult person standing in a garden" in result["prompt"]


def test_video_runtime_adds_temporal_guard_without_rewriting_action(monkeypatch):
    monkeypatch.setattr("prompt_intelligence.runtime._strategy_for", lambda task, context: "temporal_guard")
    result = prepare_generation_prompt(
        "A person slowly waves at the camera",
        task="video",
        context={"profile_id": "global", "model_family": "animatediff_sd15"},
    )
    assert "slowly waves at the camera" in result["prompt"]
    assert "smooth physically plausible motion" in result["prompt"]
    assert "temporal flicker" in result["negative_prompt"]
