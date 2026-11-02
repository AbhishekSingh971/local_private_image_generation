import json
from unittest.mock import patch

import pytest

from prompt_intelligence.engine import PromptEngine, PromptEngineError


def test_fallback_preserves_user_prompt_and_builds_negative_prompt():
    result = PromptEngine(endpoint="", model="").build(
        "A cinematic portrait beside a window", task="image"
    )
    assert result.original_prompt == "A cinematic portrait beside a window"
    assert result.positive_prompt == result.original_prompt
    assert result.negative_prompt
    assert result.provider == "fallback"


def test_empty_prompt_is_rejected():
    with pytest.raises(PromptEngineError):
        PromptEngine(endpoint="", model="").build("  ")


def test_context_is_carried_into_fallback_result():
    result = PromptEngine(endpoint="", model="").build(
        "Turn the face to the left",
        task="dataset_generation",
        context={"target_angle": "left_profile"},
    )
    assert result.parameters["context"]["target_angle"] == "left_profile"


def test_provider_response_is_structured():
    engine = PromptEngine(endpoint="https://example.invalid/v1/chat/completions", model="test")
    with patch.object(engine, "_request_provider", return_value={
        "positive_prompt": "studio portrait, soft window light",
        "negative_prompt": "blur, distorted face",
        "parameters": {"lighting": "soft window light"},
        "task_instructions": "Keep the subject centered",
    }):
        result = engine.build("Portrait", task="image")
    assert result.provider == "openai_compatible"
    assert result.parameters["lighting"] == "soft window light"


def test_invalid_provider_result_is_rejected():
    with pytest.raises(PromptEngineError):
        PromptEngine._validate_provider_result("portrait", "image", {
            "positive_prompt": "", "negative_prompt": "", "parameters": {}
        })
