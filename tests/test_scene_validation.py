import json

from PIL import Image

from scene_validation import (
    _local_endpoint_error,
    build_scene_validation_prompt,
    evaluate_scene_integrity,
    parse_scene_assessment,
)


def test_scene_validation_is_not_configured_by_default(monkeypatch):
    monkeypatch.delenv("SCENE_VISION_MODEL", raising=False)
    report = evaluate_scene_integrity(
        prompt="A clock on a desk",
        generated_image=Image.new("RGB", (512, 512), "white"),
    )
    assert report["status"] == "not_configured"
    assert report["automatic_retry"] is False


def test_only_loopback_ollama_endpoints_are_allowed():
    assert _local_endpoint_error("http://127.0.0.1:11434/api/generate") is None
    assert _local_endpoint_error("http://localhost:11434/api/generate") is None
    assert _local_endpoint_error("https://vision.example.com/api/generate")
    assert _local_endpoint_error("http://192.168.1.5:11434/api/generate")


def test_endpoint_rejects_credentials_and_query_parameters():
    assert _local_endpoint_error("http://user:pass@127.0.0.1:11434/api/generate")
    assert _local_endpoint_error("http://127.0.0.1:11434/api/generate?forward=1")


def test_prompt_describes_reference_and_generated_image_order():
    prompt = build_scene_validation_prompt(
        "Change the background but keep the person the same",
        has_reference_image=True,
        has_trained_identity=True,
    )
    assert "Image 1 is the user's reference" in prompt
    assert "Image 2 is the generated result" in prompt
    assert "<user_request>" in prompt


def test_prompt_only_review_uses_the_generated_image():
    prompt = build_scene_validation_prompt(
        "A digital clock on a desk",
        has_reference_image=False,
        has_trained_identity=False,
    )
    assert "The only image is the generated result" in prompt
    assert "digital" in prompt


def test_assessment_pass_requires_high_confidence_and_no_reported_issues():
    report = parse_scene_assessment({
        "result": "pass",
        "confidence": 0.91,
        "scene_integrity": "pass",
        "requested_changes_applied": "yes",
        "issues": [],
        "unintended_changes": [],
        "targeted_retry_hint": "",
    })
    assert report["status"] == "passed"
    assert report["recommendation"] == "none"


def test_uncertain_or_low_confidence_is_never_marked_passed():
    report = parse_scene_assessment({
        "result": "pass",
        "confidence": 0.55,
        "scene_integrity": "pass",
        "requested_changes_applied": "yes",
        "issues": [],
        "unintended_changes": [],
        "targeted_retry_hint": "",
    })
    assert report["status"] == "needs_review"


def test_reported_issue_generates_only_a_recommendation_not_automatic_retry():
    report = parse_scene_assessment({
        "result": "fail",
        "confidence": 0.96,
        "scene_integrity": "needs_review",
        "requested_changes_applied": "no",
        "issues": ["The minute hand is malformed."],
        "unintended_changes": [],
        "targeted_retry_hint": "Correct the minute hand while retaining the same clock face.",
    })
    assert report["status"] == "needs_review"
    assert report["recommendation"] == "targeted_retry_suggested"
    assert report["automatic_retry"] is False


def test_model_output_lists_and_hints_are_bounded():
    report = parse_scene_assessment({
        "result": "needs_review",
        "confidence": 0.8,
        "scene_integrity": "needs_review",
        "requested_changes_applied": "uncertain",
        "issues": [("x" * 1000)] * 12,
        "unintended_changes": [],
        "targeted_retry_hint": "h" * 1000,
    })
    assert len(report["issues"]) == 5
    assert len(report["issues"][0]) <= 240
    assert len(report["targeted_retry_hint"]) == 320


def test_non_finite_confidence_is_reduced_to_zero():
    report = parse_scene_assessment({
        "result": "pass",
        "confidence": float("nan"),
        "scene_integrity": "pass",
        "requested_changes_applied": "yes",
    })
    assert report["confidence"] == 0.0
    assert report["status"] == "needs_review"


def test_unknown_response_values_are_conservative():
    report = parse_scene_assessment({
        "result": "perfect",
        "confidence": 1.0,
        "scene_integrity": "perfect",
        "requested_changes_applied": "sure",
    })
    assert report["status"] == "needs_review"
    assert report["model_result"] == "needs_review"
    assert report["requested_changes_applied"] == "uncertain"


def test_parser_accepts_json_wrapped_in_explanatory_text():
    report = parse_scene_assessment(
        'Assessment: {"result":"needs_review","confidence":0.8,'
        '"scene_integrity":"needs_review","requested_changes_applied":"uncertain",'
        '"issues":["One object looks malformed."]}'
    )
    assert report["issues"] == ["One object looks malformed."]


def test_local_validation_does_not_forward_through_proxy_or_redirect(monkeypatch):
    from scene_validation import evaluate_scene_integrity
    from PIL import Image
    import requests

    captured = {}

    class FakeResponse:
        status_code = 302

        def raise_for_status(self):
            raise AssertionError("Redirects must be rejected before raise_for_status.")

    class FakeSession:
        trust_env = True

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, endpoint, **kwargs):
            captured["endpoint"] = endpoint
            captured["trust_env"] = self.trust_env
            captured["allow_redirects"] = kwargs.get("allow_redirects")
            return FakeResponse()

    monkeypatch.setenv("SCENE_VISION_MODEL", "local-vision")
    monkeypatch.setattr(requests, "Session", FakeSession)
    report = evaluate_scene_integrity(
        prompt="A clock",
        generated_image=Image.new("RGB", (64, 64), "white"),
    )
    assert captured["trust_env"] is False
    assert captured["allow_redirects"] is False
    assert report["status"] == "unavailable"
    assert "redirect" in report["reason"].lower()


def test_selected_identity_without_reference_must_remain_unverified():
    prompt = build_scene_validation_prompt(
        "A portrait of the trained person",
        has_reference_image=False,
        has_trained_identity=True,
    )
    assert "no identity reference is visible" in prompt
    assert "Do not claim identity consistency" in prompt
