import pytest
from prompt_intelligence.evaluation import evaluate_generation
from prompt_intelligence.reward import PreferenceBandit

def test_human_ratings_are_normalized_and_weighted():
    report = evaluate_generation(human_scores={"identity":5,"angle":4,"expression":3,
        "body_proportions":5,"image_quality":4,"prompt_adherence":5})
    assert report.reward > 0.5
    assert report.scores["identity"] == 1.0
    assert report.scores["expression"] == 0.0

def test_pose_failure_creates_penalty_and_review():
    report = evaluate_generation(pose_status="needs_review",
        requested_dimensions=["angle","identity"], human_scores={"identity":5})
    assert report.scores["angle"] < 0
    assert "angle" in report.needs_human_review
    assert report.penalties

def test_quality_failure_does_not_claim_identity():
    report = evaluate_generation(technical_quality={"status":"needs_review","reasons":["Blur"]},
        requested_dimensions=["image_quality","identity"])
    assert report.scores["image_quality"] < 0
    assert "identity" in report.needs_human_review
    assert "identity" not in report.scores

def test_missing_dimensions_are_not_falsely_scored_as_zero():
    report = evaluate_generation(requested_dimensions=["identity","expression"])
    assert report.needs_human_review == ("identity","expression")
    assert report.scores == {}

def test_contextual_bandit_round_trips_and_separates_tasks():
    bandit = PreferenceBandit()
    image = {"task":"image","model_family":"sd15"}
    face = {"task":"face_training","model_family":"sd15"}
    bandit.register("cfg-a", image)
    bandit.register("cfg-b", image)
    bandit.record("cfg-a", 0.8, image)
    restored = PreferenceBandit.from_dict(bandit.to_dict())
    assert restored.select(image) == "cfg-b"
    with pytest.raises(ValueError): restored.select(face)

def test_reward_range_is_enforced():
    with pytest.raises(ValueError): PreferenceBandit().record("cfg", 1.1, {"task":"image"})

def test_human_rating_takes_precedence_over_pose_heuristic():
    report = evaluate_generation(
        human_scores={"angle": 5, "image_quality": 5},
        pose_status="needs_review",
        technical_quality={"status": "needs_review", "reasons": ["Blur"]},
    )
    assert report.scores["angle"] == 1.0
    assert report.scores["image_quality"] == 1.0
    assert report.sources["angle"] == "human"
    assert report.sources["image_quality"] == "human"


def test_contextual_bandit_never_shares_identity_feedback_between_profiles():
    bandit = PreferenceBandit()
    first = {"task": "image", "model_family": "sd15", "profile_id": "face-a"}
    second = {"task": "image", "model_family": "sd15", "profile_id": "face-b"}
    bandit.record("cfg-a", 1.0, first)
    bandit.register("cfg-b", second)
    restored = PreferenceBandit.from_dict(bandit.to_dict())
    assert restored.select(second) == "cfg-b"
    with pytest.raises(ValueError):
        restored.select({"task": "image", "model_family": "sd15", "profile_id": "face-c"})



def test_feedback_service_uses_one_atomic_persistence_operation(monkeypatch):
    from prompt_intelligence import feedback_service
    captured = {}

    def fake_atomic_save(**kwargs):
        captured.update(kwargs)
        return {"feedback_id": "f" * 32, "configuration_trials": 1, "mean_reward": 0.8}

    monkeypatch.setattr(feedback_service, "save_feedback_and_bandit_state", fake_atomic_save)
    result = feedback_service.record_generation_feedback(
        asset_id="a" * 32,
        profile_id="face-profile-a",
        task="image",
        configuration_id="identity_guard",
        context={"model_family": "sd15", "backend_class": "local_gpu", "control_mode": "txt2img"},
        human_scores={"identity": 5},
    )
    assert result["learning"]["atomic"] is True
    assert captured["configuration_id"] == "identity_guard"
    assert captured["context"]["profile_id"] == "face-profile-a"
    assert captured["context"]["strategy_version"] == "v2"
    assert captured["context_key"] == PreferenceBandit.context_key(captured["context"])
