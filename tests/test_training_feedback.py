import pytest

from prompt_intelligence.training_feedback import evaluate_training_run


def test_successful_loss_curve_gets_only_small_automatic_reward():
    report = evaluate_training_run(status="completed", losses=[1.0, 0.8, 0.6, 0.4])
    assert 0 < report.reward <= 0.2
    assert "sample_quality" in report.needs_human_review
    assert "identity" in report.needs_human_review
    assert report.penalties


def test_failed_training_is_penalized_and_not_promotable():
    report = evaluate_training_run(status="failed", losses=[1.0, 0.7])
    assert report.reward == -1.0
    assert report.scores["training_stability"] == -1.0
    assert any("must not be promoted" in item for item in report.penalties)


def test_human_sample_and_identity_ratings_can_raise_or_lower_reward():
    report = evaluate_training_run(
        status="completed",
        losses=[1.0, 0.8, 0.6],
        human_scores={"identity": 5, "sample_quality": 5},
    )
    assert report.reward > 0.2
    assert report.scores["identity"] == 1.0
    assert report.scores["sample_quality"] == 1.0
    assert "identity" not in report.needs_human_review


def test_invalid_human_dimension_is_rejected():
    with pytest.raises(ValueError):
        evaluate_training_run(status="completed", human_scores={"bad": 5})
