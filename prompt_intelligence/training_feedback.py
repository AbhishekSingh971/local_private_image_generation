"""Conservative reward/penalty evaluation for completed LoRA training runs.

Training loss is a diagnostic, not proof of identity fidelity or image quality.
The evaluator therefore caps automatic positive reward and requests human review
of generated validation samples before a model is promoted.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .evaluation import EvaluationReport, _normalized_rating
from .feedback_store import load_bandit_state, save_bandit_state, save_feedback
from .reward import PreferenceBandit


def evaluate_training_run(
    *,
    status: str,
    losses: Sequence[float] | None = None,
    human_scores: Mapping[str, Any] | None = None,
) -> EvaluationReport:
    state = (status or "").strip().lower()
    values: list[float] = []
    invalid_loss_seen = False
    for raw in losses or ():
        try:
            value = float(raw)
        except (TypeError, ValueError):
            invalid_loss_seen = True
            continue
        if math.isfinite(value) and value >= 0:
            values.append(value)
        else:
            invalid_loss_seen = True

    scores: dict[str, float] = {}
    sources: dict[str, str] = {}
    penalties: list[str] = []
    review: list[str] = []

    if state != "completed":
        scores["training_stability"] = -1.0
        sources["training_stability"] = "job_status"
        penalties.append("Training did not complete successfully; this run must not be promoted.")
        review.extend(("sample_quality", "identity"))
    elif not values:
        review.extend(("training_stability", "loss_trend", "sample_quality", "identity"))
        penalties.append("No finite training-loss observations were available for evaluation.")
    else:
        # A bounded diagnostic score: a falling loss is weak evidence of optimization,
        # not evidence that the resulting LoRA generalizes or preserves identity.
        scores["training_stability"] = -0.5 if invalid_loss_seen else 0.25
        sources["training_stability"] = "training_log"
        if invalid_loss_seen:
            penalties.append("Non-finite or invalid loss observations were detected in the training log.")
        if len(values) >= 2:
            first = sum(values[: max(1, len(values) // 4)]) / max(1, len(values) // 4)
            last_values = values[-max(1, len(values) // 4):]
            last = sum(last_values) / len(last_values)
            relative_change = (first - last) / max(abs(first), 1e-8)
            scores["loss_trend"] = max(-0.5, min(0.5, relative_change))
            sources["loss_trend"] = "training_log"
            if relative_change < -0.05:
                penalties.append("Training loss increased; inspect samples for overfitting or instability.")
            elif relative_change <= 0.02:
                penalties.append("Training loss did not show a clear downward trend.")
        else:
            review.append("loss_trend")
        review.extend(("sample_quality", "identity"))
        penalties.append("Training loss alone cannot validate likeness, prompt adherence, or generalization.")

    for key, value in (human_scores or {}).items():
        if key not in {"identity", "sample_quality", "image_quality", "prompt_adherence"}:
            raise ValueError("Unsupported training feedback dimension: " + str(key))
        normalized_key = "sample_quality" if key == "image_quality" else key
        scores[normalized_key] = _normalized_rating(value)
        sources[normalized_key] = "human"
        if normalized_key in review:
            review.remove(normalized_key)

    weights = {
        key: (2.0 if key in {"identity", "sample_quality"} else 1.0)
        for key in scores
    }
    reward = PreferenceBandit.combine_feedback(scores, weights) if scores else 0.0
    # Never let an automated loss curve alone produce a strong positive reward.
    if not human_scores:
        reward = min(reward, 0.2)
    return EvaluationReport(
        reward=round(max(-1.0, min(1.0, reward)), 5),
        scores=scores,
        weights=weights,
        sources=sources,
        penalties=tuple(dict.fromkeys(penalties)),
        needs_human_review=tuple(dict.fromkeys(review)),
    )


def record_training_run_feedback(
    *,
    profile_id: str,
    job_id: str,
    status: str,
    configuration_id: str,
    context: Mapping[str, Any],
    losses: Sequence[float] | None = None,
    human_scores: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report = evaluate_training_run(status=status, losses=losses, human_scores=human_scores)
    context_key = PreferenceBandit.context_key(context)
    bandit = PreferenceBandit.from_dict(load_bandit_state(context_key))
    stats = bandit.record(configuration_id, report.reward, context)
    feedback_id = save_feedback(
        asset_id=None,
        profile_id=profile_id,
        source_id=job_id,
        source_type="training_job",
        task="face_training",
        configuration_id=configuration_id,
        context=context,
        report=report.to_dict(),
    )
    save_bandit_state(context_key, bandit.to_dict())
    return {
        "feedback_id": feedback_id,
        "evaluation": report.to_dict(),
        "learning": {"context": context_key, "configuration_trials": stats.trials,
                     "mean_reward": stats.mean_reward},
    }
