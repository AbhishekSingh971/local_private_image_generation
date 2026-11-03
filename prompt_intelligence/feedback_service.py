"""Orchestrate evaluation, atomic observations, and contextual configuration learning."""
from __future__ import annotations

from typing import Any, Mapping

from .evaluation import evaluate_generation
from .feedback_store import save_feedback_and_bandit_state
from .reward import PreferenceBandit


def record_generation_feedback(
    *, asset_id: str | None, profile_id: str | None,
    source_id: str | None = None, source_type: str | None = None,
    task: str, configuration_id: str, context: Mapping[str, Any],
    human_scores: Mapping[str, Any] | None = None,
    pose_status: str | None = None,
    technical_quality: Mapping[str, Any] | None = None,
    requested_dimensions: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    report = evaluate_generation(
        human_scores=human_scores, pose_status=pose_status,
        technical_quality=technical_quality,
        requested_dimensions=requested_dimensions,
    )
    scoped_context = dict(context)
    scoped_context.setdefault("profile_id", profile_id or "global")
    scoped_context.setdefault("strategy_version", "v2")
    key = PreferenceBandit.context_key(scoped_context)
    persisted = save_feedback_and_bandit_state(
        asset_id=asset_id, profile_id=profile_id, source_id=source_id,
        source_type=source_type, task=task, configuration_id=configuration_id,
        context=scoped_context, report=report.to_dict(), context_key=key,
    )
    return {
        "feedback_id": persisted["feedback_id"],
        "evaluation": report.to_dict(),
        "learning": {
            "context": key,
            "configuration_trials": persisted["configuration_trials"],
            "mean_reward": persisted["mean_reward"],
            "atomic": True,
        },
    }
