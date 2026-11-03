"""Orchestrate evaluation, persistent observations, and configuration learning."""
from __future__ import annotations
from typing import Any, Mapping
from .evaluation import evaluate_generation
from .feedback_store import load_bandit_state, save_bandit_state, save_feedback
from .reward import PreferenceBandit

def record_generation_feedback(*, asset_id: str | None, profile_id: str | None, source_id: str | None = None, source_type: str | None = None, task: str,
    configuration_id: str, context: Mapping[str, Any],
    human_scores: Mapping[str, Any] | None = None, pose_status: str | None = None,
    technical_quality: Mapping[str, Any] | None = None,
    requested_dimensions: tuple[str, ...] | list[str] | None = None) -> dict[str, Any]:
    report = evaluate_generation(human_scores=human_scores, pose_status=pose_status,
        technical_quality=technical_quality, requested_dimensions=requested_dimensions)
    scoped_context = dict(context)
    scoped_context.setdefault("profile_id", profile_id or "global")
    scoped_context.setdefault("strategy_version", "v1")
    key = PreferenceBandit.context_key(scoped_context)
    bandit = PreferenceBandit.from_dict(load_bandit_state(key))
    stats = bandit.record(configuration_id, report.reward, scoped_context)
    feedback_id = save_feedback(asset_id=asset_id, profile_id=profile_id, source_id=source_id, source_type=source_type, task=task,
        configuration_id=configuration_id, context=scoped_context, report=report.to_dict())
    save_bandit_state(key, bandit.to_dict())
    return {"feedback_id": feedback_id, "evaluation": report.to_dict(),
        "learning": {"context": key, "configuration_trials": stats.trials,
                     "mean_reward": stats.mean_reward}}
