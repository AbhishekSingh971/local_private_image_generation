"""Runtime prompt planning and bounded preference-strategy selection.

The selector only chooses among small, reviewed prompt additions. It never changes
the requested subject, scene, pose, or actions, and it never starts extra inference.
"""
from __future__ import annotations

import os
from typing import Any, Mapping

from .engine import PromptEngine, PromptEngineError
from .reward import PreferenceBandit

IMAGE_STRATEGIES = {
    "baseline": ("", ""),
    "identity_guard": (
        "preserve the selected person's recognizable facial identity and distinctive facial structure",
        "identity drift, altered facial structure, changed distinctive features",
    ),
    "full_body_guard": (
        "show the complete requested figure within the frame, head to feet when full-body framing is requested, with coherent natural anatomy",
        "cropped head, cropped feet, cut-off limbs, anatomically inconsistent proportions",
    ),
}
VIDEO_STRATEGIES = {
    "baseline": ("", ""),
    "temporal_guard": (
        "maintain consistent subject identity, clothing, lighting and scene layout throughout the clip; use smooth physically plausible motion",
        "frame-to-frame identity drift, temporal flicker, jitter, morphing face, changing clothing, unstable background",
    ),
}


def _strategy_for(task: str, context: Mapping[str, Any]) -> str:
    if task == "video":
        candidates = VIDEO_STRATEGIES
    else:
        candidates = ["baseline"]
        if context.get("profile_id") not in (None, "", "global"):
            candidates.append("identity_guard")
        if context.get("full_body"):
            candidates.append("full_body_guard")
        candidates = {name: IMAGE_STRATEGIES[name] for name in candidates}

    if os.getenv("PROMPT_STRATEGY_LEARNING_ENABLED", "true").strip().lower() in {"0", "false", "no", "off"}:
        return "baseline"
    if not os.getenv("DATABASE_URL", "").strip():
        return "baseline"
    try:
        from .feedback_store import load_bandit_state
        scoped = dict(context)
        scoped.setdefault("task", task)
        scoped.setdefault("strategy_version", "v2")
        key = PreferenceBandit.context_key(scoped)
        bandit = PreferenceBandit.from_dict(load_bandit_state(key))
        for arm in candidates:
            bandit.register(arm, scoped)
        return bandit.select(scoped)
    except Exception:
        # Preference learning is advisory. A database outage must not block
        # generation or make an unvalidated strategy appear to have been learned.
        return "baseline"


def prepare_generation_prompt(
    prompt: str,
    negative_prompt: str = "",
    *,
    task: str = "image",
    context: Mapping[str, Any] | None = None,
    preset: str | None = None,
    has_reference_image: bool = False,
) -> dict[str, Any]:
    """Build a transparent prompt plan for image or video generation."""
    safe_context = dict(context or {})
    safe_context.setdefault("task", task)
    result = PromptEngine().build(prompt, task=task, context=safe_context)
    strategies = VIDEO_STRATEGIES if task == "video" else IMAGE_STRATEGIES
    strategy_id = _strategy_for(task, safe_context)
    positive_addition, negative_addition = strategies[strategy_id]
    positive_parts = [result.positive_prompt.strip(), positive_addition]
    negative_parts = [result.negative_prompt.strip(), (negative_prompt or "").strip(), negative_addition]
    positive = ", ".join(part for part in positive_parts if part)
    negative = ", ".join(dict.fromkeys(part for part in negative_parts if part))
    if task == "image":
        from image_generation import build_generation_prompts
        positive, negative = build_generation_prompts(
            positive, negative, preset or "standard",
            has_reference_image=has_reference_image,
        )
    return {
        "prompt": positive,
        "negative_prompt": negative,
        "original_prompt": result.original_prompt,
        "provider": result.provider,
        "strategy_id": strategy_id,
        "strategy_version": "v2",
        "task_instructions": result.task_instructions,
        "parameters": result.parameters,
    }


__all__ = ["prepare_generation_prompt", "PromptEngineError"]
