"""Context-aware preference learning over validated generation configurations.
This is a contextual bandit, not gradient-based RL or diffusion-model training.
Callers persist state with to_dict()/from_dict() and only register configurations
that have passed safety and resource validation.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from math import log, sqrt
from typing import Any, Mapping

@dataclass
class ArmStats:
    trials: int = 0
    mean_reward: float = 0.0

@dataclass
class PreferenceBandit:
    """UCB1 selector partitioned by task/profile context."""
    contexts: dict[str, dict[str, ArmStats]] = field(default_factory=dict)

    @staticmethod
    def context_key(context: Mapping[str, Any] | None = None) -> str:
        values = context or {}
        parts = (
            str(values.get("task", "image")).strip().lower()[:40],
            str(values.get("model_family", "default")).strip().lower()[:60],
            str(values.get("backend_class", "any")).strip().lower()[:30],
            str(values.get("control_mode", "default")).strip().lower()[:40],
        )
        return "|".join(parts)

    def register(self, arm_id: str, context: Mapping[str, Any] | None = None) -> None:
        key = (arm_id or "").strip()
        if not key or len(key) > 120:
            raise ValueError("arm_id must contain 1–120 characters")
        self.contexts.setdefault(self.context_key(context), {}).setdefault(key, ArmStats())

    def select(self, context: Mapping[str, Any] | None = None) -> str:
        bucket = self.contexts.get(self.context_key(context), {})
        if not bucket:
            raise ValueError("Register at least one validated configuration for this context.")
        for key, stats in bucket.items():
            if stats.trials == 0:
                return key
        total = sum(item.trials for item in bucket.values())
        return max(bucket, key=lambda key: bucket[key].mean_reward
                   + sqrt(2.0 * log(max(total, 1)) / bucket[key].trials))

    def record(self, arm_id: str, reward: float, context: Mapping[str, Any] | None = None) -> ArmStats:
        value = float(reward)
        if not -1.0 <= value <= 1.0:
            raise ValueError("reward must be normalized to the range [-1, 1]")
        self.register(arm_id, context)
        stats = self.contexts[self.context_key(context)][arm_id]
        stats.trials += 1
        stats.mean_reward += (value - stats.mean_reward) / stats.trials
        return stats

    def to_dict(self) -> dict[str, Any]:
        return {"version": 2, "contexts": {
            context: {arm: asdict(stats) for arm, stats in arms.items()}
            for context, arms in self.contexts.items()
        }}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "PreferenceBandit":
        result = cls()
        raw_contexts = (payload or {}).get("contexts", {})
        if not isinstance(raw_contexts, Mapping):
            return result
        for context, arms in raw_contexts.items():
            if not isinstance(context, str) or not isinstance(arms, Mapping):
                continue
            bucket = {}
            for arm, raw in arms.items():
                if not isinstance(arm, str) or not isinstance(raw, Mapping):
                    continue
                trials = max(0, int(raw.get("trials", 0)))
                reward = float(raw.get("mean_reward", 0.0))
                if -1.0 <= reward <= 1.0:
                    bucket[arm] = ArmStats(trials=trials, mean_reward=reward)
            if bucket:
                result.contexts[context] = bucket
        return result

    @staticmethod
    def combine_feedback(scores: Mapping[str, float], weights: Mapping[str, float]) -> float:
        active = [(name, float(value), float(weights.get(name, 0.0)))
                  for name, value in scores.items() if name in weights]
        active = [(name, value, weight) for name, value, weight in active if weight > 0]
        if not active:
            raise ValueError("At least one positively weighted feedback score is required.")
        if any(not -1.0 <= value <= 1.0 for _, value, _ in active):
            raise ValueError("Feedback scores must be normalized to [-1, 1].")
        denominator = sum(weight for _, _, weight in active)
        return sum(value * weight for _, value, weight in active) / denominator
