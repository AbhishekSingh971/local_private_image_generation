"""Lightweight preference learning for generation-setting selection.

This is a contextual-bandit foundation, not a claim that full RL or model-weight
training is happening. Persist observations through the application's database
adapter before using it across restarts or multiple workers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Mapping


@dataclass
class ArmStats:
    trials: int = 0
    mean_reward: float = 0.0


@dataclass
class PreferenceBandit:
    """UCB1 selector over named, prevalidated generation configurations."""

    arms: dict[str, ArmStats] = field(default_factory=dict)

    def register(self, arm_id: str) -> None:
        key = (arm_id or "").strip()
        if not key:
            raise ValueError("arm_id is required")
        self.arms.setdefault(key, ArmStats())

    def select(self) -> str:
        if not self.arms:
            raise ValueError("Register at least one generation configuration.")
        for key, stats in self.arms.items():
            if stats.trials == 0:
                return key
        total = sum(item.trials for item in self.arms.values())
        return max(
            self.arms,
            key=lambda key: self.arms[key].mean_reward
            + sqrt(2.0 * __import__("math").log(total) / self.arms[key].trials),
        )

    def record(self, arm_id: str, reward: float) -> ArmStats:
        if not -1.0 <= float(reward) <= 1.0:
            raise ValueError("reward must be normalized to the range [-1, 1]")
        self.register(arm_id)
        stats = self.arms[arm_id]
        stats.trials += 1
        stats.mean_reward += (float(reward) - stats.mean_reward) / stats.trials
        return stats

    @staticmethod
    def combine_feedback(scores: Mapping[str, float], weights: Mapping[str, float]) -> float:
        """Combine normalized user/evaluator scores; missing dimensions are ignored."""
        active = [(name, float(value), float(weights.get(name, 0.0)))
                  for name, value in scores.items() if name in weights]
        active = [(name, value, weight) for name, value, weight in active if weight > 0]
        if not active:
            raise ValueError("At least one positively weighted feedback score is required.")
        if any(not -1.0 <= value <= 1.0 for _, value, _ in active):
            raise ValueError("Feedback scores must be normalized to [-1, 1].")
        denominator = sum(weight for _, _, weight in active)
        return sum(value * weight for _, value, weight in active) / denominator
