"""Human-plus-heuristic evaluation for image and face-dataset outputs.
Automated metrics are conservative: identity, expression, and body proportions
need explicit human ratings until dedicated evaluators are validated.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Mapping
from .reward import PreferenceBandit

DIMENSIONS = ("identity", "angle", "expression", "body_proportions", "image_quality", "prompt_adherence")
DEFAULT_WEIGHTS = {"identity": 2.0, "angle": 1.5, "expression": 1.5,
                   "body_proportions": 1.5, "image_quality": 1.0, "prompt_adherence": 1.5}

@dataclass(frozen=True)
class EvaluationReport:
    reward: float
    scores: dict[str, float]
    weights: dict[str, float]
    sources: dict[str, str]
    penalties: tuple[str, ...] = ()
    needs_human_review: tuple[str, ...] = ()
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def _normalized_rating(value: Any) -> float:
    number = float(value)
    if 1.0 <= number <= 5.0:
        return (number - 3.0) / 2.0
    if -1.0 <= number <= 1.0:
        return number
    raise ValueError("Ratings must be -1..1 or 1..5.")

def evaluate_generation(*, human_scores: Mapping[str, Any] | None = None,
                         pose_status: str | None = None,
                         technical_quality: Mapping[str, Any] | None = None,
                         requested_dimensions: tuple[str, ...] | list[str] | None = None,
                         weights: Mapping[str, float] | None = None) -> EvaluationReport:
    scores: dict[str, float] = {}
    sources: dict[str, str] = {}
    penalties: list[str] = []
    review: list[str] = []
    for dimension, value in (human_scores or {}).items():
        if dimension not in DIMENSIONS:
            raise ValueError("Unsupported feedback dimension: " + str(dimension))
        scores[dimension] = _normalized_rating(value)
        sources[dimension] = "human"
    if pose_status in {"passed", "needs_review"} and "angle" not in scores:
        scores["angle"] = 1.0 if pose_status == "passed" else -0.65
        sources["angle"] = "pose_estimator"
        if pose_status == "needs_review":
            penalties.append("Requested head pose did not pass automatic validation.")
            review.append("angle")
    quality = technical_quality or {}
    if quality.get("status") == "technical_checks_passed" and "image_quality" not in scores:
        scores["image_quality"] = 0.65
        sources["image_quality"] = "technical_heuristic"
    elif quality.get("status") == "needs_review" and "image_quality" not in scores:
        scores["image_quality"] = -0.65
        sources["image_quality"] = "technical_heuristic"
        penalties.extend(str(item) for item in quality.get("reasons", [])[:8])
        review.append("image_quality")
    for dimension in dict.fromkeys(requested_dimensions or DIMENSIONS):
        if dimension in DIMENSIONS and dimension not in scores:
            review.append(dimension)
    selected_weights = {key: float((weights or DEFAULT_WEIGHTS).get(key, 0.0)) for key in scores}
    reward = PreferenceBandit.combine_feedback(scores, selected_weights) if scores else 0.0
    return EvaluationReport(round(reward, 5), scores, selected_weights, sources,
                            tuple(dict.fromkeys(penalties)), tuple(dict.fromkeys(review)))
