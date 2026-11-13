"""Instruction-aware preservation and scene-integrity guidance for image prompts.

This module does not override the user's prompt. It adds conservative guidance for
unrequested attributes and for the structural completeness of objects that remain
in the scene. Explicit removals and requested edits always take precedence.
"""
from __future__ import annotations

import re


_CHANGE_WORDS = r"(?:change|alter|modify|replace|swap|remove|delete|erase|hide|add|make|turn|move|reposition|redesign|transform|crop|zoom|reframe|paint|recolor|colour|color)"

_CHANGE_PATTERNS = {
    "identity": re.compile(r"\b(?:change|replace|alter|different|new)\s+(?:the\s+)?(?:person|identity|face|facial identity)\b|\bmake\s+(?:them|the person)\s+(?:look like|into)\b", re.I),
    "expression": re.compile(r"\b(?:expression|smil(?:e|ing)|laugh(?:ing)?|frown(?:ing)?|angry|sad|surprised|serious|happy)\b", re.I),
    "pose": re.compile(r"\b(?:pose|body position|posture|sitting|seated|standing|lying|kneeling|crouching|leaning|walking|running|jumping|turn(?:ed|ing)? around|raise(?:d|ing)? (?:an? )?(?:arm|hand|leg)|lower(?:ed|ing)? (?:an? )?(?:arm|hand|leg))\b", re.I),
    "body_shape": re.compile(r"\b(?:body shape|body type|body proportions|build|muscular|athletic build|slim(?:mer)?|thicker|curvier|weight|height|broader shoulders|narrower waist)\b", re.I),
    "clothing": re.compile(r"\b(?:clothes|clothing|outfit|dress|shirt|jacket|coat|trousers|pants|jeans|skirt|shoes|boots|uniform|costume|wearing)\b", re.I),
    "background": re.compile(r"\b(?:background|backdrop|setting|environment|location|scenery|room|sky|landscape)\b", re.I),
    "camera": re.compile(r"\b(?:camera|lens|close[ -]?up|wide shot|headshot|waist[ -]?up|full[ -]?body|framing|crop|zoom|viewpoint|perspective|angle of view)\b", re.I),
    "lighting": re.compile(r"\b(?:lighting|light|shadows|brighten|darken|exposure|golden hour|neon)\b", re.I),
}

_OBJECT_RULES = (
    (
        "clock|watch",
        re.compile(r"\b(?:clock|clocks|watch|watches|timepiece)\b", re.I),
        "a coherent clock or watch with a properly formed dial, a clear central pivot and correctly attached hour/minute hands",
        (
            (re.compile(r"\b(?:hands?|hour hand|minute hand|second hand)\b", re.I), "clock hands"),
            (re.compile(r"\b(?:numbers?|numerals?|digits?|markers?)\b", re.I), "clock numerals or markers"),
        ),
    ),
    (
        "car|vehicle",
        re.compile(r"\b(?:car|cars|vehicle|vehicles|automobile|truck|van)\b", re.I),
        "a structurally coherent vehicle with aligned wheels, connected body panels, windows and plausible proportions",
        (
            (re.compile(r"\b(?:wheel|wheels|tyres?|tires?)\b", re.I), "vehicle wheels"),
            (re.compile(r"\b(?:windows?|windshield|windscreen)\b", re.I), "vehicle windows"),
        ),
    ),
    (
        "bicycle",
        re.compile(r"\b(?:bicycle|bike|bicycles|bikes)\b", re.I),
        "a complete bicycle with a coherent frame, two aligned wheels, fork, handlebars, chain and pedals",
        (
            (re.compile(r"\b(?:wheel|wheels|tyres?|tires?)\b", re.I), "bicycle wheels"),
            (re.compile(r"\b(?:pedals?|chain|handlebars?)\b", re.I), "bicycle drivetrain or handlebars"),
        ),
    ),
    (
        "chair|table",
        re.compile(r"\b(?:chair|chairs|table|tables|stool|desk)\b", re.I),
        "a stable piece of furniture with connected surfaces, supports and plausible joints",
        (
            (re.compile(r"\b(?:leg|legs|support|supports)\b", re.I), "furniture legs or supports"),
        ),
    ),
    (
        "building",
        re.compile(r"\b(?:building|house|home|cottage|tower|skyscraper|hut)\b", re.I),
        "a coherent building with connected walls, roof, openings and consistent perspective",
        (
            (re.compile(r"\b(?:roof|wall|walls|door|doors|window|windows)\b", re.I), "building components"),
        ),
    ),
    (
        "cup|glass|bottle",
        re.compile(r"\b(?:cup|mug|glass|bottle|jar)\b", re.I),
        "a complete, physically coherent vessel with a continuous body and correctly connected opening, base and handle where applicable",
        (
            (re.compile(r"\b(?:handle|handles|base|opening|lid)\b", re.I), "vessel components"),
        ),
    ),
    (
        "book",
        re.compile(r"\b(?:book|notebook|magazine)\b", re.I),
        "a coherent book with aligned pages, a connected spine and covers",
        (
            (re.compile(r"\b(?:page|pages|spine|cover|covers)\b", re.I), "book components"),
        ),
    ),
)

_HUMAN_SUBJECT = re.compile(
    r"\b(?:person|people|woman|women|man|men|girl|boy|human|portrait|"
    r"face|character|model|dancer|athlete|bride|groom)\b",
    re.I,
)


def _explicitly_removes(prompt: str, subject_pattern: re.Pattern[str]) -> bool:
    """Detect removal of the whole object, not just one of its components."""
    words = subject_pattern.pattern
    return bool(re.search(
        rf"\b(?:remove|delete|erase|take\s+out|hide|without)\b"
        rf"[\w\s-]{{0,24}}(?:{words})",
        prompt,
        re.I,
    ))


def _explicitly_removes_component(prompt: str, component_pattern: re.Pattern[str]) -> bool:
    return bool(re.search(
        r"\b(?:remove|delete|erase|take\s+out|hide|without|no|omit)\b"
        r"[\w\s-]{0,24}" + component_pattern.pattern,
        prompt,
        re.I,
    ))


def build_instruction_guidance(
    prompt: str,
    *,
    has_reference_image: bool = False,
    has_trained_identity: bool = False,
) -> str:
    """Return concise prompt guidance without blocking explicit user changes."""
    source = str(prompt or "").strip()
    if not source:
        return ""

    changed = {
        name for name, pattern in _CHANGE_PATTERNS.items()
        if pattern.search(source)
    }
    guidance: list[str] = []

    if has_reference_image:
        guidance.append(
            "Edit the supplied reference selectively: change only what the user explicitly requested; "
            "keep all other visible details, object count, spatial relationships and scene structure consistent. "
            "Do not silently add, remove, duplicate or rearrange unrequested elements."
        )
        if has_trained_identity and "identity" not in changed:
            guidance.append(
                "Keep the selected trained person's identity and stable facial structure consistent; "
                "a requested expression change is allowed without changing who the person is."
            )
        elif "identity" not in changed:
            guidance.append("Keep the reference subject's identity and stable facial structure consistent.")
        if "pose" not in changed:
            guidance.append("Keep the original body pose and posture.")
        elif "body_shape" not in changed:
            guidance.append("The requested pose may change, but keep the person's underlying body proportions consistent.")
        if "body_shape" not in changed:
            guidance.append("Do not unintentionally change body shape or proportions.")
        if "expression" not in changed:
            guidance.append("Keep the original facial expression.")
        if "clothing" not in changed:
            guidance.append("Keep the original clothing and accessories.")
        if "background" not in changed:
            guidance.append("Keep the original background and environment.")
        if "camera" not in changed:
            guidance.append("Keep the original camera viewpoint, framing and perspective.")
        if "lighting" not in changed:
            guidance.append("Keep the original lighting and shadow direction.")

    if _HUMAN_SUBJECT.search(source):
        guidance.append(
            "Render anatomically coherent human anatomy with naturally connected limbs, "
            "plausible joints, complete hands and naturally formed fingers unless the user explicitly requests otherwise."
        )

    object_guidance: list[str] = []
    for _name, subject_pattern, completeness, components in _OBJECT_RULES:
        if not subject_pattern.search(source) or _explicitly_removes(source, subject_pattern):
            continue
        component_requirements = []
        for component_pattern, component_name in components:
            if _explicitly_removes_component(source, component_pattern):
                continue
            if component_pattern.search(source):
                component_requirements.append(component_name)
        detail = completeness
        if component_requirements:
            detail += "; keep " + " and ".join(component_requirements) + " correctly formed"
        object_guidance.append(detail)

    if object_guidance:
        guidance.append(
            "Keep depicted objects complete, recognizable and physically coherent: "
            + "; ".join(object_guidance)
            + ". Do not invent components the user explicitly asked to remove."
        )

    # Preserve order while avoiding duplicate sentences from overlapping categories.
    unique = list(dict.fromkeys(guidance))
    return "Instruction-aware guidance: " + " ".join(unique) if unique else ""
