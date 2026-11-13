"""Instruction-aware preservation and scene-integrity guidance for image prompts.

This module adds conservative guidance for unrequested attributes and for the
structural completeness of objects that remain in the scene. Explicit user edits
and removals always take priority.
"""
from __future__ import annotations

import re


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

# Each object has a non-removable structural core plus individually removable
# components. This avoids asking the model to restore a part the user removed.
_OBJECT_RULES = (
    (
        re.compile(r"\b(?:clock|clocks|watch|watches|timepiece)\b", re.I),
        "a coherent clock or watch with a properly formed dial and clear central pivot",
        (
            (re.compile(r"\b(?:hands?|hour hand|minute hand|second hand)\b", re.I), "correctly attached hour/minute hands"),
            (re.compile(r"\b(?:numbers?|numerals?|digits?|markers?)\b", re.I), "clear, evenly spaced numerals or markers"),
        ),
    ),
    (
        re.compile(r"\b(?:car|cars|vehicle|vehicles|automobile|truck|van)\b", re.I),
        "a coherent vehicle body with connected panels and plausible proportions",
        (
            (re.compile(r"\b(?:wheel|wheels|tyres?|tires?)\b", re.I), "aligned, correctly attached wheels"),
            (re.compile(r"\b(?:windows?|windshield|windscreen)\b", re.I), "properly fitted windows"),
        ),
    ),
    (
        re.compile(r"\b(?:bicycle|bike|bicycles|bikes)\b", re.I),
        "a coherent bicycle frame with a connected fork and plausible mechanical geometry",
        (
            (re.compile(r"\b(?:wheel|wheels|tyres?|tires?)\b", re.I), "two aligned wheels"),
            (re.compile(r"\b(?:pedals?|chain)\b", re.I), "a connected chain and pedals"),
            (re.compile(r"\b(?:handlebars?)\b", re.I), "properly attached handlebars"),
        ),
    ),
    (
        re.compile(r"\b(?:chair|chairs|table|tables|stool|desk)\b", re.I),
        "stable, coherent furniture with a properly formed seat or tabletop and plausible joints",
        (
            (re.compile(r"\b(?:leg|legs|support|supports)\b", re.I), "securely connected legs or supports"),
        ),
    ),
    (
        re.compile(r"\b(?:building|house|home|cottage|tower|skyscraper|hut)\b", re.I),
        "a coherent building with connected structural elements and consistent perspective",
        (
            (re.compile(r"\b(?:roof)\b", re.I), "a properly fitted roof"),
            (re.compile(r"\b(?:wall|walls)\b", re.I), "connected walls"),
            (re.compile(r"\b(?:door|doors|window|windows)\b", re.I), "plausible doors and windows"),
        ),
    ),
    (
        re.compile(r"\b(?:cup|mug|glass|bottle|jar)\b", re.I),
        "a complete, physically coherent vessel with a continuous body",
        (
            (re.compile(r"\b(?:handle|handles)\b", re.I), "a correctly connected handle where applicable"),
            (re.compile(r"\b(?:base|opening|lid)\b", re.I), "a coherent base and opening or lid"),
        ),
    ),
    (
        re.compile(r"\b(?:book|notebook|magazine)\b", re.I),
        "a coherent publication with aligned pages",
        (
            (re.compile(r"\b(?:spine)\b", re.I), "a connected spine"),
            (re.compile(r"\b(?:cover|covers)\b", re.I), "properly aligned covers"),
        ),
    ),
)

_WHOLE_OBJECT_REMOVAL = re.compile(
    r"\b(?:remove|delete|erase|take\s+out|hide|without)\b[\w\s-]{0,24}",
    re.I,
)
_COMPONENT_WORDS = re.compile(
    r"\b(?:hands?|hour|minute|second|wheels?|tires?|tyres?|legs?|supports?|"
    r"windows?|doors?|roof|walls?|pages?|spine|covers?|handles?|pedals?|"
    r"chain|base|opening|lid|numerals?|digits?|markers?)\b",
    re.I,
)
_HUMAN_SUBJECT = re.compile(
    r"\b(?:person|people|woman|women|man|men|girl|boy|human|portrait|"
    r"face|character|model|dancer|athlete|bride|groom)\b",
    re.I,
)


def _whole_object_removed(prompt: str, subject_pattern: re.Pattern[str]) -> bool:
    for match in _WHOLE_OBJECT_REMOVAL.finditer(prompt):
        tail = prompt[match.end():match.end() + 45]
        subject = subject_pattern.search(tail)
        if not subject:
            continue
        after_subject = tail[subject.end():]
        # "Remove the clock hands" removes a component, not the clock itself.
        if _COMPONENT_WORDS.match(after_subject.strip()):
            continue
        return True
    return False


def _component_removed(prompt: str, component_pattern: re.Pattern[str]) -> bool:
    for match in _WHOLE_OBJECT_REMOVAL.finditer(prompt):
        tail = prompt[match.end():match.end() + 55]
        component = component_pattern.search(tail)
        if component:
            return True
    return bool(re.search(
        r"\b(?:no|omit)\b[\w\s-]{0,20}" + component_pattern.pattern,
        prompt,
        re.I,
    ))


def build_instruction_guidance(
    prompt: str,
    *,
    has_reference_image: bool = False,
    has_trained_identity: bool = False,
) -> str:
    """Return prompt guidance while allowing every explicit user-requested edit."""
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
        if "identity" not in changed:
            if has_trained_identity:
                guidance.append(
                    "Keep the selected trained person's identity and stable facial structure consistent; "
                    "a requested expression change is allowed without changing who the person is."
                )
            else:
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
    for subject_pattern, structural_core, components in _OBJECT_RULES:
        if not subject_pattern.search(source) or _whole_object_removed(source, subject_pattern):
            continue
        details = [structural_core]
        for component_pattern, component_description in components:
            if not _component_removed(source, component_pattern):
                details.append(component_description)
        object_guidance.append(", ".join(details))

    if object_guidance:
        guidance.append(
            "Keep depicted objects complete, recognizable and physically coherent: "
            + "; ".join(object_guidance)
            + ". Do not invent components the user explicitly asked to remove."
        )

    unique = list(dict.fromkeys(guidance))
    return "Instruction-aware guidance: " + " ".join(unique) if unique else ""
