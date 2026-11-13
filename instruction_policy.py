"""Instruction-aware preservation and scene-integrity guidance for image prompts.

This module adds conservative guidance for unrequested attributes and for the
structural completeness of objects that remain in the scene. Explicit user edits
and removals always take priority.
"""
from __future__ import annotations

import re


_CHANGE_ACTION = re.compile(
    r"\b(?:change|replace|alter|different|new|make|give|put|turn|move|reposition|"
    r"add|remove|delete|erase|hide|crop|zoom|reframe|switch|set|adjust|rotate|"
    r"transform|brighten|darken)\b",
    re.I,
)

_CHANGE_PATTERNS = {
    "identity": re.compile(r"\b(?:person|identity|face|facial identity)\b", re.I),
    "expression": re.compile(r"\b(?:expression|smil(?:e|ing)|laugh(?:ing)?|frown(?:ing)?|angry|sad|surprised|serious|happy|happier)\b", re.I),
    "pose": re.compile(r"\b(?:pose|body position|posture|sitting|seated|standing|lying|kneeling|crouching|leaning|walking|running|jumping|turn(?:ed|ing)? around|face left|face right|head turn|raise(?:d|ing)? (?:an? )?(?:arm|hand|leg)|lower(?:ed|ing)? (?:an? )?(?:arm|hand|leg))\b", re.I),
    "body_shape": re.compile(r"\b(?:body shape|body type|body proportions|build|muscular|athletic build|slim(?:mer)?|thinner|thicker|curvier|weight|height|taller|shorter|heavier|broader shoulders|narrower waist)\b", re.I),
    "clothing": re.compile(r"\b(?:clothes|clothing|outfit|dress|shirt|jacket|coat|trousers|pants|jeans|skirt|shoes|boots|uniform|costume|wearing)\b", re.I),
    "background": re.compile(r"\b(?:background|backdrop|setting|environment|location|scenery|room|sky|landscape)\b", re.I),
    "camera": re.compile(r"\b(?:camera|lens|close[ -]?up|wide shot|headshot|waist[ -]?up|full[ -]?body|framing|crop|zoom|viewpoint|perspective|angle)\b", re.I),
    "lighting": re.compile(r"\b(?:lighting|light|shadows|brighten|darken|exposure|golden hour|neon)\b", re.I),
}


def _requests_attribute_change(
    prompt: str,
    attribute_pattern: re.Pattern[str],
    *,
    category: str = "",
) -> bool:
    """Require a nearby edit verb; merely describing an existing detail is not an edit."""
    for match in attribute_pattern.finditer(prompt):
        prefix = prompt[max(0, match.start() - 100):match.start()]
        # Keep clauses separate so "change the background but keep the outfit"
        # does not accidentally classify the outfit as a requested change.
        clause = re.split(
            r"[,.;!?]\s*|\b(?:but|while|except|keep|preserve|leave)\b",
            prefix,
            flags=re.I,
        )[-1]
        if re.search(
            r"\b(?:do\s+not|don't|never|avoid)\s+"
            r"(?:change|replace|alter|modify|remove|delete|erase)\b",
            clause,
            re.I,
        ):
            continue
        if category == "identity":
            after = prompt[match.end():]
            if re.match(
                r"\s*(?:(?:'s|’s)\s*)?(?:expression|pose|posture|clothing|outfit|body shape|"
                r"body proportions|hair|hairstyle|makeup)\b|\s+to\s+(?:a\s+)?"
                r"(?:smile|laugh|frown|angry|sad|surprised|serious|happy)\b",
                after,
                re.I,
            ):
                continue
            if re.search(r"\b(?:change|replace|alter|swap|different|new)\b", clause, re.I):
                return True
            if re.search(r"\bmake\b[^,.;!?]{0,60}\b(?:look\s+like|into)\b", clause, re.I):
                return True
        elif _CHANGE_ACTION.search(clause):
            return True
    return False

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
    r"\b(?:remove|delete|erase|take\s+out|hide|without)\b",
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
        before_subject = tail[:subject.start()]
        after_subject = tail[subject.end():]
        # Both "remove the clock hands" and "remove the hands from the clock"
        # target a component, not the entire object.
        if _COMPONENT_WORDS.search(before_subject):
            continue
        if _COMPONENT_WORDS.match(after_subject.strip().lstrip(" '’s")):
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
    # Prompt preview may already have compiled this policy. Recompute from the
    # user's original prompt only, so the final generation handoff is idempotent.
    source = source.split("Instruction-aware guidance:", 1)[0].strip()

    changed = {
        name for name, pattern in _CHANGE_PATTERNS.items()
        if _requests_attribute_change(source, pattern, category=name)
    }
    guidance: list[str] = []

    if has_trained_identity and "identity" not in changed:
        guidance.append(
            "Keep the selected trained person's identity and stable facial structure consistent; "
            "a requested expression change is allowed without changing who the person is."
        )
    elif has_reference_image and "identity" not in changed:
        guidance.append("Keep the reference subject's identity and stable facial structure consistent.")

    if has_reference_image:
        guidance.append(
            "Edit the supplied reference selectively: change only what the user explicitly requested; "
            "keep all other visible details, object count, spatial relationships and scene structure consistent. "
            "Do not silently add, remove, duplicate or rearrange unrequested elements."
        )
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
        digital_timepiece = (
            "clock" in structural_core.lower()
            and bool(re.search(r"\bdigital\b", source, re.I))
        )
        if digital_timepiece:
            details.append("a clear, correctly rendered digital time display")
        else:
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
