"""Identity-token prompt conditioning shared by local and remote generation."""

from __future__ import annotations

import re


def ensure_face_identity_trigger(prompt: str, token: str) -> str:
    """Add the exact profile trigger learned during training, at most once."""
    text = str(prompt or "").strip()
    trigger = str(token or "").strip()
    if not trigger:
        return text
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", trigger):
        raise ValueError("The selected face profile has an invalid identity trigger token.")

    token_pattern = re.compile(
        rf"(?<![A-Za-z0-9_-]){re.escape(trigger)}(?![A-Za-z0-9_-])",
        flags=re.IGNORECASE,
    )
    if token_pattern.search(text):
        return text

    identity_condition = f"a photo of {trigger} person"
    return f"{identity_condition}, {text}" if text else identity_condition
