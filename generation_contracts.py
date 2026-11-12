"""Request-contract helpers shared by the image-generation API and tests."""
from __future__ import annotations


def preserve_generation_text(value: object, field_name: str) -> str:
    """Return the complete user-supplied text without trimming or character caps."""
    text = value if isinstance(value, str) else str(value or "")
    if "\x00" in text:
        raise ValueError(f"{field_name} contains an unsupported null character.")
    return text
