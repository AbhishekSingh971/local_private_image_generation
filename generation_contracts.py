"""Request-contract helpers shared by the image-generation API and tests."""
from __future__ import annotations


def preserve_generation_text(value: object, field_name: str) -> str:
    """Return the complete user-supplied text without trimming or character caps."""
    text = value if isinstance(value, str) else str(value or "")
    if "\x00" in text:
        raise ValueError(f"{field_name} contains an unsupported null character.")
    return text



def ensure_token_capacity(token_count: int, maximum_tokens: int, field_name: str = "Prompt") -> int:
    """Fail explicitly rather than allowing a model tokenizer to silently truncate."""
    count = int(token_count)
    maximum = int(maximum_tokens)
    if count < 0 or maximum < 1:
        raise ValueError("Token count and maximum token capacity must be valid positive values.")
    if count > maximum:
        raise ValueError(
            f"{field_name} contains {count} tokens, exceeding this model's {maximum}-token input limit. "
            "The text was not shortened. Split or simplify the request, or select a model with a larger context."
        )
    return count
