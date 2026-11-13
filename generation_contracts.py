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



def _encoded_token_count(encoded: object) -> int:
    input_ids = encoded.get("input_ids") if hasattr(encoded, "get") else None
    if input_ids is None:
        raise RuntimeError("Tokenizer did not return input_ids; prompt length could not be verified.")
    shape = getattr(input_ids, "shape", None)
    if shape is not None:
        return int(shape[-1])
    if input_ids and isinstance(input_ids[0], (list, tuple)):
        return len(input_ids[0])
    return len(input_ids)


def count_tokens_without_truncation(tokenizer, text: str) -> int:
    """Count actual model input tokens without asking the tokenizer to truncate."""
    if tokenizer is None or not callable(tokenizer):
        raise RuntimeError("The selected model tokenizer is unavailable; generation was stopped.")
    encoded = tokenizer(text, add_special_tokens=True, truncation=False)
    return _encoded_token_count(encoded)


def validate_diffusion_prompt_capacity(
    pipeline,
    prompt: str,
    negative_prompt: str = "",
) -> dict[str, int]:
    """Validate positive and negative prompts against every tokenizer in a pipeline."""
    tokenizers = []
    for attribute in ("tokenizer", "tokenizer_2"):
        tokenizer = getattr(pipeline, attribute, None)
        if tokenizer is not None:
            tokenizers.append((attribute, tokenizer))
    if not tokenizers:
        raise RuntimeError(
            "Could not inspect the selected model tokenizer. Generation was stopped to avoid silent prompt truncation."
        )

    counts: dict[str, int] = {}
    for attribute, tokenizer in tokenizers:
        maximum = getattr(tokenizer, "model_max_length", None)
        if not isinstance(maximum, int) or maximum < 1 or maximum > 100000:
            raise RuntimeError(
                f"Could not determine the {attribute} context limit. Generation was stopped to avoid silent truncation."
            )
        for field_name, value in (("Prompt", prompt), ("Negative prompt", negative_prompt)):
            if not value:
                continue
            count = count_tokens_without_truncation(tokenizer, value)
            counts[f"{attribute}_{field_name.lower().replace(' ', '_')}"] = count
            ensure_token_capacity(count, maximum, f"{field_name} ({attribute})")
    return counts


def count_flux2_prompt_tokens(pipeline, prompt: str) -> int:
    """Mirror FLUX.2 Klein's Qwen chat-template formatting before counting tokens."""
    tokenizer = getattr(pipeline, "tokenizer", None)
    if tokenizer is None or not callable(tokenizer):
        raise RuntimeError(
            "Could not inspect the FLUX.2 tokenizer. Generation was stopped to avoid silent prompt truncation."
        )
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError(
            "The FLUX.2 tokenizer does not expose its chat template; generation was stopped to avoid silent truncation."
        )
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    count = count_tokens_without_truncation(tokenizer, formatted)
    maximum = int(getattr(pipeline, "tokenizer_max_length", 512))
    return ensure_token_capacity(count, maximum, "FLUX.2 prompt")
