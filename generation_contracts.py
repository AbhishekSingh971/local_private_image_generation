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


MAX_SD15_PROMPT_TOKENS = 300
"""Safety cap for chunked SD 1.5 prompts (four 75-content-token CLIP chunks)."""


def _raw_token_count(tokenizer, text: str) -> int:
    encoded = tokenizer(text, add_special_tokens=False, truncation=False)
    input_ids = encoded.get("input_ids") if hasattr(encoded, "get") else None
    if input_ids is None:
        raise RuntimeError("The SD 1.5 tokenizer did not return input_ids.")
    shape = getattr(input_ids, "shape", None)
    if shape is not None:
        return int(shape[-1])
    if input_ids and isinstance(input_ids[0], (list, tuple)):
        return len(input_ids[0])
    return len(input_ids)


def encode_sd15_prompt_embeddings(
    pipeline,
    prompt: str,
    negative_prompt: str = "",
    *,
    max_prompt_tokens: int = MAX_SD15_PROMPT_TOKENS,
) -> dict:
    """Encode long SD 1.5 prompts as multiple CLIP chunks without truncation.

    CLIP still encodes at its native 77-position window. Compel splits longer
    text into 75-content-token chunks (BOS/EOS included per chunk), then the
    embeddings are padded to equal lengths for classifier-free guidance.
    """
    tokenizer = getattr(pipeline, "tokenizer", None)
    text_encoder = getattr(pipeline, "text_encoder", None)
    if tokenizer is None or text_encoder is None:
        raise RuntimeError("The selected SD 1.5 pipeline has no tokenizer or text encoder.")

    positive_text = str(prompt or "")
    negative_text = str(negative_prompt or "")
    positive_count = _raw_token_count(tokenizer, positive_text)
    negative_count = _raw_token_count(tokenizer, negative_text)
    largest_count = max(positive_count, negative_count)
    if largest_count > int(max_prompt_tokens):
        raise ValueError(
            f"SD 1.5 supports up to {int(max_prompt_tokens)} prompt tokens with chunked CLIP encoding "
            f"({(int(max_prompt_tokens) + 74) // 75} chunks). Your prompt contains {largest_count} tokens. "
            "Keep the most important subject, action, identity, pose and scene details."
        )

    # The regular Diffusers text path is faster and identical for prompts that
    # fit in CLIP's 77 positions (75 content tokens plus BOS/EOS).
    if largest_count <= 75:
        return {
            "prompt_embeds": None,
            "negative_prompt_embeds": None,
            "prompt_token_count": positive_count + 2,
            "negative_prompt_token_count": negative_count + 2 if negative_text else 0,
            "chunk_count": 1,
            "max_prompt_tokens": int(max_prompt_tokens),
        }

    try:
        from compel import Compel, ReturnedEmbeddingsType
    except ImportError as exc:
        raise RuntimeError(
            "Long-prompt support is not installed. Install the project requirements, including compel."
        ) from exc

    embedder = getattr(pipeline, "_private_face_compel", None)
    if embedder is None:
        device = getattr(pipeline, "_execution_device", None)
        embedder = Compel(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            truncate_long_prompts=False,
            returned_embeddings_type=ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
            device=str(device) if device is not None else None,
        )
        pipeline._private_face_compel = embedder

    positive_embeds = embedder.build_conditioning_tensor(positive_text)
    negative_embeds = embedder.build_conditioning_tensor(negative_text)
    positive_embeds, negative_embeds = embedder.pad_conditioning_tensors_to_same_length(
        [positive_embeds, negative_embeds]
    )
    chunk_count = max(1, (largest_count + 74) // 75)
    return {
        "prompt_embeds": positive_embeds,
        "negative_prompt_embeds": negative_embeds,
        "prompt_token_count": positive_count + 2,
        "negative_prompt_token_count": negative_count + 2 if negative_text else 0,
        "chunk_count": chunk_count,
        "max_prompt_tokens": int(max_prompt_tokens),
    }
