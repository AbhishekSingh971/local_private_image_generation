"""Structured, provider-backed prompt understanding.

The engine accepts one natural-language prompt and returns task parameters plus
positive/negative prompts. It uses an OpenAI-compatible local or remote endpoint
when configured. Without one, it safely preserves the user's text and applies
small task-neutral defaults rather than pretending to understand semantics.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.request import Request, urlopen


SUPPORTED_TASKS = {"image", "video", "face_training", "dataset_generation"}
UNIVERSAL_NEGATIVE = (
    "low quality, blurry, compression artifacts, malformed details, "
    "unintended duplicate subjects, text artifacts, watermark"
)


class PromptEngineError(RuntimeError):
    """Raised when prompt input or provider output is invalid."""


@dataclass(frozen=True)
class PromptResult:
    task: str
    original_prompt: str
    positive_prompt: str
    negative_prompt: str
    parameters: dict[str, Any] = field(default_factory=dict)
    task_instructions: str = ""
    provider: str = "fallback"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PromptEngine:
    """Turn one user prompt into a structured, task-specific generation request."""

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 45.0,
    ) -> None:
        self.endpoint = (endpoint or os.getenv("PROMPT_LLM_BASE_URL", "")).strip()
        self.model = (model or os.getenv("PROMPT_LLM_MODEL", "")).strip()
        self.api_key = api_key if api_key is not None else os.getenv("PROMPT_LLM_API_KEY", "")
        self.timeout = timeout

    def build(
        self,
        prompt: str,
        *,
        task: str = "image",
        context: dict[str, Any] | None = None,
    ) -> PromptResult:
        original = (prompt or "").strip()
        task_key = (task or "").strip().lower()
        if not original:
            raise PromptEngineError("Enter a prompt before starting this task.")
        if task_key not in SUPPORTED_TASKS:
            raise PromptEngineError(f"Unsupported prompt task: {task_key}")

        safe_context = dict(context or {})
        if self.endpoint and self.model:
            payload = self._request_provider(original, task_key, safe_context)
            return self._validate_provider_result(original, task_key, payload)

        # Fallback is deliberately transparent: no fabricated camera, lighting,
        # scene, identity, or pose attributes are added when no semantic model exists.
        positive = original
        negative = str(safe_context.get("negative_constraints", "")).strip()
        if not negative:
            negative = UNIVERSAL_NEGATIVE
        return PromptResult(
            task=task_key,
            original_prompt=original,
            positive_prompt=positive,
            negative_prompt=negative,
            parameters={"context": safe_context},
            task_instructions="Configure a PROMPT_LLM_BASE_URL and PROMPT_LLM_MODEL "
                              "to enable semantic prompt parsing and enhancement.",
            provider="fallback",
        )

    def _request_provider(
        self, prompt: str, task: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        system = (
            "You are a structured prompt-planning component for a private diffusion "
            "studio. Return ONLY valid JSON with keys positive_prompt, negative_prompt, "
            "parameters, task_instructions. Preserve the user's intent. Do not invent "
            "subjects, people, objects, actions, locations, identity traits, or scene "
            "details. Extract only supported parameters. Generate a concise positive "
            "prompt and task-relevant negative prompt. For video, include motion and "
            "temporal parameters only when supported by the request. For face training "
            "or dataset generation, treat supplied context (profile, angle, expression, "
            "dataset target) as constraints, not optional suggestions. Never return "
            "markdown or ask the user to write positive/negative prompts."
        )
        body = {
            "model": self.model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps({
                    "task": task, "prompt": prompt, "context": context
                }, ensure_ascii=False)},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
            content = raw["choices"][0]["message"]["content"]
            result = json.loads(content)
            if not isinstance(result, dict):
                raise ValueError("Provider response is not a JSON object.")
            return result
        except Exception as exc:
            raise PromptEngineError(
                "Prompt understanding failed. Check the configured prompt model "
                "endpoint/model and retry; the request was not silently downgraded."
            ) from exc

    @staticmethod
    def _validate_provider_result(
        original: str, task: str, data: dict[str, Any]
    ) -> PromptResult:
        positive = data.get("positive_prompt")
        negative = data.get("negative_prompt")
        parameters = data.get("parameters", {})
        instructions = data.get("task_instructions", "")
        if not isinstance(positive, str) or not positive.strip():
            raise PromptEngineError("Prompt model returned an empty positive prompt.")
        if not isinstance(negative, str):
            raise PromptEngineError("Prompt model returned an invalid negative prompt.")
        if not isinstance(parameters, dict) or not isinstance(instructions, str):
            raise PromptEngineError("Prompt model returned an invalid parameter structure.")
        return PromptResult(
            task=task,
            original_prompt=original,
            positive_prompt=positive.strip(),
            negative_prompt=negative.strip(),
            parameters=parameters,
            task_instructions=instructions.strip(),
            provider="openai_compatible",
        )
