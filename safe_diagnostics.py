"""Safe, bounded diagnostics for errors shown in the web UI."""
from __future__ import annotations

import re

_POSTGRES_CREDENTIALS = re.compile(r"(?i)(postgres(?:ql)?://)[^\s@]+@")
_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)\b(password|token|secret|api[_-]?key)\b(\s*[:=]\s*)([^,;\s]+)"
)


def format_exception_detail(exc: BaseException, *, limit: int = 500) -> str:
    """Return a bounded exception summary with common credentials redacted."""
    if limit < 1:
        raise ValueError("limit must be positive")
    detail = f"{type(exc).__name__}: {exc}".strip()
    detail = _POSTGRES_CREDENTIALS.sub(r"\1[redacted]@", detail)
    detail = _SENSITIVE_ASSIGNMENT.sub(r"\1\2[redacted]", detail)
    detail = re.sub(r"[\r\n\t]+", " ", detail)
    detail = re.sub(r"\s{2,}", " ", detail).strip()
    return (detail[:limit] or "Unknown startup error.")
