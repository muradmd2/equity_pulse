"""LangSmith tracing helpers with sensitive input redaction."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from langsmith import traceable

from app.config import get_settings

SENSITIVE_KEY_PARTS = ("api_key", "authorization", "token", "secret", "password")
REDACTED = "[REDACTED]"


def configure_langsmith_environment() -> bool:
    """Expose settings-based LangSmith config to LangChain/LangGraph tracing."""

    settings = get_settings()
    if not settings.langsmith_tracing or not settings.langsmith_api_key:
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    return True


def traced(name: str, run_type: str = "chain") -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Create a LangSmith traceable decorator with shared redaction."""

    return traceable(
        name=name,
        run_type=run_type,
        #process_inputs=redact_sensitive_data,
        #process_outputs=redact_sensitive_data,
    )


def redact_sensitive_data(value: Any) -> Any:
    """Recursively redact API keys and similar secrets from trace payloads."""

    if isinstance(value, dict):
        return {
            key: _redacted_value(key, item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive_data(item) for item in value)
    return value


def _is_sensitive_key(key: str) -> bool:
    normalized_key = key.lower()
    return any(part in normalized_key for part in SENSITIVE_KEY_PARTS)


def _redacted_value(key: Any, value: Any) -> Any:
    normalized_key = str(key).lower()
    if normalized_key == "self":
        return f"<{value.__class__.__name__}>"
    if _is_sensitive_key(normalized_key):
        return REDACTED
    return redact_sensitive_data(value)
