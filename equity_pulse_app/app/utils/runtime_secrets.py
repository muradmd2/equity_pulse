"""Request-scoped runtime secrets that should not be persisted or traced."""

from __future__ import annotations

_OPENAI_API_KEYS_BY_SESSION: dict[str, str] = {}


def set_request_openai_api_key(session_id: str, api_key: str | None) -> None:
    """Store or clear the request OpenAI API key for a session."""

    if api_key:
        _OPENAI_API_KEYS_BY_SESSION[session_id] = api_key
    else:
        _OPENAI_API_KEYS_BY_SESSION.pop(session_id, None)


def get_request_openai_api_key(session_id: str | None) -> str | None:
    """Return the request OpenAI API key without persisting it in graph state."""

    if not session_id:
        return None
    return _OPENAI_API_KEYS_BY_SESSION.get(session_id)


def clear_request_openai_api_key(session_id: str) -> None:
    """Remove a request OpenAI API key after graph execution finishes."""

    _OPENAI_API_KEYS_BY_SESSION.pop(session_id, None)
