"""Tavily web search service."""

from datetime import UTC, datetime
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.utils.tracing import traced

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
MAX_QUERY_WORDS = 16
MAX_QUERY_CHARS = 140


class TavilyService:
    """Small Tavily client for concise web/news search queries."""

    def __init__(self, api_key: str | None = None, timeout_seconds: float | None = None) -> None:
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.tavily_api_key
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else settings.tavily_timeout_seconds

    @traced("tavily.search", run_type="tool")
    def search(self, query: str, max_results: int | None = None) -> dict[str, Any]:
        """Search Tavily and normalize article metadata."""

        concise_query = _validate_concise_query(query)
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY is required for Tavily search.")

        settings = get_settings()
        result_limit = max_results if max_results is not None else settings.tavily_max_results
        payload = {
            "api_key": self.api_key,
            "query": concise_query,
            "search_depth": "basic",
            "topic": "news",
            "max_results": result_limit,
            "include_answer": False,
            "include_raw_content": False,
        }

        response_data = self._post(payload)
        raw_results = response_data.get("results") or []
        retrieved_at = _now()
        results = [_normalize_result(item, retrieved_at) for item in raw_results if item]

        return {
            "query": concise_query,
            "results": results,
            "retrieved_at": retrieved_at,
        }

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        reraise=True,
    )
    @traced("tavily.post", run_type="tool")
    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(TAVILY_SEARCH_URL, json=payload)
            response.raise_for_status()
            return response.json()


def _validate_concise_query(query: str) -> str:
    concise_query = " ".join(query.split())
    if not concise_query:
        raise ValueError("Tavily query is required.")
    if len(concise_query) > MAX_QUERY_CHARS or len(concise_query.split()) > MAX_QUERY_WORDS:
        raise ValueError("Tavily query must be concise and search-oriented, not a long prompt.")
    return concise_query


def _normalize_result(item: dict[str, Any], retrieved_at: str) -> dict[str, Any]:
    return {
        "title": item.get("title") or "Untitled source",
        "url": item.get("url") or "",
        "published_at": item.get("published_date") or item.get("published_at"),
        "snippet": item.get("content") or item.get("snippet"),
        "score": item.get("score"),
        "retrieved_at": retrieved_at,
    }


def _now() -> str:
    return datetime.now(UTC).isoformat()
