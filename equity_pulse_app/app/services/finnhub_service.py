"""Finnhub data access helpers."""

from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.utils.tracing import traced

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"


class FinnhubService:
    """Small Finnhub REST client with normalized return shapes."""

    def __init__(self, api_key: str | None = None, timeout_seconds: float | None = None) -> None:
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.finnhub_api_key
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else settings.finnhub_timeout_seconds

    @traced("finnhub.get_recommendation_trends", run_type="tool")
    def get_recommendation_trends(self, ticker: str) -> dict[str, Any]:
        ticker_symbol = _normalize_ticker(ticker)
        response = self._get("/stock/recommendation", {"symbol": ticker_symbol})
        trends = response if isinstance(response, list) else []
        latest = trends[0] if trends else None

        return {
            "ticker": ticker_symbol,
            "trends": trends,
            "latest": latest,
            "counts": _recommendation_counts(latest),
            "retrieved_at": _now(),
        }

    @traced("finnhub.get_price_target", run_type="tool")
    def get_price_target(self, ticker: str) -> dict[str, Any]:
        ticker_symbol = _normalize_ticker(ticker)
        response = self._get("/stock/price-target", {"symbol": ticker_symbol})
        data = response if isinstance(response, dict) else {}

        return {
            "ticker": ticker_symbol,
            "target_high": data.get("targetHigh"),
            "target_low": data.get("targetLow"),
            "target_mean": data.get("targetMean"),
            "target_median": data.get("targetMedian"),
            "last_updated": data.get("lastUpdated"),
            "retrieved_at": _now(),
        }

    @traced("finnhub.get_company_news", run_type="tool")
    def get_company_news(self, ticker: str, days: int | None = None) -> dict[str, Any]:
        ticker_symbol = _normalize_ticker(ticker)
        settings = get_settings()
        lookback_days = days if days is not None else settings.finnhub_news_lookback_days
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)
        response = self._get(
            "/company-news",
            {
                "symbol": ticker_symbol,
                "from": start_date.isoformat(),
                "to": end_date.isoformat(),
            },
        )
        articles = [_normalize_news_item(item) for item in response] if isinstance(response, list) else []

        return {
            "ticker": ticker_symbol,
            "from": start_date.isoformat(),
            "to": end_date.isoformat(),
            "articles": articles,
            "retrieved_at": _now(),
        }

    @traced("finnhub.get_quote", run_type="tool")
    def get_quote(self, ticker: str) -> dict[str, Any]:
        ticker_symbol = _normalize_ticker(ticker)
        data = self._get("/quote", {"symbol": ticker_symbol})
        quote = data if isinstance(data, dict) else {}

        return {
            "ticker": ticker_symbol,
            "current_price": quote.get("c"),
            "day_change": quote.get("d"),
            "percent_change": quote.get("dp"),
            "day_high": quote.get("h"),
            "day_low": quote.get("l"),
            "open": quote.get("o"),
            "previous_close": quote.get("pc"),
            "market_timestamp": quote.get("t"),
            "retrieved_at": _now(),
        }

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        reraise=True,
    )
    @traced("finnhub.get", run_type="tool")
    def _get(self, path: str, params: dict[str, Any]) -> Any:
        if not self.api_key:
            raise ValueError("FINNHUB_API_KEY is required for Finnhub requests.")

        request_params = {**params, "token": self.api_key}
        with httpx.Client(base_url=FINNHUB_BASE_URL, timeout=self.timeout_seconds) as client:
            response = client.get(path, params=request_params)
            response.raise_for_status()
            return response.json()


def _normalize_ticker(ticker: str) -> str:
    value = ticker.strip().upper()
    if not value:
        raise ValueError("Ticker is required.")
    return value


def _recommendation_counts(latest: dict[str, Any] | None) -> dict[str, int] | None:
    if not latest:
        return None
    counts = {
        "strong_buy": int(latest.get("strongBuy") or 0),
        "buy": int(latest.get("buy") or 0),
        "hold": int(latest.get("hold") or 0),
        "sell": int(latest.get("sell") or 0),
        "strong_sell": int(latest.get("strongSell") or 0),
    }
    if sum(counts.values()) == 0:
        return None
    return counts


def _normalize_news_item(item: dict[str, Any]) -> dict[str, Any]:
    published_at = item.get("datetime")
    if published_at:
        try:
            published_at = datetime.fromtimestamp(int(published_at), UTC).isoformat()
        except (TypeError, ValueError, OSError):
            published_at = str(published_at)

    return {
        "title": item.get("headline") or "Untitled source",
        "url": item.get("url") or "",
        "published_at": published_at,
        "snippet": item.get("summary"),
        "source": item.get("source"),
    }


def _now() -> str:
    return datetime.now(UTC).isoformat()
