"""Subquery builder node."""

from collections.abc import Callable

from app.config import get_settings
from app.constants import (
    CATEGORY_ANALYST_RATINGS,
    CATEGORY_COMPANY_INFORMATION,
    CATEGORY_COMPANY_NEWS,
    CATEGORY_COMPETITOR_NEWS,
    CATEGORY_CURRENT_STOCK_PRICE,
    CATEGORY_DOW_JONES_INDEX,
    CATEGORY_FINANCIALS,
    CATEGORY_LATEST_REPORT_RELEASE,
    CATEGORY_SECTOR_NEWS,
    CATEGORY_TECHNICAL_ANALYSIS,
)
from app.models.state import FinancialResearchState
from app.models.types import CategoryName
from app.utils.tracing import traced


@traced("build_subqueries_node")
def build_subqueries_node(state: FinancialResearchState) -> dict[str, dict[CategoryName, str]]:
    """Create concise provider-friendly subqueries for selected categories."""

    settings = get_settings()
    context = _query_context(state)
    categories = state.get("categories", [])
    subqueries: dict[CategoryName, str] = {}

    for category in categories:
        builder = SUBQUERY_BUILDERS.get(category)
        if builder:
            subqueries[category] = builder(
                context,
                settings.default_lookback_days,
                settings.technical_analysis_lookback_days,
            )

    return {"subqueries": subqueries}


def _query_context(state: FinancialResearchState) -> str:
    tickers = state.get("detected_tickers", [])
    companies = state.get("detected_companies", [])
    context_parts = [*tickers, *companies]
    return " ".join(_dedupe_context_parts(context_parts)).strip()


def _dedupe_context_parts(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()

    for value in values:
        normalized = value.strip()
        key = normalized.lower()
        if normalized and key not in seen:
            deduped.append(normalized)
            seen.add(key)

    return deduped


def _with_context(context: str, fallback: str) -> str:
    return context or fallback


def _company_information_query(context: str, _: int, __: int) -> str:
    return f"{_with_context(context, 'company')} company profile sector industry market cap"


def _competitor_news_query(context: str, lookback_days: int, _: int) -> str:
    return f"{_with_context(context, 'company')} competitors news last {lookback_days} days"


def _sector_news_query(context: str, lookback_days: int, _: int) -> str:
    return f"{_with_context(context, 'market sector')} sector news trends last {lookback_days} days"


def _financials_query(context: str, _: int, __: int) -> str:
    return f"{_with_context(context, 'company')} latest quarterly financials revenue net income cash flow"


def _company_news_query(context: str, lookback_days: int, _: int) -> str:
    return f"{_with_context(context, 'company')} latest company news last {lookback_days} days"


def _current_stock_price_query(context: str, _: int, __: int) -> str:
    return f"{_with_context(context, 'stock')} current stock quote latest price"


def _analyst_ratings_query(context: str, _: int, __: int) -> str:
    return f"{_with_context(context, 'stock')} analyst ratings recommendation trends price target"


def _latest_report_release_query(context: str, _: int, __: int) -> str:
    return f"{_with_context(context, 'company')} latest SEC report 10-K 10-Q 8-K filing"


def _dow_jones_index_query(context: str, lookback_days: int, _: int) -> str:
    if context:
        return f"{context} latest index level daily change news last {lookback_days} days"
    return f"Dow Jones latest index level daily change news last {lookback_days} days"


def _technical_analysis_query(context: str, _: int, technical_lookback_days: int) -> str:
    return f"{_with_context(context, 'stock')} {technical_lookback_days} day historical prices SMA RSI MACD"


SubqueryBuilder = Callable[[str, int, int], str]

SUBQUERY_BUILDERS: dict[CategoryName, SubqueryBuilder] = {
    CATEGORY_COMPANY_INFORMATION: _company_information_query,
    CATEGORY_COMPETITOR_NEWS: _competitor_news_query,
    CATEGORY_SECTOR_NEWS: _sector_news_query,
    CATEGORY_FINANCIALS: _financials_query,
    CATEGORY_COMPANY_NEWS: _company_news_query,
    CATEGORY_CURRENT_STOCK_PRICE: _current_stock_price_query,
    CATEGORY_ANALYST_RATINGS: _analyst_ratings_query,
    CATEGORY_LATEST_REPORT_RELEASE: _latest_report_release_query,
    CATEGORY_DOW_JONES_INDEX: _dow_jones_index_query,
    CATEGORY_TECHNICAL_ANALYSIS: _technical_analysis_query,
}
