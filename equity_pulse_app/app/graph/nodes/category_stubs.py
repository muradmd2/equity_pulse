"""Category nodes and remaining fake stubs."""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

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
    STATUS_FAILED,
    STATUS_PARTIAL,
    STATUS_SUCCESS,
)
from app.models.citations import SourceCitation
from app.models.retrieval import RetrievalResult, RetrievalResultModel
from app.models.state import FinancialResearchState
from app.models.types import CategoryName
from app.services.finnhub_service import FinnhubService
from app.services.sec_edgar_service import SecEdgarService
from app.services.tavily_service import TavilyService
from app.services.yfinance_service import YFinanceService
from app.utils.technical_indicators import calculate_technical_indicators
from app.utils.tracing import traced

TECHNICAL_ANALYSIS_DISCLAIMER = "Technical indicators are descriptive only and are not investment advice."


@traced("company_info_node", run_type="tool")
def company_info_node(state: FinancialResearchState) -> dict[str, list[RetrievalResult] | list[SourceCitation] | list[str]]:
    return _yfinance_category_result(
        state=state,
        category=CATEGORY_COMPANY_INFORMATION,
        title="Company profile",
        fetch=lambda service, ticker: service.get_company_info(ticker),
        summary_builder=_company_info_summary,
    )


@traced("competitor_news_node", run_type="tool")
def competitor_news_node(state: FinancialResearchState) -> dict[str, list[RetrievalResult] | list[SourceCitation] | list[str]]:
    return _tavily_news_result(
        state=state,
        category=CATEGORY_COMPETITOR_NEWS,
        title="Competitor news",
        fallback_query=f"{_primary_company_or_ticker(state)} competitors news",
    )


@traced("sector_news_node", run_type="tool")
def sector_news_node(state: FinancialResearchState) -> dict[str, list[RetrievalResult] | list[SourceCitation] | list[str]]:
    return _tavily_news_result(
        state=state,
        category=CATEGORY_SECTOR_NEWS,
        title="Sector news",
        fallback_query=f"{_primary_company_or_ticker(state)} sector news",
    )


@traced("financials_node", run_type="tool")
def financials_node(state: FinancialResearchState) -> dict[str, list[RetrievalResult] | list[SourceCitation] | list[str]]:
    return _financials_result(state)


@traced("company_news_node", run_type="tool")
def company_news_node(state: FinancialResearchState) -> dict[str, list[RetrievalResult] | list[SourceCitation] | list[str]]:
    return _tavily_news_result(
        state=state,
        category=CATEGORY_COMPANY_NEWS,
        title="Company news",
        fallback_query=f"{_primary_company_or_ticker(state)} latest company news",
    )


@traced("current_stock_price_node", run_type="tool")
def current_stock_price_node(state: FinancialResearchState) -> dict[str, list[RetrievalResult] | list[SourceCitation] | list[str]]:
    return _yfinance_category_result(
        state=state,
        category=CATEGORY_CURRENT_STOCK_PRICE,
        title="Quote data",
        fetch=lambda service, ticker: service.get_quote(ticker),
        summary_builder=_quote_summary,
    )


@traced("analyst_ratings_node", run_type="tool")
def analyst_ratings_node(state: FinancialResearchState) -> dict[str, list[RetrievalResult] | list[SourceCitation] | list[str]]:
    return _analyst_ratings_result(state)


@traced("latest_report_release_node", run_type="tool")
def latest_report_release_node(state: FinancialResearchState) -> dict[str, list[RetrievalResult] | list[SourceCitation] | list[str]]:
    return _latest_report_release_result(state)


@traced("dow_jones_index_node", run_type="tool")
def dow_jones_index_node(state: FinancialResearchState) -> dict[str, list[RetrievalResult] | list[SourceCitation] | list[str]]:
    return _yfinance_category_result(
        state=state,
        category=CATEGORY_DOW_JONES_INDEX,
        title="Dow Jones quote data",
        fetch=lambda service, _: service.get_dow_jones_quote(),
        summary_builder=_dow_jones_summary,
        fallback_ticker="^DJI",
    )


@traced("technical_analysis_node", run_type="tool")
def technical_analysis_node(state: FinancialResearchState) -> dict[str, list[RetrievalResult] | list[SourceCitation] | list[str]]:
    return _technical_analysis_result(state)


CategoryNode = Callable[[FinancialResearchState], dict[str, list[RetrievalResult] | list[SourceCitation] | list[str]]]

CATEGORY_NODE_NAMES: dict[CategoryName, str] = {
    CATEGORY_COMPANY_INFORMATION: "company_info",
    CATEGORY_COMPETITOR_NEWS: "competitor_news",
    CATEGORY_SECTOR_NEWS: "sector_news",
    CATEGORY_FINANCIALS: "financials",
    CATEGORY_COMPANY_NEWS: "company_news",
    CATEGORY_CURRENT_STOCK_PRICE: "current_stock_price",
    CATEGORY_ANALYST_RATINGS: "analyst_ratings",
    CATEGORY_LATEST_REPORT_RELEASE: "latest_report_release",
    CATEGORY_DOW_JONES_INDEX: "dow_jones_index",
    CATEGORY_TECHNICAL_ANALYSIS: "technical_analysis",
}

CATEGORY_NODES: dict[str, CategoryNode] = {
    "company_info": company_info_node,
    "competitor_news": competitor_news_node,
    "sector_news": sector_news_node,
    "financials": financials_node,
    "company_news": company_news_node,
    "current_stock_price": current_stock_price_node,
    "analyst_ratings": analyst_ratings_node,
    "latest_report_release": latest_report_release_node,
    "dow_jones_index": dow_jones_index_node,
    "technical_analysis": technical_analysis_node,
}


def _stub_category_result(
    state: FinancialResearchState,
    category: CategoryName,
    title: str,
    data: dict[str, Any],
    summary: str,
) -> dict[str, list[RetrievalResult] | list[SourceCitation] | list[str]]:
    citation = _mock_citation(category=category, title=title)
    result = RetrievalResultModel(
        category=category,
        subquery=state.get("subqueries", {}).get(category, ""),
        status=STATUS_SUCCESS,
        data=data,
        summary=summary,
        citations=[citation],
        errors=[],
    ).model_dump()

    return {
        "retrieval_results": [result],
        "citations": [citation],
        "executed_nodes": [CATEGORY_NODE_NAMES[category]],
    }


def _yfinance_category_result(
    state: FinancialResearchState,
    category: CategoryName,
    title: str,
    fetch: Callable[[YFinanceService, str], dict[str, Any]],
    summary_builder: Callable[[dict[str, Any]], str],
    fallback_ticker: str | None = None,
) -> dict[str, list[RetrievalResult] | list[SourceCitation] | list[str]]:
    ticker = _primary_ticker(state, fallback=fallback_ticker)
    warnings: list[str] = []
    errors: list[str] = []

    try:
        data = fetch(YFinanceService(), ticker)
        warnings.extend(_missing_field_warnings(category, data))
        if category in {CATEGORY_CURRENT_STOCK_PRICE, CATEGORY_DOW_JONES_INDEX}:
            warnings.append("yfinance market data may be delayed.")

        status = STATUS_PARTIAL if warnings else STATUS_SUCCESS
        summary = summary_builder(data)
    except Exception as exc:
        data = {}
        status = STATUS_FAILED
        errors.append(str(exc))
        summary = f"{category} could not be retrieved from yfinance."

    citation = _yfinance_citation(category=category, title=title, ticker=ticker)
    result = RetrievalResultModel(
        category=category,
        subquery=state.get("subqueries", {}).get(category, ""),
        status=status,
        data=data,
        summary=summary,
        citations=[citation],
        errors=errors,
    ).model_dump()

    update: dict[str, list[RetrievalResult] | list[SourceCitation] | list[str]] = {
        "retrieval_results": [result],
        "citations": [citation],
        "executed_nodes": [CATEGORY_NODE_NAMES[category]],
    }
    if warnings:
        update["warnings"] = warnings
    if errors:
        update["errors"] = errors
    return update


def _tavily_news_result(
    state: FinancialResearchState,
    category: CategoryName,
    title: str,
    fallback_query: str,
) -> dict[str, list[RetrievalResult] | list[SourceCitation] | list[str]]:
    query = state.get("subqueries", {}).get(category) or fallback_query
    warnings: list[str] = []
    errors: list[str] = []

    try:
        data = TavilyService().search(query=query)
        articles = data.get("results", [])
        if articles:
            status = STATUS_SUCCESS
            summary = _news_summary(category, articles)
        else:
            status = STATUS_PARTIAL
            warnings.append(f"Tavily returned no results for {category}.")
            summary = f"No Tavily articles were returned for {category}."
    except Exception as exc:
        data = {"query": query, "results": []}
        status = STATUS_FAILED
        errors.append(str(exc))
        summary = f"{category} could not be retrieved from Tavily."

    citations = [_tavily_citation(category, index, article) for index, article in enumerate(data.get("results", []), start=1)]
    if not citations:
        citations = [_tavily_empty_citation(category, query)]

    result = RetrievalResultModel(
        category=category,
        subquery=query,
        status=status,
        data=data,
        summary=summary,
        citations=citations,
        errors=errors,
    ).model_dump()

    update: dict[str, list[RetrievalResult] | list[SourceCitation] | list[str]] = {
        "retrieval_results": [result],
        "citations": citations,
        "executed_nodes": [CATEGORY_NODE_NAMES[category]],
    }
    if warnings:
        update["warnings"] = warnings
    if errors:
        update["errors"] = errors
    return update


def _latest_report_release_result(state: FinancialResearchState) -> dict[str, list[RetrievalResult] | list[SourceCitation] | list[str]]:
    ticker = _primary_ticker(state)
    errors: list[str] = []

    try:
        data = SecEdgarService().get_latest_filings(ticker)
        filings = data.get("filings", {})
        available_filings = [filing for filing in filings.values() if filing]
        if not available_filings:
            status = STATUS_FAILED
            errors.append(f"No SEC 10-K, 10-Q, or 8-K filings found for ticker '{ticker}'.")
            summary = f"SEC filing metadata could not be found for {ticker}."
        else:
            status = STATUS_SUCCESS
            summary = _latest_report_summary(data)
    except Exception as exc:
        data = {"ticker": ticker, "filings": {}}
        status = STATUS_FAILED
        errors.append(str(exc))
        summary = f"SEC filing metadata could not be retrieved for {ticker}."

    citations = _sec_filing_citations(CATEGORY_LATEST_REPORT_RELEASE, ticker, data.get("filings", {}))
    if not citations:
        citations = [_sec_empty_citation(CATEGORY_LATEST_REPORT_RELEASE, ticker)]

    result = RetrievalResultModel(
        category=CATEGORY_LATEST_REPORT_RELEASE,
        subquery=state.get("subqueries", {}).get(CATEGORY_LATEST_REPORT_RELEASE, ""),
        status=status,
        data=data,
        summary=summary,
        citations=citations,
        errors=errors,
    ).model_dump()

    update: dict[str, list[RetrievalResult] | list[SourceCitation] | list[str]] = {
        "retrieval_results": [result],
        "citations": result["citations"],
        "executed_nodes": [CATEGORY_NODE_NAMES[CATEGORY_LATEST_REPORT_RELEASE]],
    }
    if errors:
        update["errors"] = errors
    return update


def _financials_result(state: FinancialResearchState) -> dict[str, list[RetrievalResult] | list[SourceCitation] | list[str]]:
    ticker = _primary_ticker(state)
    warnings: list[str] = []
    errors: list[str] = []
    citations: list[SourceCitation] = []

    try:
        data = YFinanceService().get_basic_financials(ticker)
        warnings.extend(_missing_field_warnings(CATEGORY_FINANCIALS, data))
        status = STATUS_PARTIAL if warnings else STATUS_SUCCESS
        summary = _financials_summary(data)
        citations.append(_yfinance_citation(CATEGORY_FINANCIALS, "Basic financials", ticker))
    except Exception as exc:
        data = {"ticker": ticker}
        status = STATUS_FAILED
        errors.append(str(exc))
        summary = f"{CATEGORY_FINANCIALS} could not be retrieved from yfinance."

    try:
        sec_data = SecEdgarService().get_latest_filings(ticker)
        data["sec_filings"] = sec_data.get("filings", {})
        citations.extend(_sec_filing_citations(CATEGORY_FINANCIALS, ticker, sec_data.get("filings", {})))
        summary = f"{summary} Official SEC filing references were retrieved."
        if status == STATUS_FAILED:
            status = STATUS_PARTIAL
    except Exception as exc:
        warnings.append(f"SEC filing references unavailable: {exc}")
        if status == STATUS_SUCCESS:
            status = STATUS_PARTIAL

    result = RetrievalResultModel(
        category=CATEGORY_FINANCIALS,
        subquery=state.get("subqueries", {}).get(CATEGORY_FINANCIALS, ""),
        status=status,
        data=data,
        summary=summary,
        citations=citations or [_yfinance_citation(CATEGORY_FINANCIALS, "Basic financials", ticker)],
        errors=errors,
    ).model_dump()

    update: dict[str, list[RetrievalResult] | list[SourceCitation] | list[str]] = {
        "retrieval_results": [result],
        "citations": result["citations"],
        "executed_nodes": [CATEGORY_NODE_NAMES[CATEGORY_FINANCIALS]],
    }
    if warnings:
        update["warnings"] = warnings
    if errors:
        update["errors"] = errors
    return update


def _analyst_ratings_result(state: FinancialResearchState) -> dict[str, list[RetrievalResult] | list[SourceCitation] | list[str]]:
    ticker = _primary_ticker(state)
    query = state.get("subqueries", {}).get(CATEGORY_ANALYST_RATINGS) or f"{ticker} analyst ratings price target"
    warnings: list[str] = []
    errors: list[str] = []
    citations: list[SourceCitation] = []

    recommendations = _safe_finnhub_call(
        lambda service: service.get_recommendation_trends(ticker),
        "Finnhub recommendation trends unavailable",
        warnings,
        errors,
    )
    price_target = _safe_finnhub_call(
        lambda service: service.get_price_target(ticker),
        "Finnhub price target unavailable",
        warnings,
        errors,
    )
    quote = _safe_finnhub_call(
        lambda service: service.get_quote(ticker),
        "Finnhub quote unavailable for target comparison",
        warnings,
        errors,
    )
    commentary, commentary_citations = _analyst_commentary(query)
    citations.extend(commentary_citations)

    if recommendations:
        citations.append(_finnhub_citation(CATEGORY_ANALYST_RATINGS, "Recommendation trends", ticker))
    if price_target:
        citations.append(_finnhub_citation(CATEGORY_ANALYST_RATINGS, "Price target", ticker))
    if quote:
        citations.append(_finnhub_citation(CATEGORY_ANALYST_RATINGS, "Quote fallback", ticker))

    counts = recommendations.get("counts") if recommendations else None
    consensus_rating = _derive_consensus_rating(counts)
    current_price = quote.get("current_price") if quote else None
    target_mean = price_target.get("target_mean") if price_target else None
    upside_downside_percent = _upside_downside_percent(target_mean, current_price)

    data = {
        "ticker": ticker,
        "buy_hold_sell_counts": counts,
        "consensus_rating": consensus_rating,
        "price_target": price_target or {},
        "current_price": current_price,
        "upside_downside_percent": upside_downside_percent,
        "recent_analyst_commentary": commentary,
    }

    has_core_data = bool(counts or target_mean is not None)
    if not has_core_data and not commentary:
        status = STATUS_FAILED
        summary = f"{ticker} analyst ratings could not be retrieved from Finnhub or Tavily."
    elif warnings or errors:
        status = STATUS_PARTIAL
        summary = _analyst_summary(ticker, data, detailed=_query_asks_for_details(state))
    else:
        status = STATUS_SUCCESS
        summary = _analyst_summary(ticker, data, detailed=_query_asks_for_details(state))

    result = RetrievalResultModel(
        category=CATEGORY_ANALYST_RATINGS,
        subquery=query,
        status=status,
        data=data,
        summary=summary,
        citations=citations or [_finnhub_empty_citation(CATEGORY_ANALYST_RATINGS, ticker)],
        errors=errors,
    ).model_dump()

    update: dict[str, list[RetrievalResult] | list[SourceCitation] | list[str]] = {
        "retrieval_results": [result],
        "citations": result["citations"],
        "executed_nodes": [CATEGORY_NODE_NAMES[CATEGORY_ANALYST_RATINGS]],
    }
    if warnings:
        update["warnings"] = warnings
    if errors:
        update["errors"] = errors
    return update


def _technical_analysis_result(state: FinancialResearchState) -> dict[str, list[RetrievalResult] | list[SourceCitation] | list[str]]:
    ticker = _primary_ticker(state)
    warnings: list[str] = []
    errors: list[str] = []

    try:
        historical_data = YFinanceService().get_historical_prices(ticker, period="1y")
        prices = _close_prices_from_history(historical_data)
        indicators = calculate_technical_indicators(prices)
        warnings.extend(_technical_indicator_warnings(indicators))
        status = STATUS_PARTIAL if warnings else STATUS_SUCCESS
        data = {**historical_data, "indicators": indicators}
        summary = _technical_analysis_summary(ticker, indicators)
    except Exception as exc:
        data = {"ticker": ticker, "indicators": {}}
        status = STATUS_FAILED
        errors.append(str(exc))
        summary = f"Technical analysis data could not be retrieved for {ticker}. {TECHNICAL_ANALYSIS_DISCLAIMER}"

    citation = _yfinance_citation(CATEGORY_TECHNICAL_ANALYSIS, "Historical prices", ticker)
    result = RetrievalResultModel(
        category=CATEGORY_TECHNICAL_ANALYSIS,
        subquery=state.get("subqueries", {}).get(CATEGORY_TECHNICAL_ANALYSIS, ""),
        status=status,
        data=data,
        summary=summary,
        citations=[citation],
        errors=errors,
    ).model_dump()

    update: dict[str, list[RetrievalResult] | list[SourceCitation] | list[str]] = {
        "retrieval_results": [result],
        "citations": [citation],
        "executed_nodes": [CATEGORY_NODE_NAMES[CATEGORY_TECHNICAL_ANALYSIS]],
    }
    if warnings:
        update["warnings"] = warnings
    if errors:
        update["errors"] = errors
    return update


def _safe_finnhub_call(
    call: Callable[[FinnhubService], dict[str, Any]],
    warning_message: str,
    warnings: list[str],
    errors: list[str],
) -> dict[str, Any]:
    try:
        data = call(FinnhubService())
    except Exception as exc:
        errors.append(str(exc))
        warnings.append(warning_message)
        return {}
    if _is_empty_finnhub_payload(data):
        warnings.append(warning_message)
        return {}
    return data


def _is_empty_finnhub_payload(data: dict[str, Any]) -> bool:
    meaningful_values = [
        value
        for key, value in data.items()
        if key not in {"ticker", "retrieved_at"} and value not in (None, {}, [], "")
    ]
    return not meaningful_values


def _analyst_commentary(query: str) -> tuple[list[dict[str, Any]], list[SourceCitation]]:
    try:
        data = TavilyService().search(query=query, max_results=3)
    except Exception:
        return [], []

    articles = data.get("results", [])[:3]
    citations = [
        _tavily_citation(CATEGORY_ANALYST_RATINGS, index, article)
        for index, article in enumerate(articles, start=1)
    ]
    return articles, citations


def _mock_citation(category: CategoryName, title: str) -> SourceCitation:
    return {
        "source_id": f"mock-{category}",
        "title": title,
        "url": "",
        "provider": "mock",
        "published_at": None,
        "retrieved_at": _retrieved_at(),
        "snippet": "Fake data for local graph wiring only.",
    }


def _yfinance_citation(category: CategoryName, title: str, ticker: str) -> SourceCitation:
    return {
        "source_id": f"yfinance-{category}-{ticker}",
        "title": f"yfinance {title} for {ticker}",
        "url": f"https://finance.yahoo.com/quote/{ticker}",
        "provider": "yfinance",
        "published_at": None,
        "retrieved_at": _retrieved_at(),
        "snippet": "Data retrieved through the yfinance Python package.",
    }


def _tavily_citation(category: CategoryName, index: int, article: dict[str, Any]) -> SourceCitation:
    return {
        "source_id": f"tavily-{category}-{index}",
        "title": article.get("title") or "Untitled source",
        "url": article.get("url") or "",
        "provider": "tavily",
        "published_at": article.get("published_at"),
        "retrieved_at": article.get("retrieved_at") or _retrieved_at(),
        "snippet": article.get("snippet"),
    }


def _tavily_empty_citation(category: CategoryName, query: str) -> SourceCitation:
    return {
        "source_id": f"tavily-{category}-empty",
        "title": f"Tavily search for {category}",
        "url": "",
        "provider": "tavily",
        "published_at": None,
        "retrieved_at": _retrieved_at(),
        "snippet": f"No Tavily results returned for query: {query}",
    }


def _finnhub_citation(category: CategoryName, title: str, ticker: str) -> SourceCitation:
    return {
        "source_id": f"finnhub-{category}-{title.lower().replace(' ', '-')}-{ticker}",
        "title": f"Finnhub {title} for {ticker}",
        "url": "https://finnhub.io/",
        "provider": "finnhub",
        "published_at": None,
        "retrieved_at": _retrieved_at(),
        "snippet": "Data retrieved through Finnhub API.",
    }


def _finnhub_empty_citation(category: CategoryName, ticker: str) -> SourceCitation:
    return {
        "source_id": f"finnhub-{category}-empty-{ticker}",
        "title": f"Finnhub analyst ratings for {ticker}",
        "url": "https://finnhub.io/",
        "provider": "finnhub",
        "published_at": None,
        "retrieved_at": _retrieved_at(),
        "snippet": "No Finnhub analyst ratings data was available.",
    }


def _sec_filing_citations(category: CategoryName, ticker: str, filings: dict[str, Any]) -> list[SourceCitation]:
    citations: list[SourceCitation] = []
    for key, filing in filings.items():
        if not filing:
            continue
        form_type = filing.get("form_type") or key.replace("latest_", "").upper()
        citations.append(
            {
                "source_id": f"sec-{category}-{ticker}-{form_type}",
                "title": f"SEC EDGAR {form_type} for {ticker}",
                "url": filing.get("filing_url") or "",
                "provider": "SEC EDGAR",
                "published_at": filing.get("filing_date"),
                "retrieved_at": _retrieved_at(),
                "snippet": (
                    f"{form_type} filed {filing.get('filing_date')}; "
                    f"accession {filing.get('accession_number')}; "
                    f"report period {filing.get('report_period')}."
                ),
            }
        )
    return citations


def _sec_empty_citation(category: CategoryName, ticker: str) -> SourceCitation:
    return {
        "source_id": f"sec-{category}-empty-{ticker}",
        "title": f"SEC EDGAR filing lookup for {ticker}",
        "url": "https://www.sec.gov/edgar/search/",
        "provider": "SEC EDGAR",
        "published_at": None,
        "retrieved_at": _retrieved_at(),
        "snippet": "No SEC filing metadata was available.",
    }


def _news_summary(category: CategoryName, articles: list[dict[str, Any]]) -> str:
    bullets = []
    for article in articles[:3]:
        title = article.get("title") or "Untitled source"
        published_at = article.get("published_at")
        date_text = f" ({published_at})" if published_at else ""
        bullets.append(f"- {title}{date_text}")
    return f"{category} Tavily results:\n" + "\n".join(bullets)


def _latest_report_summary(data: dict[str, Any]) -> str:
    ticker = data.get("ticker", "ticker")
    filings = data.get("filings", {})
    parts = []
    for label, filing in (
        ("latest 10-K", filings.get("latest_10k")),
        ("latest 10-Q", filings.get("latest_10q")),
        ("latest 8-K", filings.get("latest_8k")),
    ):
        if filing:
            parts.append(
                f"{label}: filed {filing.get('filing_date')}, "
                f"report period {filing.get('report_period')}, "
                f"accession {filing.get('accession_number')}."
            )
    return f"{ticker} SEC EDGAR filing metadata retrieved. " + " ".join(parts)


def _derive_consensus_rating(counts: dict[str, int] | None) -> str | None:
    if not counts:
        return None

    positive = counts.get("strong_buy", 0) + counts.get("buy", 0)
    neutral = counts.get("hold", 0)
    negative = counts.get("sell", 0) + counts.get("strong_sell", 0)

    if positive == neutral == negative == 0:
        return None
    if positive > neutral and positive > negative:
        return "buy"
    if negative > positive and negative > neutral:
        return "sell"
    if neutral >= positive and neutral >= negative:
        return "hold"
    return "mixed"


def _upside_downside_percent(target_mean: Any, current_price: Any) -> float | None:
    if target_mean in (None, 0) or current_price in (None, 0):
        return None
    try:
        return round(((float(target_mean) - float(current_price)) / float(current_price)) * 100, 2)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _analyst_summary(ticker: str, data: dict[str, Any], detailed: bool) -> str:
    consensus = data.get("consensus_rating")
    counts = data.get("buy_hold_sell_counts")
    price_target = data.get("price_target") or {}
    target_mean = price_target.get("target_mean")
    upside_downside = data.get("upside_downside_percent")

    parts = []
    if consensus:
        parts.append(f"{ticker} analyst consensus from available Finnhub counts is {consensus}.")
    elif counts:
        parts.append(f"{ticker} has available Finnhub analyst counts, but no clear consensus.")
    else:
        parts.append(f"{ticker} consensus rating is unavailable from current Finnhub data.")

    if target_mean is not None:
        target_text = f"Mean price target is {target_mean}."
        if upside_downside is not None:
            target_text = f"{target_text} Implied move versus current price is {upside_downside}%."
        parts.append(target_text)

    if detailed and data.get("recent_analyst_commentary"):
        commentary_titles = [
            article.get("title", "Untitled source")
            for article in data["recent_analyst_commentary"][:3]
        ]
        parts.append("Recent commentary: " + "; ".join(commentary_titles) + ".")

    return " ".join(parts)


def _query_asks_for_details(state: FinancialResearchState) -> bool:
    query = (state.get("active_query") or state.get("original_query") or "").lower()
    detail_terms = ("detail", "details", "why", "explain", "commentary", "price target", "breakdown")
    return any(term in query for term in detail_terms)


def _missing_field_warnings(category: CategoryName, data: dict[str, Any]) -> list[str]:
    required_fields = {
        CATEGORY_CURRENT_STOCK_PRICE: ("current_price", "previous_close", "market_timestamp"),
        CATEGORY_COMPANY_INFORMATION: ("company_name", "sector", "industry"),
        CATEGORY_FINANCIALS: ("revenue", "net_income"),
        CATEGORY_DOW_JONES_INDEX: ("current_price", "previous_close", "market_timestamp"),
        CATEGORY_TECHNICAL_ANALYSIS: ("row_count", "latest_close"),
    }.get(category, ())

    return [f"yfinance missing field for {category}: {field}" for field in required_fields if data.get(field) is None]


def _quote_summary(data: dict[str, Any]) -> str:
    ticker = data.get("ticker", "Ticker")
    price = data.get("current_price")
    percent_change = data.get("percent_change")
    if price is None:
        return f"{ticker} quote data was partially retrieved, but current price is unavailable."
    if percent_change is None:
        return f"{ticker} latest yfinance price is {price}."
    return f"{ticker} latest yfinance price is {price}, with a {percent_change}% day change."


def _company_info_summary(data: dict[str, Any]) -> str:
    name = data.get("company_name") or data.get("ticker") or "Company"
    sector = data.get("sector")
    industry = data.get("industry")
    if sector and industry:
        return f"{name} is listed by yfinance in the {sector} sector and {industry} industry."
    return f"{name} company information was partially retrieved from yfinance."


def _financials_summary(data: dict[str, Any]) -> str:
    ticker = data.get("ticker", "Ticker")
    revenue = data.get("revenue")
    net_income = data.get("net_income")
    if revenue is None and net_income is None:
        return f"{ticker} basic financials were partially retrieved, but key fields are unavailable."
    return f"{ticker} yfinance financials include revenue {revenue} and net income {net_income}."


def _dow_jones_summary(data: dict[str, Any]) -> str:
    price = data.get("current_price")
    percent_change = data.get("percent_change")
    return f"Dow Jones yfinance level is {price}, with a {percent_change}% day change."


def _historical_prices_summary(data: dict[str, Any]) -> str:
    ticker = data.get("ticker", "Ticker")
    row_count = data.get("row_count", 0)
    latest_close = data.get("latest_close")
    return f"{ticker} yfinance historical price retrieval returned {row_count} rows; latest close is {latest_close}."


def _close_prices_from_history(historical_data: dict[str, Any]) -> list[float]:
    prices = historical_data.get("prices", [])
    closes: list[float] = []
    for item in prices:
        close = item.get("close") if isinstance(item, dict) else None
        if close is not None:
            closes.append(float(close))
    return closes


def _technical_indicator_warnings(indicators: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    data_points = indicators.get("data_points", 0)
    if data_points < 50:
        warnings.append("Insufficient history for 50-day and 200-day moving averages.")
    elif data_points < 200:
        warnings.append("Insufficient history for 200-day moving average.")
    if indicators.get("rsi_14") is None:
        warnings.append("Insufficient history for RSI 14.")
    if indicators.get("macd", {}).get("macd") is None:
        warnings.append("Insufficient history for MACD.")
    return warnings


def _technical_analysis_summary(ticker: str, indicators: dict[str, Any]) -> str:
    latest_close = indicators.get("latest_close")
    trend = indicators.get("trend")
    macd_data = indicators.get("macd", {})
    parts = [
        f"{ticker} technical indicators describe a {trend} setup based on available historical prices.",
        f"Latest close is {latest_close}.",
    ]

    if indicators.get("sma_20") is not None:
        parts.append(f"SMA 20 is {indicators['sma_20']}.")
    if indicators.get("sma_50") is not None:
        parts.append(f"SMA 50 is {indicators['sma_50']}.")
    if indicators.get("sma_200") is not None:
        parts.append(f"SMA 200 is {indicators['sma_200']}.")
    if indicators.get("rsi_14") is not None:
        parts.append(f"RSI 14 is {indicators['rsi_14']}.")
    if macd_data.get("macd") is not None:
        parts.append(f"MACD is {macd_data['macd']} with signal {macd_data['signal']}.")
    if indicators.get("volatility") is not None:
        parts.append(f"Annualized volatility is {indicators['volatility']}.")
    if indicators.get("support") is not None and indicators.get("resistance") is not None:
        parts.append(f"Approximate recent support is {indicators['support']} and resistance is {indicators['resistance']}.")

    parts.append(TECHNICAL_ANALYSIS_DISCLAIMER)
    return " ".join(parts)


def _retrieved_at() -> str:
    return datetime.now(UTC).isoformat()


def _primary_ticker(state: FinancialResearchState, fallback: str | None = None) -> str:
    return next(iter(state.get("detected_tickers", [])), fallback or "MOCK")


def _primary_company_or_ticker(state: FinancialResearchState) -> str:
    return next(iter(state.get("detected_companies", [])), _primary_ticker(state))
