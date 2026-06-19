"""Source-grounded summary node."""

import json
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langgraph.config import get_stream_writer

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
from app.models.citations import SourceCitation
from app.models.retrieval import RetrievalResult
from app.models.state import FinancialResearchState
from app.prompts.llm_prompts import SUMMARY_PROMPT
from app.utils.runtime_secrets import get_request_openai_api_key
from app.utils.tracing import traced

EDUCATIONAL_DISCLAIMER = "This is for educational research only and is not personalized financial advice."


@traced("summary_node")
def summary_node(state: FinancialResearchState) -> dict[str, object]:
    """Create a concise final-answer draft using only retrieved results."""

    summary, streamed = summarize_with_openai(state)
    if summary is None:
        summary = _deterministic_summary(state)

    if state.get("iteration", 1) >= 2 or (state.get("review_attempted") and state.get("first_summary")):
        return {"second_summary": summary, "summary_streamed": streamed}
    return {"first_summary": summary, "summary_streamed": streamed}


def summarize_with_openai(state: FinancialResearchState) -> tuple[str | None, bool]:
    """Ask OpenAI to summarize retrieved data when configured."""

    settings = get_settings()
    api_key = get_request_openai_api_key(state.get("session_id")) or settings.openai_api_key
    if not api_key or not state.get("original_query") or not state.get("categories"):
        return None, False

    model_name = state.get("model_name") or settings.openai_model
    token_callback = _SummaryStreamCallback()
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=api_key,
        streaming=True,
        callbacks=[token_callback],
    )
    try:
        response = llm.invoke(SUMMARY_PROMPT.format(context=json.dumps(_summary_context(state), default=str, indent=2)))
    except Exception:
        return None, False
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "\n".join(str(part) for part in content)
    summary = str(content).strip()
    return summary or None, bool(summary)


class _SummaryStreamCallback(BaseCallbackHandler):
    """Forward summary LLM tokens to LangGraph custom streaming."""

    def __init__(self) -> None:
        self._writer = None

    def on_llm_new_token(self, token: str, **_: Any) -> None:
        if not token:
            return
        try:
            if self._writer is None:
                self._writer = get_stream_writer()
            self._writer({"event": "summary_token", "data": token})
        except Exception:
            return


def _summary_context(state: FinancialResearchState) -> dict[str, Any]:
    results = state.get("retrieval_results", [])
    selected_categories = state.get("categories") or [result["category"] for result in results]
    return {
        "original_query": state.get("original_query") or state.get("active_query") or "",
        "active_query": state.get("active_query") or state.get("original_query") or "",
        "detected_categories": selected_categories,
        "detected_tickers": state.get("detected_tickers", []),
        "detected_companies": state.get("detected_companies", []),
        "retrieval_results": results,
        "citations": state.get("aggregated_citations", state.get("citations", [])),
        "warnings": state.get("warnings", []),
        "errors": state.get("errors", []),
    }


def _deterministic_summary(state: FinancialResearchState) -> str:
    """Create a local source-grounded summary when OpenAI is not configured."""

    results = state.get("retrieval_results", [])
    results_by_category = {result["category"]: result for result in results}
    sections: list[str] = []

    sections.append(_section("Quick Answer", _quick_answer(results)))

    optional_sections = [
        ("Stock Snapshot", _stock_snapshot(results_by_category.get(CATEGORY_CURRENT_STOCK_PRICE))),
        ("Company Overview", _company_overview(results_by_category.get(CATEGORY_COMPANY_INFORMATION))),
        ("Recent News", _recent_news(results_by_category.get(CATEGORY_COMPANY_NEWS))),
        ("Financials", _financials(results_by_category.get(CATEGORY_FINANCIALS))),
        ("Analyst View", _analyst_view(results_by_category.get(CATEGORY_ANALYST_RATINGS), state)),
        ("Technical Picture", _technical_picture(results_by_category.get(CATEGORY_TECHNICAL_ANALYSIS))),
        (
            "Sector / Competitor Context",
            _sector_competitor_context(
                results_by_category.get(CATEGORY_SECTOR_NEWS),
                results_by_category.get(CATEGORY_COMPETITOR_NEWS),
                results_by_category.get(CATEGORY_DOW_JONES_INDEX),
            ),
        ),
        ("Latest SEC Filings", _latest_sec_filings(results_by_category.get(CATEGORY_LATEST_REPORT_RELEASE))),
        ("Risks and Limitations", _risks_and_limitations(state, results)),
        ("Sources", _sources(state.get("aggregated_citations", state.get("citations", [])))),
    ]

    for title, body in optional_sections:
        if body:
            sections.append(_section(title, body))

    sections.append(EDUCATIONAL_DISCLAIMER)
    return "\n\n".join(sections)


def _section(title: str, body: str) -> str:
    return f"## {title}\n{body.strip()}"


def _quick_answer(results: list[RetrievalResult]) -> str:
    if not results:
        return "No retrieved data is available yet."

    lines = []
    for result in results:
        summary = result.get("summary")
        if summary:
            lines.append(f"- {summary}")
    return "\n".join(lines) or "Retrieved results did not include a usable summary."


def _stock_snapshot(result: RetrievalResult | None) -> str:
    if not result:
        return ""
    data = result.get("data", {})
    lines = _failed_or_empty_lines(result)
    if lines:
        return "\n".join(lines)

    fields = [
        ("Ticker", data.get("ticker")),
        ("Current price", data.get("current_price")),
        ("Day change", data.get("day_change")),
        ("Percent change", data.get("percent_change")),
        ("Previous close", data.get("previous_close")),
        ("Market timestamp", data.get("market_timestamp")),
    ]
    return _field_lines(fields)


def _company_overview(result: RetrievalResult | None) -> str:
    if not result:
        return ""
    data = result.get("data", {})
    lines = _failed_or_empty_lines(result)
    if lines:
        return "\n".join(lines)

    fields = [
        ("Company", data.get("company_name")),
        ("Ticker", data.get("ticker")),
        ("Sector", data.get("sector")),
        ("Industry", data.get("industry")),
        ("Market cap", data.get("market_cap")),
        ("Headquarters", data.get("headquarters")),
        ("Website", data.get("website")),
    ]
    business_summary = data.get("business_summary")
    output = _field_lines(fields)
    if business_summary:
        output = f"{output}\n- Business summary: {business_summary}"
    return output


def _recent_news(result: RetrievalResult | None) -> str:
    if not result:
        return ""
    lines = _failed_or_empty_lines(result)
    if lines:
        return "\n".join(lines)

    articles = result.get("data", {}).get("results", [])
    if not articles:
        return f"- {result.get('summary', 'No recent news results were available.')}"

    return "\n".join(_article_line(article) for article in articles[:3])


def _financials(result: RetrievalResult | None) -> str:
    if not result:
        return ""
    data = result.get("data", {})
    lines = _failed_or_empty_lines(result)
    if lines:
        return "\n".join(lines)

    fields = [
        ("Revenue", data.get("revenue")),
        ("Net income", data.get("net_income")),
        ("Operating income", data.get("operating_income")),
        ("EPS", data.get("eps")),
        ("Operating cash flow", data.get("operating_cash_flow")),
        ("Free cash flow", data.get("free_cash_flow")),
        ("Period", data.get("period")),
    ]
    output = _field_lines(fields)
    hit_miss = data.get("earnings_hit_miss")
    if hit_miss:
        output = f"{output}\n- Earnings hit/miss: {hit_miss}"
    return output


def _analyst_view(result: RetrievalResult | None, state: FinancialResearchState) -> str:
    if not result:
        return ""
    data = result.get("data", {})
    lines = _failed_or_empty_lines(result)
    if lines:
        return "\n".join(lines)

    consensus = data.get("consensus_rating")
    if not consensus:
        return "- Consensus rating: unavailable from retrieved data"

    output = [f"- Consensus rating: {consensus}"]
    if _query_asks_for_details(state):
        counts = data.get("buy_hold_sell_counts")
        target = data.get("price_target") or {}
        if counts:
            output.append(f"- Rating counts: {counts}")
        if target.get("target_mean") is not None:
            output.append(f"- Mean price target: {target.get('target_mean')}")
        if data.get("upside_downside_percent") is not None:
            output.append(f"- Implied move versus current price: {data.get('upside_downside_percent')}%")
        commentary = data.get("recent_analyst_commentary") or []
        output.extend(_article_line(article) for article in commentary[:3])
    return "\n".join(output)


def _technical_picture(result: RetrievalResult | None) -> str:
    if not result:
        return ""
    data = result.get("data", {})
    indicators = data.get("indicators", {})
    lines = _failed_or_empty_lines(result)
    if lines:
        return "\n".join(lines)

    fields = [
        ("Trend description", indicators.get("trend")),
        ("Latest close", indicators.get("latest_close")),
        ("SMA 20", indicators.get("sma_20")),
        ("SMA 50", indicators.get("sma_50")),
        ("SMA 200", indicators.get("sma_200")),
        ("RSI 14", indicators.get("rsi_14")),
        ("MACD", (indicators.get("macd") or {}).get("macd")),
        ("MACD signal", (indicators.get("macd") or {}).get("signal")),
        ("Volatility", indicators.get("volatility")),
        ("Approximate support", indicators.get("support")),
        ("Approximate resistance", indicators.get("resistance")),
    ]
    output = _field_lines(fields)
    return f"{output}\n- Technical indicators are descriptive only and are not investment advice."


def _sector_competitor_context(
    sector_result: RetrievalResult | None,
    competitor_result: RetrievalResult | None,
    dow_result: RetrievalResult | None,
) -> str:
    parts = []
    for result in (sector_result, competitor_result, dow_result):
        if not result:
            continue
        if result["category"] in {CATEGORY_SECTOR_NEWS, CATEGORY_COMPETITOR_NEWS}:
            parts.append(_recent_news(result))
        else:
            parts.append(result.get("summary", ""))
    return "\n".join(part for part in parts if part)


def _latest_sec_filings(result: RetrievalResult | None) -> str:
    if not result:
        return ""
    lines = _failed_or_empty_lines(result)
    if lines:
        return "\n".join(lines)

    filings = result.get("data", {}).get("filings", {})
    output = []
    for label, filing in (
        ("Latest 10-K", filings.get("latest_10k")),
        ("Latest 10-Q", filings.get("latest_10q")),
        ("Latest 8-K", filings.get("latest_8k")),
    ):
        if not filing:
            continue
        output.append(
            "- "
            f"{label}: filed {filing.get('filing_date')}; "
            f"report period {filing.get('report_period')}; "
            f"accession {filing.get('accession_number')}; "
            f"{filing.get('filing_url')}"
        )
    return "\n".join(output)


def _risks_and_limitations(state: FinancialResearchState, results: list[RetrievalResult]) -> str:
    lines = []
    warnings = state.get("warnings", [])
    errors = state.get("errors", [])
    failed = [result for result in results if result.get("status") == "failed"]
    partial = [result for result in results if result.get("status") == "partial"]

    if partial:
        lines.append("- Some requested data was only partially available.")
    for result in failed:
        lines.append(f"- {result['category']} failed: {'; '.join(result.get('errors') or [result.get('summary', '')])}")
    for warning in warnings[:5]:
        lines.append(f"- {warning}")
    for error in errors[:3]:
        lines.append(f"- {error}")

    lines.append("- Facts are limited to retrieved source data and may reflect provider delays or missing fields.")
    return "\n".join(_dedupe_lines(lines))


def _sources(citations: list[SourceCitation]) -> str:
    if not citations:
        return "No source citations were available in retrieved data."

    lines = []
    for index, citation in enumerate(citations, start=1):
        published = f", published {citation['published_at']}" if citation.get("published_at") else ""
        url = f" - {citation['url']}" if citation.get("url") else ""
        snippet = f" Evidence: {citation['snippet']}" if citation.get("snippet") else ""
        lines.append(
            f"{index}. {citation['provider']} - {citation['title']}{published} - "
            f"retrieved {citation['retrieved_at']}{url}.{snippet}"
        )
    return "\n".join(lines)


def _field_lines(fields: list[tuple[str, Any]]) -> str:
    lines = [f"- {label}: {value}" for label, value in fields if value is not None]
    return "\n".join(lines)


def _failed_or_empty_lines(result: RetrievalResult) -> list[str]:
    if result.get("status") != "failed":
        return []
    message = result.get("summary") or f"{result['category']} was unavailable."
    return [f"- {message}"]


def _article_line(article: dict[str, Any]) -> str:
    title = article.get("title") or "Untitled source"
    published = f" ({article['published_at']})" if article.get("published_at") else ""
    url = f" - {article['url']}" if article.get("url") else ""
    snippet = f": {article['snippet']}" if article.get("snippet") else ""
    return f"- {title}{published}{url}{snippet}"


def _query_asks_for_details(state: FinancialResearchState) -> bool:
    query = (state.get("active_query") or state.get("original_query") or "").lower()
    detail_terms = ("detail", "details", "why", "explain", "commentary", "price target", "breakdown")
    return any(term in query for term in detail_terms)


def _dedupe_lines(lines: list[str]) -> list[str]:
    deduped = []
    seen = set()
    for line in lines:
        key = " ".join(line.lower().split())
        if key and key not in seen:
            deduped.append(line)
            seen.add(key)
    return deduped
