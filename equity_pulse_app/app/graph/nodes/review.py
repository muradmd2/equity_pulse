"""Review node for deciding whether one regeneration pass is needed."""

import json

from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.constants import (
    CATEGORY_COMPANY_INFORMATION,
    CATEGORY_DOW_JONES_INDEX,
    STATUS_FAILED,
)
from app.models.review import ReviewResult
from app.models.state import FinancialResearchState
from app.models.types import CategoryName
from app.prompts.llm_prompts import REVIEW_PROMPT
from app.utils.runtime_secrets import get_request_openai_api_key
from app.utils.tracing import traced


@traced("review_node")
def review_node(state: FinancialResearchState) -> dict[str, object]:
    """Review the first summary for major missing information."""

    if state.get("review_attempted", False):
        review_count = state.get("review_count", 0)
        result = ReviewResult(
            missing_info_found=False,
            missing_info=[],
            rewritten_query=None,
            reasoning="Review skipped because a regeneration attempt has already occurred.",
        )
        return _review_update(result, review_count)

    review_count = state.get("review_count", 0) + 1
    result = _review_with_openai(state)
    return _review_update(result, review_count)


def should_retry_after_review(state: FinancialResearchState) -> str:
    """Route to rewrite when the first review found missing information."""

    if state.get("missing_info_found") and not state.get("review_attempted", False):
        return "rewrite_query"
    return "final_response"


def _review_update(result: ReviewResult, review_count: int) -> dict[str, object]:
    return {
        "missing_info_found": result.missing_info_found,
        "missing_info": result.missing_info,
        "rewritten_query": result.rewritten_query,
        "review_reasoning": result.reasoning,
        "review_count": review_count,
    }


def _review_with_openai(state: FinancialResearchState) -> ReviewResult | None:
    settings = get_settings()
    api_key = get_request_openai_api_key(state.get("session_id")) or settings.openai_api_key
    model_name = state.get("model_name") or settings.openai_model

    try:
        llm = ChatOpenAI(model=model_name, temperature=0, api_key=api_key)
        reviewer = llm.with_structured_output(ReviewResult)
        context = json.dumps(_review_context(state), default=str, indent=2)
        raw_result = reviewer.invoke(REVIEW_PROMPT.format(context=context))
        if isinstance(raw_result, ReviewResult):
            result = raw_result
        else:
            result = ReviewResult.model_validate(raw_result)
    except Exception:
        return None

    missing_info = _dedupe(result.missing_info)
    missing_info_found = result.missing_info_found or bool(missing_info)
    rewritten_query = result.rewritten_query
    if missing_info_found and not rewritten_query:
        rewritten_query = _focused_rewrite_query(state, missing_info or _detect_missing_info(state))

    return ReviewResult(
        missing_info_found=missing_info_found,
        missing_info=missing_info,
        rewritten_query=rewritten_query if missing_info_found else None,
        reasoning=result.reasoning,
    )


def _review_context(state: FinancialResearchState) -> dict[str, object]:
    return {
        "original_query": state.get("original_query", ""),
        "active_query": state.get("active_query", ""),
        "detected_tickers": state.get("detected_tickers", []),
        "detected_companies": state.get("detected_companies", []),
        "selected_categories": state.get("categories", []),
        "first_summary": state.get("first_summary") or "",
        "retrieval_results": [
            {
                "category": result.get("category"),
                "status": result.get("status"),
                "summary": result.get("summary"),
                "errors": result.get("errors", []),
                "citation_count": len(result.get("citations", [])),
            }
            for result in state.get("retrieval_results", [])
        ],
        "aggregated_citation_count": len(state.get("aggregated_citations", [])),
    }



def _detect_missing_info(state: FinancialResearchState) -> list[str]:
    selected_categories = list(state.get("categories", []))
    results = state.get("retrieval_results", [])
    summary = state.get("first_summary") or ""
    result_categories = {result["category"] for result in results}
    missing_info: list[str] = []

    for category in selected_categories:
        if category not in result_categories:
            missing_info.append(f"missing_category:{category}")

    for result in results:
        if result.get("status") == STATUS_FAILED and result.get("category") in selected_categories:
            missing_info.append(f"failed_category:{result['category']}")
        if result.get("status") != STATUS_FAILED and not result.get("citations"):
            missing_info.append(f"missing_citations:{result['category']}")

    if selected_categories and not state.get("aggregated_citations") and "## Sources" not in summary:
        missing_info.append("missing_citations:sources")

    if _needs_ticker(selected_categories) and not state.get("detected_tickers"):
        missing_info.append("missing_ticker")

    if CATEGORY_COMPANY_INFORMATION in selected_categories:
        company_info = next(
            (result for result in results if result.get("category") == CATEGORY_COMPANY_INFORMATION),
            None,
        )
        company_name = (company_info or {}).get("data", {}).get("company_name")
        if not company_name and not state.get("detected_companies"):
            missing_info.append("missing_company_info")

    return _dedupe(missing_info)


def _needs_ticker(categories: list[CategoryName]) -> bool:
    non_ticker_categories = {CATEGORY_DOW_JONES_INDEX}
    return any(category not in non_ticker_categories for category in categories)


def _focused_rewrite_query(state: FinancialResearchState, missing_info: list[str]) -> str:
    context = _query_context(state)
    missing_categories = _missing_categories(missing_info)
    terms = [_category_rewrite_terms(category) for category in missing_categories]

    if not terms:
        terms = ["missing financial research data citations"]

    return " ".join([context, *terms]).strip()


def _query_context(state: FinancialResearchState) -> str:
    tickers = state.get("detected_tickers", [])
    companies = state.get("detected_companies", [])
    context_parts = [*tickers, *companies]
    if context_parts:
        return " ".join(_dedupe(context_parts))
    return state.get("original_query", "")


def _missing_categories(missing_info: list[str]) -> list[CategoryName]:
    categories: list[CategoryName] = []
    for item in missing_info:
        if ":" not in item:
            continue
        prefix, category = item.split(":", 1)
        if prefix in {"missing_category", "failed_category", "missing_citations"} and category != "sources":
            categories.append(category)  # type: ignore[arg-type]
    return _dedupe(categories)


def _category_rewrite_terms(category: CategoryName) -> str:
    terms = {
        "company_information": "company profile sector industry market cap",
        "competitor_news": "competitors news",
        "sector_news": "sector news",
        "financials": "financials revenue net income cash flow",
        "company_news": "latest company news",
        "current_stock_price": "current stock quote latest price",
        "analyst_ratings": "analyst ratings recommendation trends price target",
        "latest_report_release": "latest SEC report 10-K 10-Q 8-K filing",
        "dow_jones_index": "Dow Jones latest index level",
        "technical_analysis": "technical analysis historical prices SMA RSI MACD",
    }
    return terms.get(category, str(category))


def _dedupe(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped
