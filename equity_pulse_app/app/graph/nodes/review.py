"""Review node for one-time missing-information detection."""

import json

from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.models.review import ReviewResult
from app.models.state import FinancialResearchState
from app.prompts.llm_prompts import REVIEW_PROMPT
from app.utils.runtime_secrets import get_request_openai_api_key
from app.utils.tracing import traced


@traced("review_node")
def review_node(state: FinancialResearchState) -> dict[str, object]:
    """Ask the LLM whether the summary missed query information due to missing data."""

    if state.get("review_attempted"):
        return {
            "missing_info_found": False,
            "missing_info": [],
            "web_search_query": None,
            "review_reasoning": "Review skipped because it already ran once.",
        }

    review_count = state.get("review_count", 0) + 1
    result = _review_with_openai(state)

    return {
        "missing_info_found": result.missing_info_found,
        "missing_info": result.missing_info,
        "web_search_query": result.web_search_query if result.missing_info_found else None,
        "review_reasoning": result.reasoning,
        "review_attempted": True,
        "review_count": review_count,
    }


def should_search_after_review(state: FinancialResearchState) -> str:
    """Route to Tavily web search only when review found missing information."""

    if state.get("missing_info_found") and state.get("web_search_query"):
        return "web_search"
    return "final_response"


def _review_with_openai(state: FinancialResearchState) -> ReviewResult:
    settings = get_settings()
    api_key = get_request_openai_api_key(state.get("session_id")) or settings.openai_api_key
    model_name = state.get("model_name") or settings.openai_model

    try:
        llm = ChatOpenAI(model=model_name, temperature=0, api_key=api_key)
        reviewer = llm.with_structured_output(ReviewResult)
        raw_result = reviewer.invoke(REVIEW_PROMPT.format(context=json.dumps(_review_context(state), indent=2)))
        if isinstance(raw_result, ReviewResult):
            result = raw_result
        else:
            result = ReviewResult.model_validate(raw_result)
    except Exception as exc:
        return ReviewResult(reasoning=f"Review failed: {exc}")

    if not result.missing_info_found:
        return ReviewResult(reasoning=result.reasoning)

    return ReviewResult(
        missing_info_found=True,
        missing_info=result.missing_info,
        web_search_query=result.web_search_query,
        reasoning=result.reasoning,
    )


def _review_context(state: FinancialResearchState) -> dict[str, str]:
    return {
        "query": state.get("original_query") or state.get("active_query") or "",
        "summary_response": state.get("first_summary") or "",
    }
