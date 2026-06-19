"""Rewrite node for the one-time review regeneration loop."""

from app.models.state import FinancialResearchState
from app.utils.tracing import traced

REGENERATION_MESSAGE = "Review failed. Regenerating the response."


@traced("rewrite_query_node")
def rewrite_query_node(state: FinancialResearchState) -> dict[str, object]:
    """Prepare one focused retry for missing information only."""

    rewritten_query = state.get("rewritten_query") or _fallback_rewrite_query(state)
    return {
        "active_query": rewritten_query,
        "rewritten_query": rewritten_query,
        "review_attempted": True,
        "iteration": 2,
        "review_messages": [REGENERATION_MESSAGE],
        "missing_info_found": False,
    }


def _fallback_rewrite_query(state: FinancialResearchState) -> str:
    context_parts = [*state.get("detected_tickers", []), *state.get("detected_companies", [])]
    context = " ".join(context_parts) or state.get("original_query", "")
    missing_info = " ".join(state.get("missing_info", []))
    return f"{context} {missing_info}".strip()
