"""Final response node."""

from app.models.state import FinancialResearchState
from app.utils.tracing import traced


@traced("final_response_node")
def final_response_node(state: FinancialResearchState) -> dict[str, str]:
    """Set the final summary from available first/second pass summaries."""

    first_summary = state.get("first_summary")
    second_summary = state.get("second_summary")

    if first_summary and second_summary:
        final_summary = (
            "## First Pass Summary\n"
            f"{first_summary}\n\n"
            "## Regenerated Summary\n"
            f"{second_summary}"
        )
    else:
        final_summary = second_summary or first_summary or ""
    return {"final_summary": final_summary}
