"""LangGraph state schema."""

from typing import Annotated, TypedDict
import operator

from app.models.citations import SourceCitation
from app.models.retrieval import RetrievalResult
from app.models.types import CategoryName


class FinancialResearchState(TypedDict, total=False):
    """Shared graph state accumulated across LangGraph nodes."""

    original_query: str
    active_query: str
    model_name: str | None
    session_id: str
    rewritten_query: str | None
    detected_tickers: list[str]
    detected_companies: list[str]
    categories: list[CategoryName]
    classification_reasoning: str
    subqueries: dict[CategoryName, str]
    retrieval_results: Annotated[list[RetrievalResult], operator.add]
    citations: Annotated[list[SourceCitation], operator.add]
    executed_nodes: Annotated[list[str], operator.add]
    aggregated_citations: list[SourceCitation]
    aggregated_facts: list[str]
    successful_results: list[RetrievalResult]
    partial_results: list[RetrievalResult]
    failed_results: list[RetrievalResult]
    successful_categories: list[CategoryName]
    partial_categories: list[CategoryName]
    failed_categories: list[CategoryName]
    errors: Annotated[list[str], operator.add]
    warnings: Annotated[list[str], operator.add]
    first_summary: str | None
    second_summary: str | None
    summary_streamed: bool
    final_summary: str | None
    missing_info_found: bool
    missing_info: list[str]
    review_attempted: bool
    review_count: int
    review_messages: Annotated[list[str], operator.add]
    review_reasoning: str
    iteration: int
