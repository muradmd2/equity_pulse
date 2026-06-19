"""LangGraph builder and invocation helpers."""

from collections.abc import Iterator
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

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
from app.config import get_settings
from app.graph.nodes.aggregate import aggregate_results_node
from app.graph.nodes.build_subqueries import build_subqueries_node
from app.graph.nodes.classify import classify_query_node
from app.graph.nodes.category_stubs import CATEGORY_NODE_NAMES, CATEGORY_NODES
from app.graph.nodes.final import final_response_node
from app.graph.nodes.review import review_node, should_retry_after_review
from app.graph.nodes.rewrite import REGENERATION_MESSAGE
from app.graph.nodes.rewrite import rewrite_query_node
from app.graph.nodes.summarize import summary_node
from app.models.state import FinancialResearchState
from app.utils.runtime_secrets import clear_request_openai_api_key, set_request_openai_api_key
from app.utils.tracing import configure_langsmith_environment, traced


def build_graph(checkpoint_db_path: str | None = None):
    """Build the graph with dynamic category fan-out and aggregate fan-in."""

    configure_langsmith_environment()
    builder = StateGraph(FinancialResearchState)

    builder.add_node("classify_query", classify_query_node)
    builder.add_node("build_subqueries", build_subqueries_node)
    for node_name, node_func in CATEGORY_NODES.items():
        builder.add_node(node_name, node_func)
    builder.add_node("aggregate_results", aggregate_results_node)
    builder.add_node("summary", summary_node)
    builder.add_node("review", review_node)
    builder.add_node("rewrite_query", rewrite_query_node)
    builder.add_node("final_response", final_response_node)

    builder.add_edge(START, "classify_query")
    builder.add_edge("classify_query", "build_subqueries")
    builder.add_conditional_edges("build_subqueries", route_category_nodes, CATEGORY_NODES.keys())
    for node_name in CATEGORY_NODES:
        builder.add_edge(node_name, "aggregate_results")
    builder.add_edge("aggregate_results", "summary")
    builder.add_conditional_edges(
        "summary",
        route_after_summary,
        {
            "review": "review",
            "final_response": "final_response",
        },
    )
    builder.add_conditional_edges(
        "review",
        should_retry_after_review,
        {
            "rewrite_query": "rewrite_query",
            "final_response": "final_response",
        },
    )
    builder.add_edge("rewrite_query", "classify_query")
    builder.add_edge("final_response", END)

    checkpointer = _create_checkpointer(checkpoint_db_path)
    if checkpointer:
        return builder.compile(checkpointer=checkpointer)
    return builder.compile()


def route_category_nodes(state: FinancialResearchState) -> list[str]:
    """Route only the category nodes selected by classification."""

    selected_nodes: list[str] = []
    seen: set[str] = set()

    for category in state.get("categories", []):
        node_name = CATEGORY_NODE_NAMES.get(category)
        if node_name and node_name not in seen:
            selected_nodes.append(node_name)
            seen.add(node_name)

    return selected_nodes


def route_after_summary(state: FinancialResearchState) -> str:
    """Run review only once, after the first summary."""

    if state.get("review_attempted") or state.get("review_count", 0) >= 1:
        return "final_response"
    return "review"


@traced("financial_research_workflow", run_type="chain")
def invoke_graph(
    query: str,
    session_id: str | None = None,
    model_name: str | None = None,
    openai_api_key: str | None = None,
    checkpoint_db_path: str | None = None,
) -> FinancialResearchState:
    """Run the graph end-to-end for a user query."""

    active_session_id = session_id or _new_session_id()
    effective_model_name = _effective_model_name(model_name)
    graph = build_graph(checkpoint_db_path)
    set_request_openai_api_key(active_session_id, openai_api_key)
    try:
        result = graph.invoke(
            _initial_state(
                query,
                active_session_id,
                model_name=effective_model_name,
            ),
            config=_thread_config(active_session_id),
        )
        return result
    finally:
        clear_request_openai_api_key(active_session_id)


@traced("resume_financial_research_session", run_type="chain")
def resume_session(
    session_id: str,
    checkpoint_db_path: str | None = None,
) -> FinancialResearchState:
    """Return the latest checkpointed state for a session."""

    graph = build_graph(checkpoint_db_path)
    snapshot = graph.get_state(_thread_config(session_id))
    return snapshot.values


@traced("stream_financial_research_workflow", run_type="chain")
def stream_graph_events(
    query: str,
    session_id: str | None = None,
    model_name: str | None = None,
    openai_api_key: str | None = None,
    checkpoint_db_path: str | None = None,
) -> Iterator[dict[str, object]]:
    """Stream progress, summary text, review messages, and final answer events."""

    active_session_id = session_id or _new_session_id()
    effective_model_name = _effective_model_name(model_name)
    graph = build_graph(checkpoint_db_path)
    set_request_openai_api_key(active_session_id, openai_api_key)
    yield _event("model", effective_model_name)
    yield _event("progress", "Analyzing query...")

    try:
        for chunk in graph.stream(
            _initial_state(
                query,
                active_session_id,
                model_name=effective_model_name,
            ),
            config=_thread_config(active_session_id),
            stream_mode=["updates", "custom"],
        ):
            if isinstance(chunk, tuple) and len(chunk) == 2:
                stream_mode, update = chunk
                if stream_mode == "custom":
                    if isinstance(update, dict) and update.get("event") == "summary_token":
                        yield _event("summary_token", update.get("data", ""))
                    continue
            else:
                update = chunk

            for node_name, node_update in update.items():
                if not isinstance(node_update, dict):
                    continue
                yield from _events_for_node(node_name, node_update)
    finally:
        clear_request_openai_api_key(active_session_id)


def stream_graph_sse(
    query: str,
    session_id: str | None = None,
    model_name: str | None = None,
    openai_api_key: str | None = None,
    checkpoint_db_path: str | None = None,
) -> Iterator[str]:
    """Stream graph events as server-sent event chunks."""

    for event in stream_graph_events(
        query,
        session_id=session_id,
        model_name=model_name,
        openai_api_key=openai_api_key,
        checkpoint_db_path=checkpoint_db_path,
    ):
        yield _sse(event["event"], event["data"])


def _initial_state(
    query: str,
    session_id: str,
    model_name: str | None = None,
) -> FinancialResearchState:
    return {
        "session_id": session_id,
        "original_query": query,
        "active_query": query,
        "model_name": model_name,
        "retrieval_results": [],
        "citations": [],
        "executed_nodes": [],
        "warnings": [],
        "errors": [],
        "summary_streamed": False,
        "review_attempted": False,
        "review_count": 0,
        "review_messages": [],
        "iteration": 1,
    }


def _create_checkpointer(checkpoint_db_path: str | None) -> SqliteSaver | None:
    db_path = checkpoint_db_path if checkpoint_db_path is not None else get_settings().sqlite_checkpoint_db
    if not db_path:
        return None

    resolved_path = Path(db_path)
    if resolved_path.parent != Path("."):
        resolved_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(str(resolved_path), check_same_thread=False)
    checkpointer = SqliteSaver(connection)
    checkpointer.setup()
    return checkpointer


def _thread_config(session_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": session_id}}


def _new_session_id() -> str:
    return str(uuid4())


def _effective_model_name(model_name: str | None) -> str:
    return model_name or get_settings().openai_model


def _events_for_node(node_name: str, node_update: dict[str, object]) -> Iterator[dict[str, object]]:
    if node_name == "classify_query":
        categories = node_update.get("categories", [])
        if categories:
            readable_categories = ", ".join(_readable_category(str(category)) for category in categories)
            yield _event("progress", f"Detected categories: {readable_categories}")
        return

    if node_name == "build_subqueries":
        subqueries = node_update.get("subqueries", {})
        if isinstance(subqueries, dict):
            for category in subqueries:
                yield _event("progress", _retrieval_progress_message(str(category)))
        return

    if node_name in CATEGORY_NODES:
        yield _event("progress", f"Completed {_readable_node(node_name)}.")
        return

    if node_name == "aggregate_results":
        yield _event("progress", "Aggregating retrieved data and citations...")
        return

    if node_name == "summary":
        yield _event("progress", "Summarizing results...")
        summary = str(node_update.get("second_summary") or node_update.get("first_summary") or "")
        if node_update.get("summary_streamed"):
            return
        yield from _summary_token_events(summary)
        return

    if node_name == "review":
        if node_update.get("missing_info_found"):
            yield _event("progress", REGENERATION_MESSAGE)
        return

    if node_name == "rewrite_query":
        return

    if node_name == "final_response":
        yield _event("progress", "Finalizing response...")
        final_answer = str(node_update.get("final_summary") or "")
        yield _event("final", final_answer)


def _summary_token_events(summary: str) -> Iterator[dict[str, object]]:
    for token in summary.split():
        yield _event("summary_token", f"{token} ")


def _retrieval_progress_message(category: str) -> str:
    messages = {
        CATEGORY_COMPANY_INFORMATION: "Fetching company profile...",
        CATEGORY_COMPETITOR_NEWS: "Searching competitor news...",
        CATEGORY_SECTOR_NEWS: "Searching sector news...",
        CATEGORY_FINANCIALS: "Retrieving financials...",
        CATEGORY_COMPANY_NEWS: "Searching recent company news...",
        CATEGORY_CURRENT_STOCK_PRICE: "Fetching stock quote...",
        CATEGORY_ANALYST_RATINGS: "Retrieving analyst ratings...",
        CATEGORY_LATEST_REPORT_RELEASE: "Retrieving latest SEC filings...",
        CATEGORY_DOW_JONES_INDEX: "Fetching Dow Jones index data...",
        CATEGORY_TECHNICAL_ANALYSIS: "Calculating technical indicators...",
    }
    return messages.get(category, f"Retrieving {_readable_category(category)}...")


def _readable_category(category: str) -> str:
    return category.replace("_", " ")


def _readable_node(node_name: str) -> str:
    return node_name.replace("_", " ")


def _event(event_type: str, data: object) -> dict[str, object]:
    return {"event": event_type, "data": data}


def _sse(event_type: object, data: object) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
