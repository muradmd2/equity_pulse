"""Aggregation node for normalized retrieval results."""

from typing import Any

from app.models.citations import SourceCitation
from app.models.retrieval import RetrievalResult
from app.models.state import FinancialResearchState
from app.utils.tracing import traced


@traced("aggregate_results_node")
def aggregate_results_node(
    state: FinancialResearchState,
) -> dict[str, list[SourceCitation] | list[str] | list[RetrievalResult]]:
    """Aggregate category outputs while preserving failures and source metadata."""

    results = state.get("retrieval_results", [])
    citations = _dedupe_citations(state.get("citations", []))

    successful_results = [result for result in results if result["status"] == "success"]
    partial_results = [result for result in results if result["status"] == "partial"]
    failed_results = [result for result in results if result["status"] == "failed"]

    return {
        "aggregated_citations": citations,
        "aggregated_facts": _dedupe_facts(results),
        "successful_results": successful_results,
        "partial_results": partial_results,
        "failed_results": failed_results,
        "successful_categories": [result["category"] for result in successful_results],
        "partial_categories": [result["category"] for result in partial_results],
        "failed_categories": [result["category"] for result in failed_results],
    }


def _dedupe_citations(citations: list[SourceCitation]) -> list[SourceCitation]:
    deduped: list[SourceCitation] = []
    seen: set[tuple[str, str]] = set()

    for citation in citations:
        key = _citation_key(citation)
        if key in seen:
            continue
        deduped.append(citation)
        seen.add(key)

    return deduped


def _citation_key(citation: SourceCitation) -> tuple[str, str]:
    provider = citation.get("provider", "").strip().lower()
    url = citation.get("url", "").strip().lower()
    source_id = citation.get("source_id", "").strip().lower()

    if url:
        return ("url", f"{provider}|{url}")
    if source_id:
        return ("source_id", f"{provider}|{source_id}")
    title = citation.get("title", "").strip().lower()
    return ("title", f"{provider}|{title}")


def _dedupe_facts(results: list[RetrievalResult]) -> list[str]:
    facts: list[str] = []
    seen: set[str] = set()

    for result in results:
        for fact in _facts_from_result(result):
            key = _normalize_fact(fact)
            if not key or key in seen:
                continue
            facts.append(fact)
            seen.add(key)

    return facts


def _facts_from_result(result: RetrievalResult) -> list[str]:
    facts = [result.get("summary", "")]
    data = result.get("data", {})

    if isinstance(data, dict):
        facts.extend(_facts_from_mapping(data))

    return [fact for fact in facts if fact]


def _facts_from_mapping(data: dict[str, Any], prefix: str = "") -> list[str]:
    facts: list[str] = []

    for key, value in data.items():
        if value in (None, "", [], {}):
            continue

        label = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            facts.extend(_facts_from_mapping(value, label))
        elif isinstance(value, list):
            facts.extend(_facts_from_list(label, value))
        else:
            facts.append(f"{label}: {value}")

    return facts


def _facts_from_list(label: str, values: list[Any]) -> list[str]:
    facts: list[str] = []

    for index, item in enumerate(values[:5], start=1):
        if item in (None, "", [], {}):
            continue
        if isinstance(item, dict):
            facts.extend(_facts_from_mapping(item, f"{label}[{index}]"))
        else:
            facts.append(f"{label}[{index}]: {item}")

    return facts


def _normalize_fact(fact: str) -> str:
    return " ".join(fact.lower().split())
