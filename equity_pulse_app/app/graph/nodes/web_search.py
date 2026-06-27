"""Tavily web search node for review gaps."""

from app.constants import CATEGORY_WEB_SEARCH, STATUS_FAILED, STATUS_PARTIAL, STATUS_SUCCESS
from app.models.citations import SourceCitation
from app.models.retrieval import RetrievalResult
from app.models.state import FinancialResearchState
from app.services.tavily_service import TavilyService
from app.utils.tracing import traced


@traced("web_search_node")
def web_search_node(state: FinancialResearchState) -> dict[str, object]:
    """Search Tavily for the unanswered part found by the review node."""

    query = state.get("web_search_query") or ""

    try:                                                                    
        data = TavilyService().search(query=query)                      
        results = data.get("results", [])
        status = STATUS_SUCCESS if results else STATUS_PARTIAL
        summary = _summary(query, results)
        citations = [_citation(index, article) for index, article in enumerate(results, start=1)]
        errors: list[str] = []
    except Exception as exc:
        data = {"query": query, "results": []}
        status = STATUS_FAILED
        summary = f"Web search failed for missing information: {query}"
        citations = []
        errors = [str(exc)]

    result: RetrievalResult = {
        "category": CATEGORY_WEB_SEARCH,
        "subquery": query,
        "status": status,
        "data": data,
        "summary": summary,
        "citations": citations,
        "errors": errors,
    }

    return {
        "retrieval_results": [result],
        "citations": citations,
        "iteration": 2,
    }


def _summary(query: str, results: list[dict[str, object]]) -> str:
    if not results:
        return f"Tavily returned no web search results for missing information: {query}"
    return f"Tavily returned {len(results)} web search result(s) for missing information: {query}"


def _citation(index: int, article: dict[str, object]) -> SourceCitation:
    return {
        "source_id": f"tavily-web-search-{index}",
        "title": str(article.get("title") or "Untitled source"),
        "url": str(article.get("url") or ""),
        "provider": "tavily",
        "published_at": article.get("published_at") if isinstance(article.get("published_at"), str) else None,
        "retrieved_at": str(article.get("retrieved_at") or ""),
        "snippet": article.get("snippet") if isinstance(article.get("snippet"), str) else None,
    }
