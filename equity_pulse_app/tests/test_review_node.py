from app.graph.nodes import review as review_module
from app.graph.nodes import web_search as web_search_module
from app.graph.nodes.review import review_node, should_search_after_review
from app.graph.nodes.web_search import web_search_node
from app.models.review import ReviewResult


class FakeReviewer:
    def __init__(self, result):
        self.result = result
        self.prompt = ""

    def invoke(self, prompt):
        self.prompt = prompt
        return self.result


class FakeChatOpenAI:
    reviewer = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def with_structured_output(self, schema):
        assert schema is ReviewResult
        return self.reviewer


def test_review_node_sends_query_and_summary_to_llm(monkeypatch):
    reviewer = FakeReviewer(
        ReviewResult(
            missing_info_found=False,
            missing_info=[],
            web_search_query="this should be ignored",
            reasoning="The summary answers the query.",
        )
    )
    FakeChatOpenAI.reviewer = reviewer
    monkeypatch.setattr(review_module, "ChatOpenAI", FakeChatOpenAI)

    result = review_node(
        {
            "original_query": "What is Apple's latest stock price?",
            "first_summary": "Apple's latest retrieved quote is included. ## Sources",
            "review_count": 0,
        }
    )

    assert result["missing_info_found"] is False
    assert result["missing_info"] == []
    assert result["web_search_query"] is None
    assert result["review_reasoning"] == "The summary answers the query."
    assert result["review_attempted"] is True
    assert result["review_count"] == 1
    assert "What is Apple's latest stock price?" in reviewer.prompt
    assert "Apple's latest retrieved quote is included." in reviewer.prompt


def test_review_node_returns_web_search_query_for_missing_information(monkeypatch):
    reviewer = FakeReviewer(
        {
            "missing_info_found": True,
            "missing_info": ["latest quote missing"],
            "web_search_query": "AAPL latest stock price quote",
            "reasoning": "The summary discusses news but not the requested quote.",
        }
    )
    FakeChatOpenAI.reviewer = reviewer
    monkeypatch.setattr(review_module, "ChatOpenAI", FakeChatOpenAI)

    result = review_node(
        {
            "original_query": "What is Apple's latest stock price?",
            "first_summary": "Apple had recent product news.",
            "review_count": 2,
        }
    )

    assert result["missing_info_found"] is True
    assert result["missing_info"] == ["latest quote missing"]
    assert result["web_search_query"] == "AAPL latest stock price quote"
    assert result["review_reasoning"] == "The summary discusses news but not the requested quote."
    assert result["review_attempted"] is True
    assert result["review_count"] == 3
    assert should_search_after_review(result) == "web_search"


def test_review_node_skips_when_already_attempted():
    result = review_node({"review_attempted": True})

    assert result["missing_info_found"] is False
    assert result["missing_info"] == []
    assert result["web_search_query"] is None
    assert "skipped" in result["review_reasoning"]


def test_web_search_node_adds_tavily_result(monkeypatch):
    class FakeTavilyService:
        def search(self, query):
            return {
                "query": query,
                "results": [
                    {
                        "title": "Apple quote",
                        "url": "https://example.com/aapl",
                        "published_at": "2026-06-26",
                        "retrieved_at": "2026-06-26T12:00:00+00:00",
                        "snippet": "AAPL traded at a recent quoted price.",
                    }
                ],
                "retrieved_at": "2026-06-26T12:00:00+00:00",
            }

    monkeypatch.setattr(web_search_module, "TavilyService", FakeTavilyService)

    result = web_search_node({"web_search_query": "AAPL latest stock price quote"})

    retrieval = result["retrieval_results"][0]
    assert retrieval["category"] == "web_search"
    assert retrieval["subquery"] == "AAPL latest stock price quote"
    assert retrieval["status"] == "success"
    assert result["citations"][0]["provider"] == "tavily"
    assert result["iteration"] == 2
