from app.graph.nodes import review as review_module
from app.graph.nodes.review import review_node
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


def test_review_node_uses_llm_structured_output_when_summary_answers_query(monkeypatch):
    reviewer = FakeReviewer(
        ReviewResult(
            missing_info_found=False,
            missing_info=[],
            rewritten_query="this should be ignored",
            reasoning="The summary answers the query.",
        )
    )
    FakeChatOpenAI.reviewer = reviewer
    monkeypatch.setattr(review_module, "ChatOpenAI", FakeChatOpenAI)

    result = review_node(
        {
            "original_query": "What is Apple's latest stock price?",
            "active_query": "What is Apple's latest stock price?",
            "detected_tickers": ["AAPL"],
            "categories": ["current_stock_price"],
            "first_summary": "Apple's latest retrieved quote is included. ## Sources",
            "retrieval_results": [
                {
                    "category": "current_stock_price",
                    "subquery": "AAPL quote",
                    "status": "success",
                    "data": {},
                    "summary": "AAPL quote retrieved.",
                    "citations": [{"provider": "yfinance"}],
                    "errors": [],
                }
            ],
            "aggregated_citations": [{"provider": "yfinance"}],
            "review_count": 0,
        }
    )

    assert result["missing_info_found"] is False
    assert result["missing_info"] == []
    assert result["rewritten_query"] is None
    assert result["review_reasoning"] == "The summary answers the query."
    assert result["review_count"] == 1
    assert "What is Apple's latest stock price?" in reviewer.prompt


def test_review_node_uses_llm_missing_info_and_rewrite(monkeypatch):
    reviewer = FakeReviewer(
        {
            "missing_info_found": True,
            "missing_info": ["missing current stock price"],
            "rewritten_query": "AAPL current stock price latest quote",
            "reasoning": "The summary discusses news but not the requested quote.",
        }
    )
    FakeChatOpenAI.reviewer = reviewer
    monkeypatch.setattr(review_module, "ChatOpenAI", FakeChatOpenAI)

    result = review_node(
        {
            "original_query": "What is Apple's latest stock price?",
            "active_query": "What is Apple's latest stock price?",
            "detected_tickers": ["AAPL"],
            "categories": ["current_stock_price"],
            "first_summary": "Apple had recent product news.",
            "retrieval_results": [],
            "review_count": 2,
        }
    )

    assert result["missing_info_found"] is True
    assert result["missing_info"] == ["missing current stock price"]
    assert result["rewritten_query"] == "AAPL current stock price latest quote"
    assert result["review_reasoning"] == "The summary discusses news but not the requested quote."
    assert result["review_count"] == 3


def test_review_node_falls_back_to_structural_checks_when_llm_fails(monkeypatch):
    class FailingChatOpenAI:
        def __init__(self, **kwargs):
            pass

        def with_structured_output(self, schema):
            raise RuntimeError("no model available")

    monkeypatch.setattr(review_module, "ChatOpenAI", FailingChatOpenAI)

    result = review_node(
        {
            "original_query": "What is Apple's latest stock price?",
            "detected_tickers": ["AAPL"],
            "categories": ["current_stock_price"],
            "first_summary": "No quote was retrieved.",
            "retrieval_results": [],
            "review_count": 0,
        }
    )

    assert result["missing_info_found"] is True
    assert "missing_category:current_stock_price" in result["missing_info"]
    assert "missing_citations:sources" in result["missing_info"]
    assert "current stock quote latest price" in result["rewritten_query"]
