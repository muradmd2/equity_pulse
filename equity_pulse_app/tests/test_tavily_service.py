from app.services import tavily_service
from app.services.tavily_service import TavilyService


class FakeTavilyClient:
    calls = []

    def __init__(self, api_key):
        self.api_key = api_key

    def search(self, **kwargs):
        self.calls.append({"api_key": self.api_key, **kwargs})
        return {
            "answer": "ALK next earnings timing was found.",
            "results": [
                {
                    "title": "ALK earnings date",
                    "url": "https://example.com/alk-earnings",
                    "published_date": "2026-06-26",
                    "content": "ALK has an upcoming quarterly earnings release.",
                    "score": 0.92,
                }
            ],
        }


def test_tavily_service_uses_tavily_client(monkeypatch):
    FakeTavilyClient.calls = []
    monkeypatch.setattr(tavily_service, "TavilyClient", FakeTavilyClient)

    result = TavilyService(api_key="tvly-test-key").search(
        query="When is ALKT releasing its next quarterly earning.",
        max_results=3,
    )

    assert FakeTavilyClient.calls == [
        {
            "api_key": "tvly-test-key",
            "query": "When is ALKT releasing its next quarterly earning.",
            "include_answer": "basic",
            "search_depth": "advanced",
            "max_results": 3,
        }
    ]
    assert result["answer"] == "ALK next earnings timing was found."
    assert result["results"][0]["title"] == "ALK earnings date"
    assert result["results"][0]["snippet"] == "ALK has an upcoming quarterly earnings release."
