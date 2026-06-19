"""OpenAI-backed query classification node."""

from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.models.classification import QueryClassification
from app.models.state import FinancialResearchState
from app.prompts.llm_prompts import CLASSIFICATION_PROMPT
from app.utils.runtime_secrets import get_request_openai_api_key
from app.utils.tracing import traced


@traced("classify_query_node", run_type="chain")
def classify_query_node(state: FinancialResearchState) -> dict[str, object]:
    """Classify the active query with OpenAI structured output."""

    query = state.get("active_query") or state.get("original_query") or ""
    settings = get_settings()
    api_key = get_request_openai_api_key(state.get("session_id")) or settings.openai_api_key
    model_name = state.get("model_name") or settings.openai_model
    warnings: list[str] = []

    try:
        llm = ChatOpenAI(model=model_name, temperature=0, api_key=api_key)
        classifier = llm.with_structured_output(QueryClassification)
        raw_classification = classifier.invoke(CLASSIFICATION_PROMPT.format(query=query))
        if isinstance(raw_classification, QueryClassification):
            classification = raw_classification
        else:
            classification = QueryClassification.model_validate(raw_classification)
    except Exception as exc:
        return {
            "active_query": query,
            "detected_tickers": [],
            "detected_companies": [],
            "categories": [],
            "warnings": [f"Query classification failed: {exc}"],
            "classification_reasoning": "OpenAI classification failed; no categories were selected.",
        }

    detected_tickers = list(dict.fromkeys(classification.detected_tickers))
    detected_companies = list(dict.fromkeys(classification.detected_companies))

    return {
        "active_query": query,
        "detected_tickers": list(dict.fromkeys(detected_tickers)),
        "detected_companies": detected_companies,
        "categories": list(dict.fromkeys(classification.categories)),
        "warnings": warnings,
        "classification_reasoning": classification.reasoning,
    }
