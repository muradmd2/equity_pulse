"""Structured LLM output models for query classification."""

from pydantic import BaseModel, ConfigDict, Field

from app.models.types import CategoryName


class QueryClassification(BaseModel):
    """Classification output produced by the routing LLM."""

    model_config = ConfigDict(extra="forbid")

    detected_tickers: list[str] = Field(default_factory=list)
    detected_companies: list[str] = Field(default_factory=list)
    categories: list[CategoryName] = Field(default_factory=list)
    reasoning: str = ""
