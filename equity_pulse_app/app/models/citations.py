"""Citation models used across retrieval and final responses."""

from typing import Optional, TypedDict

from pydantic import BaseModel, ConfigDict, Field


class SourceCitation(TypedDict):
    """Source metadata carried through the graph state."""

    source_id: str
    title: str
    url: str
    provider: str
    published_at: Optional[str]
    retrieved_at: str
    snippet: Optional[str]


class SourceCitationModel(BaseModel):
    """Pydantic citation model for validation at app boundaries."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(description="Stable source identifier for this graph run.")
    title: str = Field(description="Source title or provider-specific label.")
    url: str = Field(default="", description="Source URL, when available.")
    provider: str = Field(description="Data provider, such as yfinance, Tavily, SEC EDGAR, or Finnhub.")
    published_at: str | None = Field(default=None, description="Publication timestamp, when available.")
    retrieved_at: str = Field(description="Timestamp when the app retrieved the source.")
    snippet: str | None = Field(default=None, description="Short evidence text for the cited fact.")
