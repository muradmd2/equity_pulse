"""Structured LLM output models for answer review."""

from pydantic import BaseModel, ConfigDict, Field


class ReviewResult(BaseModel):
    """Review output used to decide whether one regeneration is needed."""

    model_config = ConfigDict(extra="forbid")

    missing_info_found: bool = False
    missing_info: list[str] = Field(default_factory=list)
    rewritten_query: str | None = None
    reasoning: str = ""
