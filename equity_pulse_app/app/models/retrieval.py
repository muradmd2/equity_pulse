"""Structured retrieval result models."""

from typing import Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from app.models.citations import SourceCitation, SourceCitationModel
from app.models.types import CategoryName, ResultStatus


class RetrievalResult(TypedDict):
    """Normalized result returned by every category node."""

    category: CategoryName
    subquery: str
    status: ResultStatus
    data: dict[str, Any]
    summary: str
    citations: list[SourceCitation]
    errors: list[str]


class RetrievalResultModel(BaseModel):
    """Pydantic result model for validation at service and API boundaries."""

    model_config = ConfigDict(extra="forbid")

    category: CategoryName
    subquery: str
    status: ResultStatus
    data: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    citations: list[SourceCitationModel] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
