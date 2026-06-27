"""Shared type aliases for financial research models."""

from typing import Literal, TypeAlias

CategoryName: TypeAlias = Literal[
    "company_information",
    "competitor_news",
    "sector_news",
    "financials",
    "company_news",
    "current_stock_price",
    "analyst_ratings",
    "latest_report_release",
    "dow_jones_index",
    "technical_analysis",
    "web_search",
]

ResultStatus: TypeAlias = Literal["success", "partial", "failed"]
