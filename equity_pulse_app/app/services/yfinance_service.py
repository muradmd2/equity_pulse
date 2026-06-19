"""yfinance data access helpers."""

from datetime import UTC, datetime
from typing import Any

import pandas as pd
import yfinance as yf

from app.utils.tracing import traced


class YFinanceService:
    """Small wrapper around yfinance with normalized return shapes."""

    @traced("yfinance.get_quote", run_type="tool")
    def get_quote(self, ticker: str) -> dict[str, Any]:
        ticker_symbol = _normalize_ticker(ticker)
        ticker_obj = yf.Ticker(ticker_symbol)
        info = _safe_dict(getattr(ticker_obj, "info", {}))
        fast_info = _safe_dict(getattr(ticker_obj, "fast_info", {}))
        history = self._history(ticker_obj, period="5d")

        latest_row = _latest_history_row(history)
        previous_row = _previous_history_row(history)

        current_price = _first_number(
            fast_info.get("last_price"),
            fast_info.get("lastPrice"),
            info.get("currentPrice"),
            info.get("regularMarketPrice"),
            latest_row.get("Close"),
        )
        previous_close = _first_number(
            fast_info.get("previous_close"),
            fast_info.get("previousClose"),
            info.get("previousClose"),
            previous_row.get("Close"),
        )

        if current_price is None and not info and history.empty:
            raise ValueError(f"No quote data returned for ticker '{ticker_symbol}'.")

        day_change = _calculate_change(current_price, previous_close)
        percent_change = _calculate_percent_change(day_change, previous_close)

        return {
            "ticker": ticker_symbol,
            "current_price": current_price,
            "day_change": day_change,
            "percent_change": percent_change,
            "previous_close": previous_close,
            "open": _first_number(info.get("open"), latest_row.get("Open")),
            "day_high": _first_number(info.get("dayHigh"), latest_row.get("High")),
            "day_low": _first_number(info.get("dayLow"), latest_row.get("Low")),
            "volume": _first_number(info.get("volume"), latest_row.get("Volume")),
            "market_timestamp": _history_timestamp(history),
            "retrieved_at": _now(),
        }

    @traced("yfinance.get_company_info", run_type="tool")
    def get_company_info(self, ticker: str) -> dict[str, Any]:
        ticker_symbol = _normalize_ticker(ticker)
        info = _safe_dict(getattr(yf.Ticker(ticker_symbol), "info", {}))
        if not info:
            raise ValueError(f"No company info returned for ticker '{ticker_symbol}'.")

        return {
            "ticker": ticker_symbol,
            "company_name": info.get("longName") or info.get("shortName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "business_summary": info.get("longBusinessSummary"),
            "market_cap": info.get("marketCap"),
            "employees": info.get("fullTimeEmployees"),
            "headquarters": _headquarters(info),
            "website": info.get("website"),
            "exchange": info.get("exchange"),
            "retrieved_at": _now(),
        }

    @traced("yfinance.get_basic_financials", run_type="tool")
    def get_basic_financials(self, ticker: str) -> dict[str, Any]:
        ticker_symbol = _normalize_ticker(ticker)
        ticker_obj = yf.Ticker(ticker_symbol)
        info = _safe_dict(getattr(ticker_obj, "info", {}))
        financials = _safe_dataframe(getattr(ticker_obj, "financials", pd.DataFrame()))
        balance_sheet = _safe_dataframe(getattr(ticker_obj, "balance_sheet", pd.DataFrame()))
        cashflow = _safe_dataframe(getattr(ticker_obj, "cashflow", pd.DataFrame()))

        data = {
            "ticker": ticker_symbol,
            "revenue": _statement_value(financials, "Total Revenue"),
            "net_income": _statement_value(financials, "Net Income"),
            "operating_income": _statement_value(financials, "Operating Income"),
            "gross_profit": _statement_value(financials, "Gross Profit"),
            "eps": info.get("trailingEps") or info.get("forwardEps"),
            "operating_cash_flow": _statement_value(cashflow, "Operating Cash Flow"),
            "free_cash_flow": _statement_value(cashflow, "Free Cash Flow"),
            "total_assets": _statement_value(balance_sheet, "Total Assets"),
            "total_debt": _statement_value(balance_sheet, "Total Debt"),
            "period": _latest_statement_period(financials),
            "retrieved_at": _now(),
        }

        if not any(value is not None for key, value in data.items() if key not in {"ticker", "retrieved_at"}):
            raise ValueError(f"No basic financial data returned for ticker '{ticker_symbol}'.")

        return data

    @traced("yfinance.get_historical_prices", run_type="tool")
    def get_historical_prices(self, ticker: str, period: str = "180d") -> dict[str, Any]:
        ticker_symbol = _normalize_ticker(ticker)
        history = self._history(yf.Ticker(ticker_symbol), period=period)
        if history.empty:
            raise ValueError(f"No historical price data returned for ticker '{ticker_symbol}'.")

        latest_row = _latest_history_row(history)
        return {
            "ticker": ticker_symbol,
            "period": period,
            "row_count": int(len(history)),
            "latest_close": _first_number(latest_row.get("Close")),
            "latest_date": _history_timestamp(history),
            "prices": _history_records(history),
            "retrieved_at": _now(),
        }

    @traced("yfinance.get_dow_jones_quote", run_type="tool")
    def get_dow_jones_quote(self) -> dict[str, Any]:
        quote = self.get_quote("^DJI")
        quote["index_name"] = "Dow Jones Industrial Average"
        return quote

    def _history(self, ticker_obj: Any, period: str) -> pd.DataFrame:
        history = ticker_obj.history(period=period)
        return _safe_dataframe(history)


def _normalize_ticker(ticker: str) -> str:
    value = ticker.strip().upper()
    if not value:
        raise ValueError("Ticker is required.")
    return value


def _safe_dict(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    try:
        return dict(value)
    except Exception:
        return {}


def _safe_dataframe(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    return pd.DataFrame()


def _latest_history_row(history: pd.DataFrame) -> dict[str, Any]:
    if history.empty:
        return {}
    return history.iloc[-1].to_dict()


def _previous_history_row(history: pd.DataFrame) -> dict[str, Any]:
    if len(history) < 2:
        return {}
    return history.iloc[-2].to_dict()


def _history_timestamp(history: pd.DataFrame) -> str | None:
    if history.empty:
        return None
    latest_index = history.index[-1]
    if hasattr(latest_index, "isoformat"):
        return latest_index.isoformat()
    return str(latest_index)


def _history_records(history: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in history.iterrows():
        records.append(
            {
                "date": index.isoformat() if hasattr(index, "isoformat") else str(index),
                "open": _first_number(row.get("Open")),
                "high": _first_number(row.get("High")),
                "low": _first_number(row.get("Low")),
                "close": _first_number(row.get("Close")),
                "volume": _first_number(row.get("Volume")),
            }
        )
    return records


def _first_number(*values: Any) -> float | int | None:
    for value in values:
        if value is None or pd.isna(value):
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _calculate_change(current_price: float | int | None, previous_close: float | int | None) -> float | None:
    if current_price is None or previous_close is None:
        return None
    return round(float(current_price) - float(previous_close), 4)


def _calculate_percent_change(day_change: float | None, previous_close: float | int | None) -> float | None:
    if day_change is None or previous_close in (None, 0):
        return None
    return round((day_change / float(previous_close)) * 100, 4)


def _headquarters(info: dict[str, Any]) -> str | None:
    parts = [info.get("city"), info.get("state"), info.get("country")]
    return ", ".join(part for part in parts if part) or None


def _statement_value(statement: pd.DataFrame, row_name: str) -> float | int | None:
    if statement.empty or row_name not in statement.index or len(statement.columns) == 0:
        return None
    return _first_number(statement.loc[row_name].iloc[0])


def _latest_statement_period(statement: pd.DataFrame) -> str | None:
    if statement.empty or len(statement.columns) == 0:
        return None
    period = statement.columns[0]
    if hasattr(period, "isoformat"):
        return period.isoformat()
    return str(period)


def _now() -> str:
    return datetime.now(UTC).isoformat()
