"""SEC EDGAR data access helpers."""

from datetime import UTC, datetime
import threading
import time
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.utils.tracing import traced

SEC_DATA_BASE_URL = "https://data.sec.gov"
SEC_ARCHIVES_BASE_URL = "https://www.sec.gov/Archives/edgar/data"
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


class SecEdgarService:
    """SEC EDGAR client with ticker mapping and basic rate limiting."""

    _rate_lock = threading.Lock()
    _last_request_at = 0.0

    def __init__(self, user_agent: str | None = None, timeout_seconds: float | None = None) -> None:
        settings = get_settings()
        self.user_agent = user_agent if user_agent is not None else settings.sec_user_agent
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else settings.sec_edgar_timeout_seconds
        self.min_request_interval_seconds = settings.sec_edgar_min_request_interval_seconds

        if not self.user_agent:
            raise ValueError("SEC_USER_AGENT is required for SEC EDGAR requests.")

    @traced("sec_edgar.ticker_to_cik", run_type="tool")
    def ticker_to_cik(self, ticker: str) -> str:
        """Resolve a ticker to a 10-digit SEC CIK string."""

        ticker_symbol = _normalize_ticker(ticker)
        mapping = self._get_company_tickers()
        for entry in mapping.values():
            if str(entry.get("ticker", "")).upper() == ticker_symbol:
                return str(entry["cik_str"]).zfill(10)
        raise ValueError(f"No SEC CIK mapping found for ticker '{ticker_symbol}'.")

    @traced("sec_edgar.get_company_submissions", run_type="tool")
    def get_company_submissions(self, ticker_or_cik: str) -> dict[str, Any]:
        cik = ticker_or_cik if ticker_or_cik.isdigit() else self.ticker_to_cik(ticker_or_cik)
        return self._get(f"{SEC_DATA_BASE_URL}/submissions/CIK{cik.zfill(10)}.json")

    @traced("sec_edgar.get_latest_10k", run_type="tool")
    def get_latest_10k(self, ticker: str) -> dict[str, Any]:
        return self._latest_filing(ticker, "10-K")

    @traced("sec_edgar.get_latest_10q", run_type="tool")
    def get_latest_10q(self, ticker: str) -> dict[str, Any]:
        return self._latest_filing(ticker, "10-Q")

    @traced("sec_edgar.get_latest_8k", run_type="tool")
    def get_latest_8k(self, ticker: str) -> dict[str, Any]:
        return self._latest_filing(ticker, "8-K")

    @traced("sec_edgar.get_latest_filings", run_type="tool")
    def get_latest_filings(self, ticker: str) -> dict[str, Any]:
        """Return latest 10-K, 10-Q, and 8-K metadata from official SEC submissions."""

        cik = self.ticker_to_cik(ticker)
        submissions = self.get_company_submissions(cik)
        company_name = submissions.get("name")
        filings = {
            "latest_10k": self._latest_from_submissions(cik, submissions, "10-K"),
            "latest_10q": self._latest_from_submissions(cik, submissions, "10-Q"),
            "latest_8k": self._latest_from_submissions(cik, submissions, "8-K"),
        }

        return {
            "ticker": _normalize_ticker(ticker),
            "cik": cik,
            "company_name": company_name,
            "filings": filings,
            "retrieved_at": _now(),
        }

    @traced("sec_edgar.get_company_facts", run_type="tool")
    def get_company_facts(self, ticker: str) -> dict[str, Any]:
        cik = self.ticker_to_cik(ticker)
        return self._get(f"{SEC_DATA_BASE_URL}/api/xbrl/companyfacts/CIK{cik}.json")

    def _latest_filing(self, ticker: str, form_type: str) -> dict[str, Any]:
        cik = self.ticker_to_cik(ticker)
        submissions = self.get_company_submissions(cik)
        filing = self._latest_from_submissions(cik, submissions, form_type)
        if not filing:
            raise ValueError(f"No latest {form_type} filing found for ticker '{ticker}'.")
        return filing

    def _latest_from_submissions(self, cik: str, submissions: dict[str, Any], form_type: str) -> dict[str, Any] | None:
        recent = submissions.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        report_dates = recent.get("reportDate", [])
        accession_numbers = recent.get("accessionNumber", [])
        primary_documents = recent.get("primaryDocument", [])

        for index, form in enumerate(forms):
            if form != form_type:
                continue

            accession_number = _safe_list_get(accession_numbers, index)
            if not accession_number:
                continue

            primary_document = _safe_list_get(primary_documents, index)
            return {
                "form_type": form_type,
                "filing_date": _safe_list_get(filing_dates, index),
                "accession_number": accession_number,
                "report_period": _safe_list_get(report_dates, index),
                "filing_url": _filing_url(cik, accession_number, primary_document),
                "primary_document": primary_document,
            }

        return None

    def _get_company_tickers(self) -> dict[str, Any]:
        return self._get(SEC_COMPANY_TICKERS_URL)

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        reraise=True,
    )
    @traced("sec_edgar.get", run_type="tool")
    def _get(self, url: str) -> Any:
        self._rate_limit()
        headers = {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": _host_for_url(url),
        }
        with httpx.Client(timeout=self.timeout_seconds, headers=headers) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.json()

    def _rate_limit(self) -> None:
        with self._rate_lock:
            elapsed = time.monotonic() - self.__class__._last_request_at
            if elapsed < self.min_request_interval_seconds:
                time.sleep(self.min_request_interval_seconds - elapsed)
            self.__class__._last_request_at = time.monotonic()


def _normalize_ticker(ticker: str) -> str:
    value = ticker.strip().upper()
    if not value:
        raise ValueError("Ticker is required.")
    return value


def _safe_list_get(values: list[Any], index: int) -> Any:
    if index >= len(values):
        return None
    return values[index]


def _filing_url(cik: str, accession_number: str, primary_document: str | None) -> str:
    cik_no_padding = str(int(cik))
    accession_no_dashes = accession_number.replace("-", "")
    if primary_document:
        return f"{SEC_ARCHIVES_BASE_URL}/{cik_no_padding}/{accession_no_dashes}/{primary_document}"
    return f"{SEC_ARCHIVES_BASE_URL}/{cik_no_padding}/{accession_no_dashes}/{accession_number}-index.html"


def _host_for_url(url: str) -> str:
    if url.startswith("https://www.sec.gov"):
        return "www.sec.gov"
    return "data.sec.gov"


def _now() -> str:
    return datetime.now(UTC).isoformat()
