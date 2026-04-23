"""Typed response schemas for SEC EDGAR payloads.

EDGAR's submissions endpoint returns parallel arrays rather than a list of
records. :func:`iter_filings` zips those arrays into structured
:class:`NormalizedFiling` instances so the rest of the pipeline sees a clean
record stream.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TickerRecord(BaseModel):
    """Single entry in the EDGAR ticker-to-CIK map."""

    cik_str: int = Field(alias="cik_str")
    ticker: str
    title: str

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    @property
    def cik(self) -> str:
        """Return the CIK zero-padded to 10 characters (EDGAR's canonical form)."""

        return str(self.cik_str).zfill(10)


def parse_ticker_map(payload: dict[str, Any]) -> list[TickerRecord]:
    """Parse the ``company_tickers.json`` payload into a list of records."""

    return [TickerRecord.model_validate(value) for value in payload.values()]


class RecentFilings(BaseModel):
    """Parallel arrays describing a company's recent filings."""

    accession_number: list[str] = Field(alias="accessionNumber")
    filing_date: list[date] = Field(alias="filingDate")
    report_date: list[date | None] = Field(default_factory=list, alias="reportDate")
    form: list[str]
    primary_document: list[str] = Field(alias="primaryDocument")
    primary_doc_description: list[str] = Field(
        default_factory=list,
        alias="primaryDocDescription",
    )

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    @field_validator("report_date", mode="before")
    @classmethod
    def _blank_to_none(cls, value: Any) -> Any:
        """EDGAR encodes missing report dates as an empty string."""

        if isinstance(value, list):
            return [item if item not in ("", None) else None for item in value]
        return value


class SubmissionsFilings(BaseModel):
    """Top-level ``filings`` object from the submissions endpoint."""

    recent: RecentFilings

    model_config = ConfigDict(extra="ignore")


class SubmissionsResponse(BaseModel):
    """Typed subset of the EDGAR submissions JSON we care about."""

    cik: str
    name: str
    tickers: list[str] = Field(default_factory=list)
    exchanges: list[str] = Field(default_factory=list)
    sic: str | None = None
    sic_description: str | None = Field(default=None, alias="sicDescription")
    filings: SubmissionsFilings

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    @field_validator("cik", mode="before")
    @classmethod
    def _zero_pad_cik(cls, value: Any) -> Any:
        if isinstance(value, int):
            return str(value).zfill(10)
        if isinstance(value, str):
            return value.zfill(10)
        return value

    @property
    def primary_ticker(self) -> str | None:
        return self.tickers[0] if self.tickers else None

    @property
    def primary_exchange(self) -> str | None:
        return self.exchanges[0] if self.exchanges else None


@dataclass(frozen=True, slots=True)
class NormalizedFiling:
    """A single filing record in a consumer-friendly shape."""

    accession_number: str
    filing_date: date
    report_date: date | None
    form: str
    primary_document: str
    primary_doc_description: str | None


def iter_filings(response: SubmissionsResponse) -> Iterator[NormalizedFiling]:
    """Zip the parallel arrays from ``filings.recent`` into record form."""

    recent = response.filings.recent
    count = len(recent.accession_number)

    def _at(values: list[Any], index: int) -> Any:
        return values[index] if index < len(values) else None

    for i in range(count):
        yield NormalizedFiling(
            accession_number=recent.accession_number[i],
            filing_date=recent.filing_date[i],
            report_date=_at(recent.report_date, i),
            form=recent.form[i],
            primary_document=recent.primary_document[i],
            primary_doc_description=_at(recent.primary_doc_description, i),
        )


__all__ = [
    "NormalizedFiling",
    "RecentFilings",
    "SubmissionsFilings",
    "SubmissionsResponse",
    "TickerRecord",
    "iter_filings",
    "parse_ticker_map",
]
