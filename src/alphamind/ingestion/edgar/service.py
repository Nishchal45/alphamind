"""Ingestion service for SEC EDGAR.

Responsibilities:

- Resolve a caller-supplied ticker to a CIK using the EDGAR ticker map.
- Fetch the submissions document for each CIK.
- Upsert one Company row and (up to ``limit``) Filing rows per CIK under a
  single transaction, keyed on ``accession_number`` for idempotency.

All upserts use Postgres ``INSERT ... ON CONFLICT DO UPDATE`` so a repeated
ingest run is a no-op when nothing has changed upstream.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from alphamind.db.session import session_scope
from alphamind.ingestion.edgar.client import EdgarClient
from alphamind.ingestion.edgar.schemas import (
    NormalizedFiling,
    SubmissionsResponse,
    TickerRecord,
    iter_filings,
    parse_ticker_map,
)
from alphamind.models.company import Company
from alphamind.models.filing import Filing

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Outcome of ingesting a single CIK's submissions document."""

    cik: str
    ticker: str | None
    name: str
    filings_seen: int
    filings_written: int


async def resolve_ticker_to_cik(client: EdgarClient, ticker: str) -> str:
    """Look up the CIK for a ticker via the EDGAR ticker map."""

    payload = await client.get_company_tickers()
    records = parse_ticker_map(payload)
    target = ticker.upper().strip()

    for record in records:
        if record.ticker.upper() == target:
            return record.cik

    raise LookupError(f"ticker not found in EDGAR map: {ticker!r}")


async def upsert_company(
    session: AsyncSession,
    *,
    cik: str,
    name: str,
    ticker: str | None,
    sic: str | None,
    sic_description: str | None,
    exchange: str | None,
) -> Company:
    """Insert or update a Company row keyed on CIK."""

    stmt = (
        insert(Company)
        .values(
            cik=cik,
            name=name,
            ticker=ticker,
            sic=sic,
            sic_description=sic_description,
            exchange=exchange,
        )
        .on_conflict_do_update(
            index_elements=[Company.cik],
            set_={
                "name": name,
                "ticker": ticker,
                "sic": sic,
                "sic_description": sic_description,
                "exchange": exchange,
            },
        )
    )
    await session.execute(stmt)

    result = await session.execute(select(Company).where(Company.cik == cik))
    return result.scalar_one()


async def upsert_filings(
    session: AsyncSession,
    *,
    company: Company,
    filings: list[NormalizedFiling],
) -> int:
    """Insert or update Filing rows keyed on accession_number.

    Returns the number of filings affected (new or updated).
    """

    if not filings:
        return 0

    rows = [
        {
            "company_id": company.id,
            "accession_number": f.accession_number,
            "form": f.form,
            "filing_date": f.filing_date,
            "report_date": f.report_date,
            "primary_document": f.primary_document,
            "primary_doc_description": f.primary_doc_description,
        }
        for f in filings
    ]

    stmt = insert(Filing).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=[Filing.accession_number],
        set_={
            "form": stmt.excluded.form,
            "filing_date": stmt.excluded.filing_date,
            "report_date": stmt.excluded.report_date,
            "primary_document": stmt.excluded.primary_document,
            "primary_doc_description": stmt.excluded.primary_doc_description,
        },
    )
    await session.execute(stmt)

    return len(rows)


async def ingest_cik(
    client: EdgarClient,
    cik: str,
    *,
    form_types: frozenset[str] | None = None,
    limit: int | None = None,
) -> IngestResult:
    """Fetch submissions for ``cik`` and upsert company + filings.

    Parameters
    ----------
    form_types:
        If given, only filings whose ``form`` is in this set are kept
        (e.g. ``frozenset({"10-K", "10-Q", "8-K"})``).
    limit:
        If given, only the most recent ``limit`` matching filings are kept.
    """

    payload = await client.get_submissions(cik)
    response = SubmissionsResponse.model_validate(payload)
    all_filings = list(iter_filings(response))

    if form_types is not None:
        filtered = [f for f in all_filings if f.form in form_types]
    else:
        filtered = all_filings

    if limit is not None:
        filtered = filtered[:limit]

    async with session_scope() as session:
        company = await upsert_company(
            session,
            cik=response.cik,
            name=response.name,
            ticker=response.primary_ticker,
            sic=response.sic,
            sic_description=response.sic_description,
            exchange=response.primary_exchange,
        )
        written = await upsert_filings(session, company=company, filings=filtered)

    logger.info(
        "edgar ingest cik=%s ticker=%s seen=%d written=%d",
        response.cik,
        response.primary_ticker,
        len(all_filings),
        written,
    )

    return IngestResult(
        cik=response.cik,
        ticker=response.primary_ticker,
        name=response.name,
        filings_seen=len(all_filings),
        filings_written=written,
    )


async def ingest_ticker(
    client: EdgarClient,
    ticker: str,
    *,
    form_types: frozenset[str] | None = None,
    limit: int | None = None,
) -> IngestResult:
    """Convenience wrapper: resolve the ticker then delegate to :func:`ingest_cik`."""

    cik = await resolve_ticker_to_cik(client, ticker)
    return await ingest_cik(client, cik, form_types=form_types, limit=limit)


__all__ = [
    "IngestResult",
    "TickerRecord",
    "ingest_cik",
    "ingest_ticker",
    "resolve_ticker_to_cik",
    "upsert_company",
    "upsert_filings",
]
