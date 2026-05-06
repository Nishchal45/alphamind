"""Ingestion service for SEC EDGAR.

Responsibilities:

- Resolve a caller-supplied ticker to a CIK using the EDGAR ticker map.
- Fetch the submissions document for each CIK.
- Upsert one Company row and (up to ``limit``) Filing rows per CIK under a
  single transaction, keyed on ``accession_number`` for idempotency.
- Fetch and persist filing primary-document bodies on demand, recording each
  body's location and SHA-256 hash in :class:`FilingDocument`.

All upserts use Postgres ``INSERT ... ON CONFLICT DO UPDATE`` so a repeated
ingest run is a no-op when nothing has changed upstream.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

from sqlalchemy import func, select
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
from alphamind.models.filing_document import FilingDocument
from alphamind.storage.base import StorageBackend

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Outcome of ingesting a single CIK's submissions document."""

    cik: str
    ticker: str | None
    name: str
    filings_seen: int
    filings_written: int


@dataclass(frozen=True, slots=True)
class BodyIngestResult:
    """Outcome of fetching primary document bodies for a single CIK."""

    cik: str
    bodies_seen: int
    bodies_written: int
    bodies_unchanged: int
    bodies_failed: int


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


async def fetch_and_store_filing_body(
    *,
    client: EdgarClient,
    storage: StorageBackend,
    session: AsyncSession,
    filing_id: int,
    cik: str,
    accession_number: str,
    primary_document: str,
) -> tuple[FilingDocument, bool]:
    """Fetch one filing's primary-document body and upsert its row.

    Returns ``(document, written)``. ``written`` is ``True`` when the body
    was new (or differed from the stored hash); ``False`` when the existing
    row already pointed at the same SHA-256.
    """

    body, content_type, source_url = await client.get_primary_document(
        cik=cik,
        accession_number=accession_number,
        primary_document=primary_document,
    )
    digest = hashlib.sha256(body).hexdigest()
    byte_size = len(body)

    existing_q = await session.execute(
        select(FilingDocument).where(FilingDocument.filing_id == filing_id)
    )
    existing: FilingDocument | None = existing_q.scalar_one_or_none()

    if existing is not None and existing.content_hash == digest:
        return existing, False

    storage_uri = await storage.put(key=digest, data=body)

    stmt = (
        insert(FilingDocument)
        .values(
            filing_id=filing_id,
            storage_uri=storage_uri,
            content_hash=digest,
            byte_size=byte_size,
            content_type=content_type,
            source_url=source_url,
        )
        .on_conflict_do_update(
            index_elements=[FilingDocument.filing_id],
            set_={
                "storage_uri": storage_uri,
                "content_hash": digest,
                "byte_size": byte_size,
                "content_type": content_type,
                "source_url": source_url,
                "fetched_at": func.now(),
            },
        )
    )
    await session.execute(stmt)

    fresh_q = await session.execute(
        select(FilingDocument).where(FilingDocument.filing_id == filing_id)
    )
    return fresh_q.scalar_one(), True


async def ingest_bodies_for_cik(
    client: EdgarClient,
    storage: StorageBackend,
    cik: str,
    *,
    form_types: frozenset[str] | None = None,
    limit: int | None = None,
) -> BodyIngestResult:
    """Fetch primary-document bodies for a company's already-ingested filings.

    Iterates over Filings already persisted for the CIK (optionally filtered
    by form, optionally capped at ``limit``). Each body is fetched, hashed,
    written to storage, and recorded in :class:`FilingDocument`.

    Per-filing failures are logged and counted in ``bodies_failed`` but do
    not abort the run — one bad document should not kill an overnight backfill.
    """

    padded_cik = cik.strip().zfill(10)

    async with session_scope() as session:
        company_q = await session.execute(
            select(Company).where(Company.cik == padded_cik)
        )
        company = company_q.scalar_one_or_none()
        if company is None:
            raise LookupError(
                f"company not ingested yet for cik={padded_cik!r}; "
                "run ingest_cik first"
            )

        filings_stmt = select(Filing).where(Filing.company_id == company.id)
        if form_types is not None:
            filings_stmt = filings_stmt.where(Filing.form.in_(form_types))
        filings_stmt = filings_stmt.order_by(Filing.filing_date.desc())
        if limit is not None:
            filings_stmt = filings_stmt.limit(limit)

        filings = list((await session.execute(filings_stmt)).scalars())

        seen = len(filings)
        written = 0
        unchanged = 0
        failed = 0

        for filing in filings:
            try:
                _, did_write = await fetch_and_store_filing_body(
                    client=client,
                    storage=storage,
                    session=session,
                    filing_id=filing.id,
                    cik=padded_cik,
                    accession_number=filing.accession_number,
                    primary_document=filing.primary_document,
                )
            except Exception:
                logger.exception(
                    "edgar body fetch failed cik=%s accession=%s",
                    padded_cik,
                    filing.accession_number,
                )
                failed += 1
                continue

            if did_write:
                written += 1
            else:
                unchanged += 1

    logger.info(
        "edgar body ingest cik=%s seen=%d written=%d unchanged=%d failed=%d",
        padded_cik,
        seen,
        written,
        unchanged,
        failed,
    )

    return BodyIngestResult(
        cik=padded_cik,
        bodies_seen=seen,
        bodies_written=written,
        bodies_unchanged=unchanged,
        bodies_failed=failed,
    )


__all__ = [
    "BodyIngestResult",
    "IngestResult",
    "TickerRecord",
    "fetch_and_store_filing_body",
    "ingest_bodies_for_cik",
    "ingest_cik",
    "ingest_ticker",
    "resolve_ticker_to_cik",
    "upsert_company",
    "upsert_filings",
]
