"""Fixtures for integration tests.

These tests require a live Postgres with the ``pgvector`` extension. The
GitHub Actions ``integration`` job provisions one via a service container;
locally, ``make compose-up && make migrate`` does the same.

The single most important invariant: each test starts with empty
``companies``, ``filings``, ``filing_documents``, and ``filing_chunks``
tables. ``TRUNCATE ... RESTART IDENTITY CASCADE`` clears them between
tests. We use truncate rather than transactional rollback because some
paths (HNSW index reads, generated tsvector materialisation) need rows to
be committed in order to be visible.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import date
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from alphamind.config import get_settings
from alphamind.models.company import Company
from alphamind.models.filing import Filing
from alphamind.models.filing_chunk import FilingChunk
from alphamind.models.filing_document import FilingDocument
from alphamind.storage.local import LocalFilesystemStorage

# Tables wiped between tests, in safe order. CASCADE handles the rest in
# case any of them grow new dependents later.
_RESET_TABLES = (
    "filing_chunks",
    "filing_documents",
    "filings",
    "companies",
)


@pytest_asyncio.fixture
async def db_engine() -> AsyncIterator[AsyncEngine]:
    """Async engine pointed at the test database.

    Function-scoped on purpose: pytest-asyncio's auto-mode uses a fresh
    event loop per test, so a session-scoped engine would carry references
    to a closed loop into later tests. Setup cost is one connection per
    test — negligible compared to the SQL the tests issue.
    """
    settings = get_settings()
    # NullPool because each engine lives one test; pooling would just hold
    # onto sockets nobody is going to reuse.
    engine = create_async_engine(settings.database_url, future=True, poolclass=NullPool)
    # Smoke-check that migrations have been applied and pgvector is
    # available. If not, fail fast with a useful message rather than the
    # first test blowing up with an opaque "relation does not exist".
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1 FROM filing_chunks LIMIT 0"))
        await conn.execute(text("SELECT extname FROM pg_extension WHERE extname='vector'"))
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    """A fresh AsyncSession with all four core tables truncated."""
    async with db_engine.begin() as conn:
        await conn.execute(text(f"TRUNCATE {', '.join(_RESET_TABLES)} RESTART IDENTITY CASCADE"))

    session_factory = async_sessionmaker(db_engine, expire_on_commit=False, class_=AsyncSession)
    async with session_factory() as session:
        yield session


@pytest.fixture
def storage(tmp_path: Path) -> LocalFilesystemStorage:
    """Per-test temporary local storage backend."""
    return LocalFilesystemStorage(root=tmp_path)


# ---------------------------------------------------------------------------
# Row factories — convenience helpers that keep tests readable.
# ---------------------------------------------------------------------------


async def make_company(
    session: AsyncSession,
    *,
    cik: str = "0000320193",
    ticker: str = "AAPL",
    name: str = "Apple Inc.",
) -> Company:
    company = Company(cik=cik, ticker=ticker, name=name)
    session.add(company)
    await session.flush()
    return company


async def make_filing(
    session: AsyncSession,
    *,
    company: Company,
    accession_number: str = "0000320193-24-000123",
    form: str = "10-K",
    filing_date: date = date(2024, 9, 28),
    primary_document: str = "aapl-20240928.htm",
) -> Filing:
    filing = Filing(
        company_id=company.id,
        accession_number=accession_number,
        form=form,
        filing_date=filing_date,
        primary_document=primary_document,
    )
    session.add(filing)
    await session.flush()
    return filing


async def make_filing_document(
    session: AsyncSession,
    *,
    filing: Filing,
    storage_uri: str = "file:///tmp/test",
    content_hash: str = "0" * 64,
    byte_size: int = 0,
    content_type: str = "text/html",
    source_url: str = "https://www.sec.gov/Archives/...",
) -> FilingDocument:
    document = FilingDocument(
        filing_id=filing.id,
        storage_uri=storage_uri,
        content_hash=content_hash,
        byte_size=byte_size,
        content_type=content_type,
        source_url=source_url,
    )
    session.add(document)
    await session.flush()
    return document


async def make_chunk(
    session: AsyncSession,
    *,
    filing: Filing,
    ordinal: int,
    text_body: str,
    section: str | None = "Item 1. Business",
    embedding: list[float] | None = None,
) -> FilingChunk:
    chunk = FilingChunk(
        filing_id=filing.id,
        filing_date=filing.filing_date,
        ordinal=ordinal,
        section=section,
        text=text_body,
        token_count=len(text_body.split()),
        char_start=0,
        char_end=len(text_body),
        embedding=embedding,
    )
    session.add(chunk)
    await session.flush()
    return chunk


# Re-export the row factories so tests can `from .conftest import make_*`.
__all__ = [
    "db_engine",
    "db_session",
    "make_chunk",
    "make_company",
    "make_filing",
    "make_filing_document",
    "storage",
]
