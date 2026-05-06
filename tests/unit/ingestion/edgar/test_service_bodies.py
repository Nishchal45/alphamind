"""Unit tests for the body-ingestion service flow.

These tests exercise :func:`fetch_and_store_filing_body` with respx mocking
the EDGAR Archives endpoint, a real on-disk
:class:`alphamind.storage.local.LocalFilesystemStorage`, and a hand-rolled
fake :class:`AsyncSession` that captures ``execute`` calls without needing
a live Postgres instance.

Behaviour-of-the-upsert (does ``ON CONFLICT DO UPDATE`` actually upsert?)
belongs in an integration test against a real database — see issue #2 for
that follow-up.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import respx

from alphamind.ingestion.edgar.client import SEC_WWW_BASE, EdgarClient
from alphamind.ingestion.edgar.service import fetch_and_store_filing_body
from alphamind.models.filing_document import FilingDocument
from alphamind.storage.local import LocalFilesystemStorage

pytestmark = pytest.mark.asyncio


@dataclass
class FakeResult:
    """Minimal stand-in for SQLAlchemy's Result — only the methods we use."""

    value: Any = None

    def scalar_one_or_none(self) -> Any:
        return self.value

    def scalar_one(self) -> Any:
        if self.value is None:
            raise AssertionError("scalar_one called with no canned value")
        return self.value


class FakeSession:
    """Session double that returns canned results for select(), records execute().

    The service issues two ``select`` calls on the write path (read existing,
    then read back the upserted row) and one on the no-op path. We pre-queue
    a generic placeholder for the post-upsert read since the test only cares
    that the upsert was issued, not what the returned row looks like.
    """

    def __init__(self, *, existing: FilingDocument | None) -> None:
        self._queue: list[Any] = [
            existing,
            _make_filing_document(content_hash="post-upsert"),
        ]
        self.executed: list[Any] = []

    async def execute(self, stmt: Any) -> FakeResult:
        self.executed.append(stmt)
        # Heuristic: select(...) statements consume the queue; insert()/
        # update() statements just record and return an empty result.
        stmt_str = str(stmt).strip().lower()
        if stmt_str.startswith("select"):
            value = self._queue.pop(0) if self._queue else None
            return FakeResult(value=value)
        return FakeResult()


def _make_filing_document(*, content_hash: str) -> FilingDocument:
    doc = FilingDocument()
    doc.id = 1
    doc.filing_id = 42
    doc.storage_uri = "file:///prev"
    doc.content_hash = content_hash
    doc.byte_size = 0
    doc.content_type = "text/html"
    doc.source_url = "https://www.sec.gov/Archives/..."
    return doc


@pytest.fixture
def user_agent() -> str:
    return "AlphaMind test test@example.com"


def _archive_url(*, cik: str, accession: str, document: str) -> str:
    cik_int = int(cik.lstrip("0"))
    accession_dashless = accession.replace("-", "")
    return f"{SEC_WWW_BASE}/Archives/edgar/data/{cik_int}/{accession_dashless}/{document}"


async def test_writes_new_body_when_no_document_exists(
    tmp_path: Path,
    user_agent: str,
) -> None:
    body = b"<html>fresh 10-K</html>"
    expected_hash = hashlib.sha256(body).hexdigest()
    url = _archive_url(
        cik="0000320193",
        accession="0000320193-24-000123",
        document="aapl-20240928.htm",
    )

    storage = LocalFilesystemStorage(root=tmp_path)
    session = FakeSession(existing=None)

    async with respx.mock(assert_all_called=True) as mock:
        mock.get(url).respond(content=body, headers={"Content-Type": "text/html"})

        async with EdgarClient(user_agent=user_agent, rate=100) as client:
            _, written = await fetch_and_store_filing_body(
                client=client,
                storage=storage,
                session=session,  # type: ignore[arg-type]
                filing_id=42,
                cik="0000320193",
                accession_number="0000320193-24-000123",
                primary_document="aapl-20240928.htm",
            )

    assert written is True
    # Storage actually received the bytes under the SHA-256 key.
    stored_uri = f"file://{(tmp_path / expected_hash[:2] / expected_hash).resolve()}"
    assert await storage.exists(stored_uri)
    assert await storage.get(stored_uri) == body
    # An insert was issued (one select happened first, then the upsert).
    insert_stmts = [s for s in session.executed if "insert" in str(s).lower()]
    assert len(insert_stmts) == 1


async def test_skips_storage_write_when_hash_matches_existing(
    tmp_path: Path,
    user_agent: str,
) -> None:
    body = b"<html>unchanged 10-K</html>"
    digest = hashlib.sha256(body).hexdigest()
    url = _archive_url(
        cik="0000320193",
        accession="0000320193-24-000123",
        document="aapl-20240928.htm",
    )

    storage = LocalFilesystemStorage(root=tmp_path)
    existing = _make_filing_document(content_hash=digest)
    session = FakeSession(existing=existing)

    async with respx.mock() as mock:
        mock.get(url).respond(content=body, headers={"Content-Type": "text/html"})

        async with EdgarClient(user_agent=user_agent, rate=100) as client:
            doc, written = await fetch_and_store_filing_body(
                client=client,
                storage=storage,
                session=session,  # type: ignore[arg-type]
                filing_id=42,
                cik="0000320193",
                accession_number="0000320193-24-000123",
                primary_document="aapl-20240928.htm",
            )

    assert written is False
    assert doc is existing
    # No insert was issued because the body hash matched.
    insert_stmts = [s for s in session.executed if "insert" in str(s).lower()]
    assert insert_stmts == []
    # Storage stays empty — no .tmp/no real key was written.
    keys_on_disk = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert keys_on_disk == []


async def test_writes_when_existing_hash_differs(
    tmp_path: Path,
    user_agent: str,
) -> None:
    """Filing was amended in place: hash changes, we update the row."""

    body = b"<html>amended 10-K</html>"
    url = _archive_url(
        cik="0000320193",
        accession="0000320193-24-000123",
        document="aapl-20240928.htm",
    )

    storage = LocalFilesystemStorage(root=tmp_path)
    existing = _make_filing_document(content_hash="stale-hash")
    session = FakeSession(existing=existing)

    async with respx.mock() as mock:
        mock.get(url).respond(content=body, headers={"Content-Type": "text/html"})

        async with EdgarClient(user_agent=user_agent, rate=100) as client:
            _, written = await fetch_and_store_filing_body(
                client=client,
                storage=storage,
                session=session,  # type: ignore[arg-type]
                filing_id=42,
                cik="0000320193",
                accession_number="0000320193-24-000123",
                primary_document="aapl-20240928.htm",
            )

    assert written is True
    insert_stmts = [s for s in session.executed if "insert" in str(s).lower()]
    assert len(insert_stmts) == 1
