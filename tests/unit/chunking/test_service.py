"""Unit tests for the chunking persistence service.

These tests exercise :func:`chunk_filing_document` against a real on-disk
:class:`LocalFilesystemStorage` (so byte fetches are end-to-end) and a
hand-rolled fake :class:`AsyncSession` that captures issued statements
without needing a live Postgres. Behaviour-of-the-upsert tests belong in
integration; here we only verify the service's control flow.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from alphamind.chunking.service import chunk_filing_document
from alphamind.models.filing_document import FilingDocument
from alphamind.storage.local import LocalFilesystemStorage

pytestmark = pytest.mark.asyncio


@dataclass
class FakeResult:
    """Minimal stand-in for SQLAlchemy's Result — only the methods we use."""

    value: Any = None

    def scalar_one(self) -> Any:
        return self.value


class FakeSession:
    """Session double that returns canned counts for ``SELECT`` and records writes."""

    def __init__(self, *, existing_chunk_count: int = 0) -> None:
        self._existing_chunk_count = existing_chunk_count
        self.executed: list[Any] = []

    async def execute(self, stmt: Any) -> FakeResult:
        self.executed.append(stmt)
        stmt_str = str(stmt).strip().lower()
        if stmt_str.startswith("select"):
            return FakeResult(value=self._existing_chunk_count)
        return FakeResult()


async def _seed_body(tmp_path: Path, body: bytes) -> tuple[LocalFilesystemStorage, str, str]:
    storage = LocalFilesystemStorage(root=tmp_path)
    digest = hashlib.sha256(body).hexdigest()
    uri = await storage.put(key=digest, data=body)
    return storage, uri, digest


def _make_document(*, doc_id: int, storage_uri: str, content_hash: str) -> FilingDocument:
    doc = FilingDocument()
    doc.id = doc_id
    doc.filing_id = 1
    doc.storage_uri = storage_uri
    doc.content_hash = content_hash
    doc.byte_size = 0
    doc.content_type = "text/html"
    doc.source_url = "https://www.sec.gov/Archives/..."
    return doc


def _count_kind(executed: list[Any], kind: str) -> int:
    return sum(1 for s in executed if str(s).strip().lower().startswith(kind))


async def test_chunks_new_document_writes_rows(tmp_path: Path) -> None:
    body = (
        b"<html><body>"
        b"<p>Item 1. Business</p>"
        b"<p>We design and sell widgets globally.</p>"
        b"<p>Our segments include software and hardware.</p>"
        b"</body></html>"
    )
    storage, uri, digest = await _seed_body(tmp_path, body)
    document = _make_document(doc_id=7, storage_uri=uri, content_hash=digest)
    session = FakeSession(existing_chunk_count=0)

    result = await chunk_filing_document(
        storage=storage,
        session=session,  # type: ignore[arg-type]
        document=document,
    )

    assert result.was_skipped is False
    assert result.chunks_written >= 1
    # One SELECT (count), one DELETE, one INSERT.
    assert _count_kind(session.executed, "select") == 1
    assert _count_kind(session.executed, "delete") == 1
    assert _count_kind(session.executed, "insert") == 1


async def test_skips_when_hash_already_chunked(tmp_path: Path) -> None:
    body = b"<html><body><p>Item 1. Business</p><p>Body.</p></body></html>"
    storage, uri, digest = await _seed_body(tmp_path, body)
    document = _make_document(doc_id=7, storage_uri=uri, content_hash=digest)
    # Pretend filing_chunks already has rows for this (doc_id, content_hash).
    session = FakeSession(existing_chunk_count=4)

    result = await chunk_filing_document(
        storage=storage,
        session=session,  # type: ignore[arg-type]
        document=document,
    )

    assert result.was_skipped is True
    assert result.chunks_written == 0
    # Only the count SELECT runs; no DELETE/INSERT.
    assert _count_kind(session.executed, "select") == 1
    assert _count_kind(session.executed, "delete") == 0
    assert _count_kind(session.executed, "insert") == 0


async def test_force_rechunks_even_when_hash_matches(tmp_path: Path) -> None:
    body = b"<html><body><p>Item 1. Business</p><p>Body content here.</p></body></html>"
    storage, uri, digest = await _seed_body(tmp_path, body)
    document = _make_document(doc_id=7, storage_uri=uri, content_hash=digest)
    session = FakeSession(existing_chunk_count=4)

    result = await chunk_filing_document(
        storage=storage,
        session=session,  # type: ignore[arg-type]
        document=document,
        force=True,
    )

    assert result.was_skipped is False
    assert result.chunks_written >= 1
    # ``force=True`` skips the count SELECT entirely.
    assert _count_kind(session.executed, "select") == 0
    assert _count_kind(session.executed, "delete") == 1
    assert _count_kind(session.executed, "insert") == 1


async def test_rechunks_when_no_matching_hash_exists(tmp_path: Path) -> None:
    """Body in storage is unchanged but the prior chunks were derived from
    a different hash, so the count query returns zero and we re-chunk."""

    body = b"<html><body><p>Item 1. Business</p><p>Body paragraph.</p></body></html>"
    storage, uri, digest = await _seed_body(tmp_path, body)
    document = _make_document(doc_id=7, storage_uri=uri, content_hash=digest)
    session = FakeSession(existing_chunk_count=0)

    result = await chunk_filing_document(
        storage=storage,
        session=session,  # type: ignore[arg-type]
        document=document,
    )

    assert result.was_skipped is False
    assert result.chunks_written >= 1
    assert _count_kind(session.executed, "delete") == 1
    assert _count_kind(session.executed, "insert") == 1


async def test_writes_no_rows_when_body_has_no_text(tmp_path: Path) -> None:
    body = b"<html><body><style>p{color:red}</style></body></html>"
    storage, uri, digest = await _seed_body(tmp_path, body)
    document = _make_document(doc_id=7, storage_uri=uri, content_hash=digest)
    session = FakeSession(existing_chunk_count=0)

    result = await chunk_filing_document(
        storage=storage,
        session=session,  # type: ignore[arg-type]
        document=document,
    )

    assert result.was_skipped is False
    assert result.chunks_written == 0
    # DELETE still runs (clears any stale rows); no INSERT because no chunks.
    assert _count_kind(session.executed, "delete") == 1
    assert _count_kind(session.executed, "insert") == 0
