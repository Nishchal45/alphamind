"""Unit tests for :func:`embed_chunks_for_document`.

The service mutates :class:`FilingChunk` ORM objects in place; persistence
happens via the session's autoflush. These tests use a fake session that
returns a canned list of in-memory chunk objects and counts flushes —
that's enough to verify the service's control flow without booting a
database. Behaviour-of-the-vector-column tests belong in integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from alphamind.embeddings.deterministic import DeterministicEmbedder
from alphamind.embeddings.service import embed_chunks_for_document
from alphamind.models.filing_chunk import FilingChunk

pytestmark = pytest.mark.asyncio


@dataclass
class FakeScalars:
    items: list[Any]

    def __iter__(self) -> Any:
        return iter(self.items)


@dataclass
class FakeResult:
    items: list[Any]

    def scalars(self) -> FakeScalars:
        return FakeScalars(items=self.items)


class FakeSession:
    """Returns canned chunks for any ``select`` and tracks flush calls."""

    def __init__(self, chunks: list[FilingChunk]) -> None:
        self._chunks = chunks
        self.flush_count = 0

    async def execute(self, stmt: Any) -> FakeResult:
        # The service issues exactly one SELECT — return all chunks.
        return FakeResult(items=list(self._chunks))

    async def flush(self) -> None:
        self.flush_count += 1


def _make_chunk(
    *,
    chunk_id: int,
    text: str,
    embedding: list[float] | None = None,
    embedding_model: str | None = None,
) -> FilingChunk:
    chunk = FilingChunk()
    chunk.id = chunk_id
    chunk.filing_document_id = 1
    chunk.section_label = "preamble"
    chunk.section_title = None
    chunk.chunk_index = chunk_id
    chunk.text = text
    chunk.char_count = len(text)
    chunk.source_content_hash = "a" * 64
    chunk.embedding = embedding
    chunk.embedding_model = embedding_model
    chunk.embedded_at = None
    return chunk


async def test_embeds_unembedded_chunks() -> None:
    chunks = [
        _make_chunk(chunk_id=1, text="first chunk"),
        _make_chunk(chunk_id=2, text="second chunk"),
    ]
    session = FakeSession(chunks=chunks)
    embedder = DeterministicEmbedder(dimension=32)

    result = await embed_chunks_for_document(
        session=session,  # type: ignore[arg-type]
        embedder=embedder,
        filing_document_id=1,
    )

    assert result.chunks_embedded == 2
    assert result.chunks_skipped == 0
    for chunk in chunks:
        assert chunk.embedding is not None
        assert len(chunk.embedding) == 32
        assert chunk.embedding_model == embedder.model_name
        assert chunk.embedded_at is not None


async def test_skips_chunks_already_embedded_under_same_model() -> None:
    embedder = DeterministicEmbedder(dimension=32)
    fresh_vec = [0.1] * 32
    chunks = [
        _make_chunk(
            chunk_id=1,
            text="already done",
            embedding=fresh_vec,
            embedding_model=embedder.model_name,
        ),
        _make_chunk(chunk_id=2, text="needs embedding"),
    ]
    session = FakeSession(chunks=chunks)

    result = await embed_chunks_for_document(
        session=session,  # type: ignore[arg-type]
        embedder=embedder,
        filing_document_id=1,
    )

    assert result.chunks_embedded == 1
    assert result.chunks_skipped == 1
    # The pre-embedded chunk's vector was not overwritten.
    assert chunks[0].embedding == fresh_vec


async def test_re_embeds_chunks_under_different_model() -> None:
    other_vec = [0.5] * 32
    chunks = [
        _make_chunk(
            chunk_id=1,
            text="needs re-embed",
            embedding=other_vec,
            embedding_model="some-other-model",
        ),
    ]
    session = FakeSession(chunks=chunks)
    embedder = DeterministicEmbedder(dimension=32)

    result = await embed_chunks_for_document(
        session=session,  # type: ignore[arg-type]
        embedder=embedder,
        filing_document_id=1,
    )

    assert result.chunks_embedded == 1
    assert result.chunks_skipped == 0
    assert chunks[0].embedding_model == embedder.model_name
    assert chunks[0].embedding != other_vec


async def test_force_re_embeds_chunks_under_same_model() -> None:
    embedder = DeterministicEmbedder(dimension=32)
    chunks = [
        _make_chunk(
            chunk_id=1,
            text="reembed me",
            embedding=[0.0] * 32,
            embedding_model=embedder.model_name,
        ),
    ]
    session = FakeSession(chunks=chunks)

    result = await embed_chunks_for_document(
        session=session,  # type: ignore[arg-type]
        embedder=embedder,
        filing_document_id=1,
        force=True,
    )

    assert result.chunks_embedded == 1
    assert result.chunks_skipped == 0


async def test_flushes_once_per_batch() -> None:
    chunks = [_make_chunk(chunk_id=i, text=f"chunk {i}") for i in range(5)]
    session = FakeSession(chunks=chunks)
    embedder = DeterministicEmbedder(dimension=16)

    await embed_chunks_for_document(
        session=session,  # type: ignore[arg-type]
        embedder=embedder,
        filing_document_id=1,
        batch_size=2,
    )

    # 5 chunks, batch=2 -> batches of (2, 2, 1) -> 3 flushes.
    assert session.flush_count == 3


async def test_does_nothing_when_no_chunks_exist() -> None:
    session = FakeSession(chunks=[])
    embedder = DeterministicEmbedder(dimension=16)

    result = await embed_chunks_for_document(
        session=session,  # type: ignore[arg-type]
        embedder=embedder,
        filing_document_id=42,
    )

    assert result.chunks_embedded == 0
    assert result.chunks_skipped == 0
    assert session.flush_count == 0


async def test_rejects_non_positive_batch_size() -> None:
    session = FakeSession(chunks=[])
    embedder = DeterministicEmbedder(dimension=16)

    with pytest.raises(ValueError, match="batch_size must be positive"):
        await embed_chunks_for_document(
            session=session,  # type: ignore[arg-type]
            embedder=embedder,
            filing_document_id=1,
            batch_size=0,
        )
