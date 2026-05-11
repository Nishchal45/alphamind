"""ORM model for a single chunk derived from a filing's primary document.

A :class:`FilingChunk` row records one paragraph-aware slice of a filing
body. Chunks are derived from the :class:`FilingDocument`'s bytes, so the
foreign key points there rather than directly at :class:`Filing`.

``source_content_hash`` stores the SHA-256 of the body the chunk was
derived from. The chunking service compares it to the parent document's
current ``content_hash`` to decide whether existing chunks are still valid
or need to be regenerated — bodies on EDGAR are immutable for a given
accession number in practice, but the column makes the staleness check
trivial and explicit.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from alphamind.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from alphamind.models.filing_document import FilingDocument

# Default embedding dimension. Must match ``embedder_dimension`` in
# :class:`alphamind.config.Settings` and the ``vector(N)`` size set by
# migration ``0006_widen_embedding_to_768``. Picked to match Gemini's
# ``text-embedding-004`` output. Changing this is a migration.
EMBEDDING_DIMENSION = 768


class FilingChunk(Base, TimestampMixin):
    """One retrieval-ready slice of a filing's primary document."""

    __tablename__ = "filing_chunks"
    __table_args__ = (
        UniqueConstraint("filing_document_id", "chunk_index"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    filing_document_id: Mapped[int] = mapped_column(
        ForeignKey("filing_documents.id", ondelete="CASCADE"),
        index=True,
    )

    # Lowercase slug such as ``"item_1a"`` or ``"preamble"``. Indexed so the
    # retrieval layer can filter to a specific section cheaply.
    section_label: Mapped[str] = mapped_column(String(64), index=True)
    section_title: Mapped[str | None] = mapped_column(String(256))

    # Zero-based position of this chunk within its parent document, in
    # reading order. Paired with ``filing_document_id`` to form the natural
    # uniqueness constraint for upserts.
    chunk_index: Mapped[int] = mapped_column(Integer)

    text: Mapped[str] = mapped_column(Text)
    char_count: Mapped[int] = mapped_column(Integer)

    # SHA-256 of the FilingDocument body this chunk was derived from. When
    # the parent document's content_hash diverges, the chunks are stale and
    # the service regenerates them.
    source_content_hash: Mapped[str] = mapped_column(String(64), index=True)

    # Dense vector representation. Nullable because chunks land before they
    # are embedded — the embedding service populates this column in a
    # separate pass. ``embedding_model`` records which backend produced the
    # vector so the service can detect model changes and re-embed.
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIMENSION),
        nullable=True,
    )
    embedding_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    embedded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Generated tsvector for BM25-style retrieval. Maintained by Postgres,
    # backed by a GIN index from migration ``0007_retrieval_indexes``. The
    # Python type is loose because we never read the raw tsvector — only
    # match against it in SQL via ``@@`` and rank with ``ts_rank_cd``.
    text_tsv: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', text)", persisted=True),
        nullable=True,
    )

    document: Mapped[FilingDocument] = relationship(back_populates="chunks")

    def __repr__(self) -> str:
        return (
            f"FilingChunk(document_id={self.filing_document_id}, "
            f"section={self.section_label!r}, idx={self.chunk_index}, "
            f"chars={self.char_count})"
        )
