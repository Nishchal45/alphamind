"""ORM model for a single chunk of a filing's primary document body.

Chunks are produced by :class:`alphamind.retrieval.chunking.ChunkingPipeline`
and become the unit of retrieval — both BM25 and dense ANN run against this
table. The denormalised ``filing_date`` column is the time-horizon filter
(see ADR 0002): we want ``WHERE filing_date <= :as_of`` to be cheap and
indexable without joining ``filings`` on every search.

The ``embedding`` column ships in migration ``0004_filing_chunks`` alongside
the rest of the table. It's nullable, so chunks can be persisted before
the embedder has run. The HNSW index over ``embedding`` is created in the
same migration; on an empty column the index is essentially free, and
shipping it now means the embedder service doesn't have to manage DDL.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, Date, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from alphamind.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from alphamind.models.filing import Filing

# Embedding dimension matches BAAI/bge-small-en-v1.5 (the default real
# embedder once Phase 3 wires that in). Changing this requires a migration
# and a re-embed pass.
EMBEDDING_DIM = 384


class FilingChunk(Base, TimestampMixin):
    """One retrievable unit of a filing's primary document body."""

    __tablename__ = "filing_chunks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    filing_id: Mapped[int] = mapped_column(
        ForeignKey("filings.id", ondelete="CASCADE"),
        index=True,
    )

    # Denormalised from the parent filing so retrieval can apply the
    # time-horizon predicate without a join — see ADR 0005.
    filing_date: Mapped[date] = mapped_column(Date, index=True)

    ordinal: Mapped[int] = mapped_column(Integer)
    section: Mapped[str | None] = mapped_column(String(128), index=True)

    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)

    # Postgres-maintained tsvector for BM25 / lexical search.
    # ``GENERATED ALWAYS`` keeps the column in sync with ``text`` automatically.
    text_tsv: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', text)", persisted=True),
    )

    # Dense embedding for semantic search. Nullable so chunks can be
    # persisted before the embedding pass runs.
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))

    filing: Mapped[Filing] = relationship(back_populates="chunks")

    __table_args__ = (
        Index(
            "ix_filing_chunks_filing_id_ordinal",
            "filing_id",
            "ordinal",
            unique=True,
        ),
    )

    def __repr__(self) -> str:
        section_repr = self.section or "—"
        preview = (self.text[:60] + "…") if len(self.text) > 60 else self.text
        return (
            f"FilingChunk(filing_id={self.filing_id}, ord={self.ordinal}, "
            f"section={section_repr!r}, text={preview!r})"
        )
