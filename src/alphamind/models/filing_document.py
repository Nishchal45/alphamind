"""ORM model for the persisted body of a SEC filing's primary document.

A :class:`FilingDocument` row records *where* the bytes for a filing's primary
document live (in the storage backend) and *what* they are (size, MIME type,
content hash). The bytes themselves never live in Postgres — see ADR 0004.

The relationship to :class:`Filing` is one-to-one. EDGAR amendments produce
new accession numbers (and therefore new ``Filing`` rows), so a single
``Filing`` is never associated with more than one body.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from alphamind.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from alphamind.models.filing import Filing
    from alphamind.models.filing_chunk import FilingChunk


class FilingDocument(Base, TimestampMixin):
    """Pointer + metadata for the primary document body of a single filing."""

    __tablename__ = "filing_documents"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    filing_id: Mapped[int] = mapped_column(
        ForeignKey("filings.id", ondelete="CASCADE"),
        unique=True,
        index=True,
    )

    # Where the bytes live in the storage backend (e.g. "file:///path" for
    # local, "s3://bucket/key" for S3 once that backend exists).
    storage_uri: Mapped[str] = mapped_column(Text)

    # Lower-case SHA-256 hex digest of the body. Used both for idempotent
    # writes (skip refetch when nothing has changed) and as the content-
    # addressable storage key.
    content_hash: Mapped[str] = mapped_column(String(64), index=True)

    byte_size: Mapped[int] = mapped_column(BigInteger)
    content_type: Mapped[str] = mapped_column(String(128))

    # The EDGAR URL we fetched from. Kept for auditability — the URL is
    # canonical and stable, and lets a human reproduce the fetch.
    source_url: Mapped[str] = mapped_column(Text)

    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    filing: Mapped[Filing] = relationship(back_populates="document")
    chunks: Mapped[list[FilingChunk]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return (
            f"FilingDocument(filing_id={self.filing_id}, "
            f"hash={self.content_hash[:12]!r}, bytes={self.byte_size})"
        )
