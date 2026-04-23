"""ORM model for an individual SEC filing."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import Date, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from alphamind.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from alphamind.models.company import Company


class Filing(Base, TimestampMixin):
    """A single SEC filing — e.g. a 10-K, 10-Q, or 8-K.

    ``accession_number`` is the SEC's globally unique identifier and is used
    as the idempotency key during ingestion.
    """

    __tablename__ = "filings"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    company_id: Mapped[int] = mapped_column(
        ForeignKey("companies.id", ondelete="CASCADE"),
        index=True,
    )
    accession_number: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    form: Mapped[str] = mapped_column(String(16), index=True)
    filing_date: Mapped[date] = mapped_column(Date, index=True)
    report_date: Mapped[date | None] = mapped_column(Date)
    primary_document: Mapped[str] = mapped_column(String(512))
    primary_doc_description: Mapped[str | None] = mapped_column(String(256))

    company: Mapped[Company] = relationship(back_populates="filings")

    def __repr__(self) -> str:
        return (
            f"Filing(accession={self.accession_number!r}, "
            f"form={self.form!r}, filing_date={self.filing_date.isoformat()})"
        )
