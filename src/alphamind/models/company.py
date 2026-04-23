"""ORM model for a publicly traded company known to the SEC."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from alphamind.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from alphamind.models.filing import Filing


class Company(Base, TimestampMixin):
    """A SEC-registered company identified by its zero-padded CIK.

    CIK is the canonical identifier. Ticker can change (e.g. through corporate
    actions) and companies can list under multiple tickers, so ticker is
    stored for convenience but never used as a unique key.
    """

    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    cik: Mapped[str] = mapped_column(String(10), unique=True, index=True)
    ticker: Mapped[str | None] = mapped_column(String(16), index=True)
    name: Mapped[str] = mapped_column(String(512))
    sic: Mapped[str | None] = mapped_column(String(8))
    sic_description: Mapped[str | None] = mapped_column(String(256))
    exchange: Mapped[str | None] = mapped_column(String(32))

    filings: Mapped[list[Filing]] = relationship(
        back_populates="company",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return f"Company(cik={self.cik!r}, ticker={self.ticker!r}, name={self.name!r})"
