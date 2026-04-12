"""create companies and filings tables

Revision ID: 0002_companies_filings
Revises: 0001_baseline
Create Date: 2026-04-22 13:00:00+00:00

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002_companies_filings"
down_revision: str | Sequence[str] | None = "0001_baseline"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "companies",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("cik", sa.String(length=10), nullable=False),
        sa.Column("ticker", sa.String(length=16), nullable=True),
        sa.Column("name", sa.String(length=512), nullable=False),
        sa.Column("sic", sa.String(length=8), nullable=True),
        sa.Column("sic_description", sa.String(length=256), nullable=True),
        sa.Column("exchange", sa.String(length=32), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_companies")),
        sa.UniqueConstraint("cik", name=op.f("uq_companies_cik")),
    )
    op.create_index(op.f("ix_companies_cik"), "companies", ["cik"], unique=False)
    op.create_index(op.f("ix_companies_ticker"), "companies", ["ticker"], unique=False)

    op.create_table(
        "filings",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("accession_number", sa.String(length=32), nullable=False),
        sa.Column("form", sa.String(length=16), nullable=False),
        sa.Column("filing_date", sa.Date(), nullable=False),
        sa.Column("report_date", sa.Date(), nullable=True),
        sa.Column("primary_document", sa.String(length=512), nullable=False),
        sa.Column("primary_doc_description", sa.String(length=256), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["company_id"],
            ["companies.id"],
            name=op.f("fk_filings_company_id_companies"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_filings")),
        sa.UniqueConstraint("accession_number", name=op.f("uq_filings_accession_number")),
    )
    op.create_index(op.f("ix_filings_accession_number"), "filings", ["accession_number"])
    op.create_index(op.f("ix_filings_company_id"), "filings", ["company_id"])
    op.create_index(op.f("ix_filings_filing_date"), "filings", ["filing_date"])
    op.create_index(op.f("ix_filings_form"), "filings", ["form"])


def downgrade() -> None:
    op.drop_index(op.f("ix_filings_form"), table_name="filings")
    op.drop_index(op.f("ix_filings_filing_date"), table_name="filings")
    op.drop_index(op.f("ix_filings_company_id"), table_name="filings")
    op.drop_index(op.f("ix_filings_accession_number"), table_name="filings")
    op.drop_table("filings")

    op.drop_index(op.f("ix_companies_ticker"), table_name="companies")
    op.drop_index(op.f("ix_companies_cik"), table_name="companies")
    op.drop_table("companies")
