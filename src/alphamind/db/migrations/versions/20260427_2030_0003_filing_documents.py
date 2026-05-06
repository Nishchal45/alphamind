"""create filing_documents table

Revision ID: 0003_filing_documents
Revises: 0002_companies_filings
Create Date: 2026-04-27 20:30:00+00:00

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003_filing_documents"
down_revision: str | Sequence[str] | None = "0002_companies_filings"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "filing_documents",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("filing_id", sa.Integer(), nullable=False),
        sa.Column("storage_uri", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("byte_size", sa.BigInteger(), nullable=False),
        sa.Column("content_type", sa.String(length=128), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column(
            "fetched_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
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
            ["filing_id"],
            ["filings.id"],
            name=op.f("fk_filing_documents_filing_id_filings"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_filing_documents")),
        sa.UniqueConstraint("filing_id", name=op.f("uq_filing_documents_filing_id")),
    )
    op.create_index(
        op.f("ix_filing_documents_content_hash"),
        "filing_documents",
        ["content_hash"],
    )
    op.create_index(
        op.f("ix_filing_documents_filing_id"),
        "filing_documents",
        ["filing_id"],
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_filing_documents_filing_id"), table_name="filing_documents")
    op.drop_index(op.f("ix_filing_documents_content_hash"), table_name="filing_documents")
    op.drop_table("filing_documents")
