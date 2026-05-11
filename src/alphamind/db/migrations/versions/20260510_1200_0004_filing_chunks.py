"""create filing_chunks table

Revision ID: 0004_filing_chunks
Revises: 0003_filing_documents
Create Date: 2026-05-10 12:00:00+00:00

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0004_filing_chunks"
down_revision: str | Sequence[str] | None = "0003_filing_documents"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "filing_chunks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("filing_document_id", sa.Integer(), nullable=False),
        sa.Column("section_label", sa.String(length=64), nullable=False),
        sa.Column("section_title", sa.String(length=256), nullable=True),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("char_count", sa.Integer(), nullable=False),
        sa.Column("source_content_hash", sa.String(length=64), nullable=False),
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
            ["filing_document_id"],
            ["filing_documents.id"],
            name=op.f("fk_filing_chunks_filing_document_id_filing_documents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_filing_chunks")),
        sa.UniqueConstraint(
            "filing_document_id",
            "chunk_index",
            name=op.f("uq_filing_chunks_filing_document_id"),
        ),
    )
    op.create_index(
        op.f("ix_filing_chunks_filing_document_id"),
        "filing_chunks",
        ["filing_document_id"],
    )
    op.create_index(
        op.f("ix_filing_chunks_section_label"),
        "filing_chunks",
        ["section_label"],
    )
    op.create_index(
        op.f("ix_filing_chunks_source_content_hash"),
        "filing_chunks",
        ["source_content_hash"],
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_filing_chunks_source_content_hash"),
        table_name="filing_chunks",
    )
    op.drop_index(op.f("ix_filing_chunks_section_label"), table_name="filing_chunks")
    op.drop_index(
        op.f("ix_filing_chunks_filing_document_id"),
        table_name="filing_chunks",
    )
    op.drop_table("filing_chunks")
