"""create filing_chunks table with tsvector and pgvector embedding

Revision ID: 0004_filing_chunks
Revises: 0003_filing_documents
Create Date: 2026-05-04 11:00:00+00:00

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import TSVECTOR

# revision identifiers, used by Alembic.
revision: str = "0004_filing_chunks"
down_revision: str | Sequence[str] | None = "0003_filing_documents"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIM = 384


def upgrade() -> None:
    op.create_table(
        "filing_chunks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("filing_id", sa.Integer(), nullable=False),
        sa.Column("filing_date", sa.Date(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("section", sa.String(length=128), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column(
            "text_tsv",
            TSVECTOR(),
            sa.Computed("to_tsvector('english', text)", persisted=True),
            nullable=False,
        ),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True),
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
            name=op.f("fk_filing_chunks_filing_id_filings"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_filing_chunks")),
    )

    op.create_index(
        op.f("ix_filing_chunks_filing_id"),
        "filing_chunks",
        ["filing_id"],
    )
    op.create_index(
        op.f("ix_filing_chunks_filing_date"),
        "filing_chunks",
        ["filing_date"],
    )
    op.create_index(
        op.f("ix_filing_chunks_section"),
        "filing_chunks",
        ["section"],
    )
    op.create_index(
        "ix_filing_chunks_filing_id_ordinal",
        "filing_chunks",
        ["filing_id", "ordinal"],
        unique=True,
    )

    # GIN over the generated tsvector for BM25 / lexical retrieval.
    op.create_index(
        "ix_filing_chunks_text_tsv",
        "filing_chunks",
        ["text_tsv"],
        postgresql_using="gin",
    )

    # HNSW over embeddings for dense ANN. Cosine distance to match the
    # default of every modern sentence-transformer.
    op.execute(
        """
        CREATE INDEX ix_filing_chunks_embedding_hnsw
            ON filing_chunks
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_filing_chunks_embedding_hnsw;")
    op.drop_index("ix_filing_chunks_text_tsv", table_name="filing_chunks")
    op.drop_index("ix_filing_chunks_filing_id_ordinal", table_name="filing_chunks")
    op.drop_index(op.f("ix_filing_chunks_section"), table_name="filing_chunks")
    op.drop_index(op.f("ix_filing_chunks_filing_date"), table_name="filing_chunks")
    op.drop_index(op.f("ix_filing_chunks_filing_id"), table_name="filing_chunks")
    op.drop_table("filing_chunks")
