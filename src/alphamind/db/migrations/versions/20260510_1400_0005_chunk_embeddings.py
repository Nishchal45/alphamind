"""add embedding columns to filing_chunks

Revision ID: 0005_chunk_embeddings
Revises: 0004_filing_chunks
Create Date: 2026-05-10 14:00:00+00:00

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "0005_chunk_embeddings"
down_revision: str | Sequence[str] | None = "0004_filing_chunks"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Must match ``EMBEDDING_DIMENSION`` in alphamind.models.filing_chunk and
# ``embedder_dimension`` in alphamind.config. Changing this requires a
# follow-up migration that ALTERs the column type.
EMBEDDING_DIMENSION = 384


def upgrade() -> None:
    op.add_column(
        "filing_chunks",
        sa.Column("embedding", Vector(EMBEDDING_DIMENSION), nullable=True),
    )
    op.add_column(
        "filing_chunks",
        sa.Column("embedding_model", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "filing_chunks",
        sa.Column("embedded_at", sa.DateTime(timezone=True), nullable=True),
    )
    # Partial index on (embedding_model) over rows that have been embedded.
    # Lets the embedding service cheaply find the "needs (re-)embedding" set
    # for a given model name without scanning the whole table.
    op.create_index(
        op.f("ix_filing_chunks_embedding_model"),
        "filing_chunks",
        ["embedding_model"],
        postgresql_where=sa.text("embedding_model IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_filing_chunks_embedding_model"), table_name="filing_chunks")
    op.drop_column("filing_chunks", "embedded_at")
    op.drop_column("filing_chunks", "embedding_model")
    op.drop_column("filing_chunks", "embedding")
