"""widen filing_chunks.embedding from vector(384) to vector(768)

Revision ID: 0006_widen_embedding_to_768
Revises: 0005_chunk_embeddings
Create Date: 2026-05-10 16:00:00+00:00

Picked to match Gemini's ``text-embedding-004`` output (768 dim). The
column type is replaced (drop + add) rather than ALTERed; vectors stored
under the old dimension are unrecoverable at the new size and the table
is small in early development.

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "0006_widen_embedding_to_768"
down_revision: str | Sequence[str] | None = "0005_chunk_embeddings"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

NEW_DIMENSION = 768
OLD_DIMENSION = 384


def upgrade() -> None:
    # NULL out the column before swapping types; pgvector cannot coerce a
    # 384-vector into a 768-vector. ``embedding_model`` is cleared too so
    # the embedding service re-encodes any orphaned rows.
    op.execute("UPDATE filing_chunks SET embedding = NULL, embedding_model = NULL")
    op.drop_column("filing_chunks", "embedding")
    op.add_column(
        "filing_chunks",
        sa.Column("embedding", Vector(NEW_DIMENSION), nullable=True),
    )


def downgrade() -> None:
    op.execute("UPDATE filing_chunks SET embedding = NULL, embedding_model = NULL")
    op.drop_column("filing_chunks", "embedding")
    op.add_column(
        "filing_chunks",
        sa.Column("embedding", Vector(OLD_DIMENSION), nullable=True),
    )
