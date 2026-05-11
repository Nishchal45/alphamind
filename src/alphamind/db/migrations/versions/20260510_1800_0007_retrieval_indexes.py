"""add retrieval indexes for hybrid search

Revision ID: 0007_retrieval_indexes
Revises: 0006_widen_embedding_to_768
Create Date: 2026-05-10 18:00:00+00:00

Two physical indexes back :mod:`alphamind.retrieval`:

* a generated ``tsvector`` column with a GIN index for BM25-style keyword
  search via ``ts_rank_cd``;
* an HNSW index on ``embedding`` using ``vector_cosine_ops`` for the dense
  side of the hybrid search.

The ``english`` tsvector configuration is a reasonable default for SEC
filings; switching to a custom config later is a follow-up migration.
HNSW parameters are left at pgvector defaults (m=16, ef_construction=64)
which work well for corpora up to a few million vectors.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0007_retrieval_indexes"
down_revision: str | Sequence[str] | None = "0006_widen_embedding_to_768"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Generated tsvector column. STORED so reads do not recompute.
    op.execute(
        "ALTER TABLE filing_chunks "
        "ADD COLUMN text_tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english', text)) STORED"
    )
    op.execute("CREATE INDEX ix_filing_chunks_text_tsv ON filing_chunks USING GIN (text_tsv)")

    # HNSW on the embedding column for cosine-distance ANN.
    op.execute(
        "CREATE INDEX ix_filing_chunks_embedding_hnsw "
        "ON filing_chunks USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_filing_chunks_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_filing_chunks_text_tsv")
    op.execute("ALTER TABLE filing_chunks DROP COLUMN IF EXISTS text_tsv")
