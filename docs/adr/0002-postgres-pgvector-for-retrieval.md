# 2. Use PostgreSQL with pgvector for documents and embeddings

- **Status**: accepted
- **Date**: 2026-04-22

## Context

AlphaMind needs a durable store for three kinds of data:

1. Source documents (SEC filings, news articles, earnings transcripts) with metadata including filer, filing date, and document type.
2. Chunked passages derived from those documents, each with a stable identifier back to its source span.
3. Dense vector embeddings of those passages, queried by nearest neighbour at request time.

The retrieval layer is hybrid — semantic (dense) plus lexical (BM25) — and requires joins between chunks, their embeddings, and the source-document metadata (filing date, ticker) so that time-aware filters can be applied before any similarity search runs.

We evaluated three storage options:

| Option | Pros | Cons |
| --- | --- | --- |
| **PostgreSQL + pgvector** | Single engine for documents, metadata, and vectors. Transactional. Rich SQL for filtering before ANN. Mature tooling (Alembic, SQLAlchemy). Free and self-hostable. | ANN recall at very large scale requires tuning (HNSW/IVFFlat). Not as fast at extreme vector counts as dedicated engines. |
| **Dedicated vector DB (Qdrant / Weaviate / Pinecone)** | Best-in-class ANN latency and recall. Hybrid search primitives built in. | Second system to run, back up, and secure. Metadata joins require pulling data back through the app. Extra network hop on every query. Pinecone adds cost and vendor lock-in. |
| **Full-text only (BM25, no vectors)** | Simplest; fits natively in Postgres. | Misses semantic matches that are critical for financial-prose retrieval. Rejected — we need hybrid retrieval. |

## Decision

Use PostgreSQL 16 with the `pgvector` extension as the single source of truth for source documents, chunks, and embeddings. Hybrid retrieval is implemented by composing SQL-driven filters, `tsvector` lexical scoring, and `pgvector` ANN in one query plan, with a cross-encoder rerank executed in application code over the top-k candidates.

## Consequences

- One operational surface (Postgres) is backed up, monitored, and migrated.
- Time-aware retrieval is expressible as a `WHERE filing_date <= :horizon` predicate on the same table as the vector index, which directly prevents lookahead bias in backtests.
- Alembic is the sole migration tool; the baseline migration (`0001_baseline`) enables the `vector` extension so every environment boots pgvector-capable.
- If ANN throughput becomes a bottleneck past ~10 M chunks, we can move just the vector index to a dedicated engine without touching the document store. That is a later-stage problem.
- Redis is retained for short-lived caches (embedding results, LLM responses, rate limits) where Postgres durability is unnecessary and a cache miss is cheap.
