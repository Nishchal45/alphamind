# 2. Use PostgreSQL with pgvector for documents and embeddings

- **Status**: accepted
- **Date**: 2026-04-05

## Context

AlphaMind needs to store three things:

1. Source documents (SEC filings, news, earnings transcripts) with their metadata — filer, filing date, form type.
2. Chunked passages from those documents, each with a stable pointer back to its source span.
3. Dense vector embeddings of those passages, queried by nearest-neighbour at request time.

Retrieval is hybrid — dense + BM25 — and needs to filter by metadata (filing date, ticker) *before* similarity search runs. Lookahead bias during backtests is the single biggest correctness risk in the project. A filing dated 2024-02-01 must never be retrievable when the analysis horizon is 2024-01-15. That constraint decides the storage choice.

## Options considered

| Option | Verdict |
| --- | --- |
| **PostgreSQL + pgvector** | Chosen. |
| **Dedicated vector DB (Qdrant / Weaviate / Pinecone)** | Rejected for this phase. |
| **BM25 only, no vectors** | Rejected — misses paraphrase matches that are routine in financial prose. |

## Decision

PostgreSQL 16 with the `pgvector` extension, as the single source of truth for documents, chunks, embeddings, and metadata. Hybrid retrieval composes `tsvector` lexical scoring, `pgvector` ANN, and SQL-level metadata filters in one query plan. A cross-encoder rerank happens in application code over the top-k candidates.

### Why not Qdrant

Qdrant is faster at pure ANN and the API is pleasant. But the moment you need to filter by `filing_date <= :horizon`, that predicate lives in a different system than the vector index. You either (a) push metadata into Qdrant's payload and re-implement relational joins as payload filters, or (b) do a two-step query across two systems and hope the candidate set you pulled back is large enough. (a) means duplicating state; (b) means losing the ability to push filters down before ANN. Neither is a trade I want to make on day one of a correctness-critical project.

When vector volume passes ~10M chunks and ANN latency becomes the bottleneck, the vector index can move to a dedicated engine without touching the document store. That's a problem for later.

### Why not BM25 only

Filings use synonym-heavy prose — "inventory write-down", "excess and obsolete reserve", "NRV adjustment" all mean roughly the same thing. Lexical-only retrieval misses the paraphrase. Dense retrieval catches it but ranks boilerplate too high. Hybrid gets both.

## Consequences

- One system (Postgres) to back up, monitor, and migrate.
- `WHERE filing_date <= :horizon` is a native SQL predicate on the same table as the vector index. Lookahead bias becomes a query-writing problem rather than a coordination problem.
- Alembic is the sole migration tool. The baseline migration enables `vector` so every environment boots pgvector-capable.
- Redis stays around for what it's good at: short-lived caches (embeddings, LLM responses, rate-limit counters) where durability is unnecessary and a cache miss is cheap.
