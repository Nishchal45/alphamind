# Runbook — Retrieval pipeline

## What this covers

Running the chunker, the embedder, and the hybrid search pipeline against
the data ingested by the EDGAR runbook. Assumes Phase 1 + the body
ingestion (`--with-bodies`) have already populated `filings` and
`filing_documents`.

## Prerequisites

- Local services up: `make compose-up`
- Migrations applied through `0004_filing_chunks`: `make migrate`
- At least one filing has a body row in `filing_documents`. Confirm:
  `psql -c 'select count(*) from filing_documents;'` in the Postgres
  container.

## Chunking a single filing

```python
from alphamind.db.session import session_scope
from alphamind.retrieval.chunking.service import chunk_filing
from alphamind.storage.factory import get_storage

storage = get_storage()
async with session_scope() as session:
    result = await chunk_filing(
        storage=storage,
        session=session,
        filing_id=42,
    )
print(result)
```

The `ChunkingResult` reports how many chunks were written and how many
existed before (replaced inside one transaction). If the filing has no
stored body, `skipped_reason="no_body_stored"` is returned without an
exception — batch jobs should treat that as a normal outcome.

## Chunking every filing for a CIK

```python
from alphamind.retrieval.chunking.service import chunk_filings_for_cik
from alphamind.storage.factory import get_storage

results = await chunk_filings_for_cik(
    get_storage(),
    cik="0000320193",
    limit=10,  # most recent 10 filings
)
```

## Embedding chunks

```python
from alphamind.retrieval.embeddings.factory import get_embedder
from alphamind.retrieval.embeddings.service import embed_chunks_for_filing
from alphamind.db.session import session_scope

embedder = get_embedder()
async with session_scope() as session:
    result = await embed_chunks_for_filing(
        embedder=embedder,
        session=session,
        filing_id=42,
    )
print(result)
```

`EMBEDDING_BACKEND` defaults to `deterministic` — a hash-seeded RNG
embedder useful only for testing the pipeline. Real semantic search
requires switching to a sentence-transformer backend (Phase 3 work);
a re-embed pass with `force=True` will overwrite existing vectors when
that lands.

## Searching

```python
from datetime import date
from alphamind.db.session import session_scope
from alphamind.retrieval.embeddings.factory import get_embedder
from alphamind.retrieval.search import HybridSearch
from alphamind.retrieval.search.rerank import DeterministicReranker

search = HybridSearch(
    embedder=get_embedder(),
    reranker=DeterministicReranker(),
)

async with session_scope() as session:
    hits = await search.search(
        session,
        query="risk factors related to China revenue concentration",
        as_of=date(2024, 12, 31),  # NEVER default this
        top_k=10,
    )

for hit in hits:
    print(f"{hit.filing_date}  {hit.section}  {hit.text[:120]}")
```

### The as-of date is required, not optional

There is no default for `as_of`. A backtest at horizon 2023-06-01 that
accidentally retrieves chunks from 2024 silently produces alpha that
doesn't exist. Forcing the caller to pass the horizon makes the contract
explicit and is the project's single most important correctness
invariant — see ADR 0005 and ADR 0002.

## Failure modes

| Symptom | Likely cause | Action |
| --- | --- | --- |
| `pgvector extension is not installed` from healthcheck | Migrations not applied through 0004 | `make migrate` |
| `no_body_stored` for every filing | Body ingest hasn't run | `uv run python scripts/ingest_edgar.py --ticker AAPL --with-bodies` |
| `ChunkingResult(chunks_written=0)` for filings that have bodies | HTML parser failed silently — non-HTML body, or unsupported encoding | Log the offending `accession_number` and inspect the body manually under `STORAGE_LOCAL_PATH` |
| Empty search results when chunks exist | Query terms not in the corpus, or `as_of` predates every filing | Widen `as_of`; check `select count(*) from filing_chunks where filing_date <= :as_of` |
| Slow dense search | HNSW index hasn't warmed up after migration | Run a few warm-up queries; first ANN scan after server start is always slowest |

## Operational notes

- Re-chunking a filing replaces its chunk set inside one transaction.
  Readers never see a half-chunked filing.
- `embed_chunks_for_filing` is idempotent: chunks with non-null
  embeddings are skipped unless `force=True`.
- Time-horizon filtering is enforced at three layers (schema,
  per-branch query, pipeline-level hydration). If a refactor breaks one,
  the others still hold the invariant.
