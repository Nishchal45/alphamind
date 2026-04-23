# 3. SEC EDGAR ingestion design

- **Status**: accepted
- **Date**: 2026-04-19

## Context

EDGAR is free, structured, and the slowest-moving dependency in the project. It's also the one place where a silent correctness bug will contaminate retrieval, agent output, and every backtest downstream — so ingestion has to get a few things exactly right on the first try.

Three constraints the ingestion layer has to meet:

1. **Fair-access compliance.** SEC rejects requests without an identifying `User-Agent` and rate-limits past ~10 requests per second. Going over the ceiling risks an IP ban, which would be annoying to unwind.
2. **Idempotency.** Backfills, re-runs, and incremental pulls must all converge to the same schema state. Re-running AAPL on Monday after a Sunday run can't create duplicates.
3. **Resilience.** Transient 429s and 5xxs are routine — more so during market hours and earnings season. A single blip can't abort a batch.

## Decision

One async adapter, three layers, each with a single job:

| Layer | Module | Responsibility |
| --- | --- | --- |
| Transport | `ingestion.edgar.client` | `User-Agent`, token-bucket rate limit (default 8 req/s, under the ceiling), tenacity retries on 429/5xx and transport errors. |
| Parsing | `ingestion.edgar.schemas` | Pydantic models for submissions and ticker map. `iter_filings()` zips EDGAR's parallel arrays into record form. |
| Persistence | `ingestion.edgar.service` | Postgres `INSERT ... ON CONFLICT DO UPDATE` keyed on `cik` and `accession_number`, all under one `session_scope()` transaction. |

The CLI (`scripts/ingest_edgar.py`) wraps the three layers, catches per-item exceptions so one bad ticker doesn't kill the batch, and prints a summary. A non-zero exit means "at least one item failed — check logs."

### Token bucket, not semaphore

A semaphore of size 10 lets 10 requests start simultaneously, which from SEC's side is 10 requests in a single second — over the ceiling. A token bucket amortises the budget across the window. Under `asyncio.gather` it's provably correct and there's a test (`test_concurrent_requests_stay_within_rate`) that exercises exactly that scenario.

### `ON CONFLICT DO UPDATE`, not read-then-write

Read-then-write needs an explicit lock or an `INSERT ... RETURNING` dance to avoid races under concurrent ingestion. One atomic upsert statement is fewer round trips and needs no application-side coordination. It's also shorter code.

### Why metadata only (for now)

Filing bodies range from a few kilobytes for an 8-K to tens of megabytes for a 10-K with exhibits. Inlining that in Postgres would bloat the page cache and make every schema migration slower. Phase 2 will store chunked passages with stable pointers back to EDGAR's canonical URL, and the raw body will live in object storage.

## Consequences

- Ingestion is safe to run on cron. The rate limit is a local property, not a distributed coordination problem.
- Time-aware retrieval works because every filing row carries `filing_date` (indexed). That's available as a `WHERE` predicate before any ANN search runs.
- New sources (news, transcripts) follow the same three-layer template: transport, typed schemas, upsert service. Nothing about EDGAR's design is special to EDGAR.
- If SEC tightens the ceiling or moves to authenticated access, the change is isolated to `client.EdgarClient.__init__`.
