# 3. SEC EDGAR ingestion design

- **Status**: accepted
- **Date**: 2026-04-22

## Context

EDGAR is a free, public source of structured filings data. It is also the slowest-moving dependency in AlphaMind: any correctness bug in ingestion will silently contaminate retrieval, agent reasoning, and backtest results.

Three properties the ingestion layer must guarantee:

1. **Fair-access compliance.** SEC rejects requests without an identifying `User-Agent` and rate-limits clients past ~10 requests per second. Exceeding the ceiling risks an IP ban.
2. **Idempotency.** Backfills, re-runs, and incremental updates must converge to the same schema state. Re-ingesting AAPL on Monday after a Sunday run must not duplicate rows.
3. **Resilience.** Transient 429 / 5xx responses are common during market hours and earnings seasons. A single blip must not abort a batch job.

## Decision

Build a single async adapter with three layers:

| Layer | Module | Responsibility |
| --- | --- | --- |
| Transport | `ingestion.edgar.client` | `User-Agent`, token-bucket rate limit (default 8 req/s), tenacity retries on 429/5xx and transport errors. |
| Parsing | `ingestion.edgar.schemas` | Pydantic models for submissions and ticker map; `iter_filings()` zips EDGAR's parallel arrays into record form. |
| Persistence | `ingestion.edgar.service` | Postgres `INSERT ... ON CONFLICT DO UPDATE` keyed on `cik` and `accession_number`, all under a single `session_scope()` transaction. |

The CLI entrypoint (`scripts/ingest_edgar.py`) wraps all three, catches per-item exceptions, and prints a terminal summary. CI and operators treat a non-zero exit as "at least one item failed; check logs."

### Why token-bucket, not a simple semaphore

A semaphore of size 10 allows 10 requests to begin simultaneously, which briefly exceeds the ceiling from SEC's perspective because all 10 land inside the same second. The token bucket amortises the budget across the window and is provably correct under `asyncio.gather`, which is exercised in `test_concurrent_requests_stay_within_rate`.

### Why `ON CONFLICT DO UPDATE`, not read-then-write

Read-then-write would require a lock or an `INSERT ... RETURNING` dance to avoid race conditions under concurrent ingestion. The single statement is atomic at the database, needs no application-side lock, and produces fewer round trips.

### Why we store metadata only (no document bodies) in this phase

Filing bodies range from a few kilobytes (an 8-K) to tens of megabytes (a 10-K with exhibits). Storing them inline in Postgres would bloat page caches and make schema migrations painful. The retrieval layer (Phase 2) will store chunked passages with stable identifiers back to EDGAR's canonical document URL, and the raw body will live in object storage.

## Consequences

- Ingestion can run on cron and in overnight batch jobs safely; the rate limit is a local property, not a coordination problem.
- Time-aware retrieval (preventing lookahead bias during backtests) is possible because every filing row carries `filing_date`, which is indexed and available as a `WHERE` predicate before any ANN search.
- Adding new sources (news, earnings transcripts) follows the same three-layer template: a transport module, typed schemas, an upsert service.
- If SEC ever tightens the rate ceiling or introduces authenticated access, the change is isolated to `client.EdgarClient.__init__`.
