# Runbook — Ingest SEC EDGAR filings

## What this runs

Resolves one or more tickers (or CIKs) via the SEC ticker map, fetches their
submissions metadata, and upserts company and filing rows into Postgres.
Safe to re-run: every write uses `INSERT ... ON CONFLICT DO UPDATE`.

## Prerequisites

- Local services are up: `make compose-up`
- Migrations are applied: `make migrate`
- `.env` contains a valid `SEC_USER_AGENT` identifying you by name and email
  (SEC policy — unidentified requests are rejected with `403`).
- Dependencies are installed: `make install`

Verify the environment with `make healthcheck`.

## Common invocations

Ingest the three large-cap tech names' most recent periodic filings:

```bash
uv run python scripts/ingest_edgar.py --ticker AAPL MSFT NVDA
```

Ingest by CIK and only 10-K / 10-Q (default form filter):

```bash
uv run python scripts/ingest_edgar.py --cik 320193 789019
```

Backfill the 50 most recent filings of all types for one ticker:

```bash
uv run python scripts/ingest_edgar.py --ticker AAPL --forms ALL --limit 50
```

## Reading the output

The script prints a terminal summary:

```
CIK          TICKER     SEEN  WRITTEN  NAME
0000320193   AAPL         12        5  Apple Inc.
0000789019   MSFT         14        6  Microsoft Corporation
```

- `SEEN` — filings returned by EDGAR for this company in the recent-filings window.
- `WRITTEN` — filings passed through the form / limit filters and upserted.

Exit code is `0` iff every requested ticker/CIK ran cleanly. A non-zero exit
means at least one item failed; sibling items are still attempted and their
writes are committed.

## Failure modes and what to do

| Symptom | Likely cause | Action |
| --- | --- | --- |
| `403 Forbidden` immediately | Missing or generic `SEC_USER_AGENT` | Set a real name and contact email in `.env`. |
| Repeated `429 Too Many Requests` | Another process is hitting SEC from the same IP | Lower `rate` in `EdgarClient`, or serialise jobs. |
| `LookupError: ticker not found` | Ticker not in EDGAR map (e.g. a recent IPO) | Fall back to `--cik` with the CIK from SEC's company search. |
| `pgvector extension is not installed` from healthcheck | Migrations not applied | Run `make migrate`. |
| Hangs on startup | Postgres container not ready | `make compose-logs` and wait for `database system is ready`. |

## Operational notes

- The script is idempotent; re-running on the same data is a no-op.
- A single invocation opens one HTTP client and one database transaction per
  CIK. Transactions are short — no long-running locks.
- Upstream scheduling: this is safe to run on cron. Start with daily, then
  move to after-hours only once backfill is complete.
