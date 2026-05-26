# Runbook — backtest harness

Runs the production research DAG against a YAML universe of
historical `(ticker, query, as_of)` cases and compares the resulting
positions to forward price action plus an SPY benchmark.

**This is a sanity check, not a measurement of tradable alpha.**
The caveats live at the top of every generated report; the same
discipline applies here. Design in
[ADR 0010](../adr/0010-backtest-harness.md).

## What it does

```
universe.yaml  →  for each case:
                  graph.ainvoke({query, as_of, top_k})  →  thesis  →
                  signal = (n_bull - n_bear) / (n_bull + n_bear)  →
                  long/short/flat per signal_threshold  →
                  entry price (first trading day > as_of)  →
                  exit price (last trading day <= as_of + horizon_days)  →
                  alpha vs SPY over the same window
                  →  per-case row + aggregate summary
                  →  markdown report + JSON sidecar + equity-curve PNG
```

## Prerequisites

The same ingestion path as `scripts/research.py`:

- Phase 1 metadata ingest run for every ticker in the universe.
- Phase 1 body ingest (`--with-bodies`).
- Phase 2 chunking + embedding (`chunk_filings_for_cik`,
  `embed_chunks_for_filing`).
- `.env` with `DATABASE_URL`, `REDIS_URL`, `SEC_USER_AGENT`.

To get real (non-echo) theses:

```env
LLM_BACKEND=anthropic
LLM_MODEL=claude-sonnet-4-5
ANTHROPIC_API_KEY=sk-ant-...
EMBEDDING_BACKEND=gemini
GOOGLE_API_KEY=...
RERANKER_BACKEND=cross_encoder
```

The default backends (deterministic embedder, deterministic reranker,
echo LLM) make the run cheap but produce no real signal — useful for
smoke-testing the wiring, not for measuring anything.

## Running the harness

```bash
LLM_BACKEND=anthropic ANTHROPIC_API_KEY=sk-ant-... make backtest
```

That target runs:

```bash
uv run python scripts/backtest.py \
  --universe evals/backtest_universe.yaml \
  --report-md docs/eval/backtest.md \
  --report-json evals/backtest_report.json \
  --chart docs/eval/backtest_equity.png
```

First run hits Yahoo Finance for each ticker + SPY; subsequent runs
read from the on-disk price cache under `data/prices/` (gitignored).

## Authoring a case

Edit `evals/backtest_universe.yaml`. Each case requires four fields:

```yaml
- case_id: nvda-2024-q1
  ticker: NVDA
  query: What is the bull / bear case on NVDA after the data-center inflection?
  as_of: 2024-03-31
```

Run-level config (all optional, defaults shown):

```yaml
horizon_days: 90
signal_threshold: 0.20
benchmark_ticker: SPY
top_k: 8
```

The loader validates strictly: missing fields, bad dates, duplicate
`case_id`s, non-positive horizons all raise at load time.

## Reading the report

`docs/eval/backtest.md` opens with the methodological caveats, then:

- **Configuration** — the run-level config that produced this report.
- **Aggregate metrics** — `n_active` / `hit_rate` / `mean_alpha` /
  descriptive `cagr`, `sharpe`, `max_drawdown`.
- **Per-case results** — one row per case with signal, return, alpha,
  and correctness.

`evals/backtest_report.json` is the machine-readable sidecar with the
same data.

`docs/eval/backtest_equity.png` is the equity curve — a cumulative-
product chart of position-aware returns ordered chronologically by
`as_of`. The dashed line at `1.0` is notional. ADR 0010 documents
that this is a portfolio *narrative*, not a time-aware accounting.

## Failure modes

| Symptom | Likely cause | Action |
| --- | --- | --- |
| Case row shows `error: graph: ...` | The agent DAG raised — usually a missing chunk for the ticker / as-of, or an LLM-provider transient | Check that filings are ingested + chunked for that ticker, and that `filing_date <= as_of` for at least one of them |
| Case row shows `error: prices: ...` | yfinance returned nothing for the window | Ticker may be delisted or symbol-changed; widen `horizon_days` or fix the ticker |
| `n_active == 0` | Every thesis came out flat | Likely an `LLM_BACKEND=echo` run (echo client never produces real claims), or the synthesizer is degrading to parse failures — check the per-case JSON sidecar |
| `cagr` / `sharpe` are `n/a` | Fewer than two active cases | Add more cases, or accept that span-based annualisation needs ≥2 observations |
| Chart isn't generated | No active case in the universe | Same root cause as `n_active == 0` |

## Operational notes

- **Time-horizon discipline.** Per ADR 0010, the entry price is the
  first trading day **strictly after** `as_of`. The as-of-day price
  is never used. The exit price is the last trading day on or before
  `as_of + horizon_days`. SPY is fetched on the same calendar so
  dividend handling and trading-day boundaries match.
- **Price cache.** First run takes time (one yfinance call per
  ticker per window). Subsequent runs are pure disk reads. Wipe the
  cache by deleting `data/prices/`.
- **What this isn't.** Per CLAUDE.md non-negotiable #3 and ADR 0010:
  no transaction costs, no slippage, no risk-adjusted sizing, no
  survivorship adjustment, no out-of-sample split, no multi-period
  rebalancing within a case. **Nothing this harness reports is
  financial advice.**
