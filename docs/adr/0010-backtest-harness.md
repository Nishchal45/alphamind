# 10. Backtest harness

- **Status**: accepted
- **Date**: 2026-05-26

## Context

The eval harness (ADR 0008) measures *citation honesty* — does the
system's output trace cleanly back to the source pool, and does the
critic catch the hallucinations that escape the structural checks.
That's a necessary check but it's silent on the harder question: when
the system expresses a directional view on a company, is that view
correlated with what subsequently happens to the stock.

Phase 6's second slice answers that question by replaying the agent
DAG against historical as-of dates and comparing the resulting view
to forward price action and to SPY as a benchmark.

Three design points needed pinning down before any of that could
ship:

1. How to turn a thesis (qualitative bull / bear claims with citations)
   into a directional signal.
2. How to source price data without compromising the time-horizon
   invariant ADR 0005 has spent five layers protecting.
3. What this backtest is *not* — the methodological caveats that make
   the difference between "research artefact" and "investment claim."

## Decision

### Signal extraction: mechanical, not LLM-rated

For each thesis the synthesizer produces, the harness computes

```
signal = (n_bull - n_bear) / (n_bull + n_bear)
```

clamped to `[-1, 1]`, where `n_bull` and `n_bear` are the number of
claims on each side. Empty-thesis cases (no claims either way) score
`0.0` and are excluded from the active universe for that period.

A configurable threshold (`signal_threshold`, default `0.20`) maps
the continuous score to a discrete position:

| Signal | Position |
| --- | --- |
| `signal >  +threshold` | long |
| `signal < -threshold` | short |
| otherwise | flat (no position) |

**Why mechanical and not an LLM-rated score.** Three reasons:

- **No extra leakage surface.** An LLM rater that reads the thesis
  could in principle be influenced by context outside the source
  pool. Counting claims has no such surface area.
- **Deterministic.** The same thesis produces the same signal forever.
  Backtest reruns are exactly reproducible.
- **No extra cost.** No additional LLM calls per case; backtests over
  hundreds of cases stay cheap.

The trade-off is signal granularity — a thesis with three confident
bull claims and one weak bear scores the same as one with three
hedged bulls and one definitive bear. That's an acceptable loss for
v1; a weighted variant (claim confidence × claim count) is a natural
follow-up.

### Universe: YAML-driven, mirrors the eval golden-set shape

Each backtest case is a `(case_id, ticker, query, as_of)` tuple plus
optional metadata. The runner walks the list, invokes
`build_research_graph` per case, extracts a signal, fetches forward
returns, and writes a per-case row to the report.

Same format choice as ADR 0008's golden set: YAML on disk,
hand-curated, validated eagerly at load time. One file
(`evals/backtest_universe.yaml`) ships as a seed — small enough to
fit on a screen, large enough to demonstrate the harness end-to-end.

### Horizon: forward window, configurable, default 90 days

Each case holds the position for `horizon_days` calendar days after
the as-of date (mapped to the closest trading day on both ends).
Default `90` — one quarter, long enough that filing-derived signal
has time to play out, short enough that iteration is fast.

Entry price: close on the first trading day **strictly after**
`as_of`. Exit price: close on the last trading day on or before
`as_of + horizon_days`. Adjusted close throughout — corporate-action
adjusted but not future-information-leaking in any way that matters
for relative returns.

### Price data: yfinance with disk cache

`yfinance` is the path of least resistance for a portfolio project.
It's free, the data quality is acceptable for relative-return
calculations on liquid US tickers, and the dependency surface is
contained.

The prices adapter caches raw OHLCV CSVs in `data/prices/` (gitignored,
keyed by `ticker_start_end.csv`). First run hits Yahoo; subsequent
runs are offline. Tests stub the adapter — no network calls under
`pytest`.

### Benchmark: SPY, same source

SPY for the benchmark, fetched the same way as the universe tickers
so dividend handling and trading-day calendar match exactly. Alpha
per case is `position_return - spy_return` over the same window.

### Lookahead defenses

The single most important correctness invariant in the project, given
fresh attention at this layer:

1. **Agent DAG.** Already enforces `as_of` at three layers (ADR 0005:
   per-branch retrieval WHERE clauses + pipeline-level hydration
   filter). The backtest does not bypass any of them — it goes through
   the same `build_research_graph` and the same retrieval adapter
   `scripts/research.py` uses.
2. **Signal extraction.** Operates on the thesis only. The thesis is
   a function of the agent run, which is a function of data dated
   on-or-before `as_of`. No path from future data to signal.
3. **Entry price.** Computed from the first trading day **strictly
   after** `as_of`. The as-of-day price is not used. This matters
   because the as-of-day's close is in principle visible to a human
   reading a filing dated that day, but for a clean backtest we want
   no possibility of acting on the as-of-day's market reaction to
   that filing.
4. **Exit price.** Computed from a date `horizon_days` calendar days
   after `as_of`. This is by construction in the future relative to
   the thesis and is the value we are predicting. No invariant
   violation; this is the dependent variable.
5. **SPY.** Same window, same source, same adjustment.

The `as_of`-required contract from ADR 0005 propagates all the way
through: the YAML loader requires it on every case, the runner
threads it into `ResearchState`, the price adapter takes it as an
explicit parameter, and the report records it in every row.

### Portfolio sizing: equal-weight long-short across active cases

Each case with `position in {long, short}` contributes equally to
the portfolio. Flat cases are excluded for that period. Aggregate
portfolio return for a period is the mean of per-case position
returns over the cases that took a position.

Continuous signal-weighted sizing (`weight = signal`) is a natural
extension; deferred so the v1 report is interpretable as "this is
what the thesis said in plain terms, here is what happened."

### Metrics

Per-case (in the report row):

- `signal` (raw, in `[-1, 1]`)
- `position` (`long` / `short` / `flat`)
- `position_return` (raw return over the hold window)
- `spy_return` (raw return over the same window)
- `alpha` (`position_return - spy_return`)
- `correct` (boolean: did position align with realised price direction)

Aggregate (in the summary):

- `n_cases` and `n_active` (cases that took a position)
- `hit_rate` (fraction of active cases where `correct`)
- `mean_alpha` (arithmetic mean across active cases)
- `cagr` (compound annual growth rate of the equal-weight portfolio)
- `sharpe` (annualised, rf = 0; small-sample, treat as descriptive)
- `max_drawdown` (worst peak-to-trough on the equity curve)

`sharpe` and `cagr` are computed even on small `n_cases` for
diagnostic value, but the report calls out their unreliability at
small sample size explicitly. None of these are the project's
headline claim — see "What this backtest is not" below.

### Report

Markdown to `docs/eval/backtest.md` with a committed equity-curve PNG
under `docs/eval/`. JSON sidecar to `evals/backtest_report.json` for
machine consumers. The markdown report opens with the methodological
caveats — not buried at the end — so anyone reading it forms an
honest mental model before reading any number.

## What this backtest is *not*

Per CLAUDE.md non-negotiable #3 (no financial-advice framing), every
number in the report comes with explicit caveats. The honest framing:

> This harness measures whether the agent system's expressed view is
> correlated with subsequent price action over a small sample of
> hand-picked cases. It does not measure tradable alpha, it does not
> account for transaction costs or slippage, the universe is not
> survivorship-adjusted, and the sample size is too small for the
> aggregate metrics to be statistically meaningful. It is a sanity
> check that the system produces directional views that aren't
> random; nothing more.

Concretely, what's deliberately not done:

- **No transaction costs / no slippage.** Real long-short portfolios
  pay both. The backtest is gross return.
- **No risk-adjusted position sizing.** Equal-weight ignores
  volatility differences. Kelly, vol-targeting, etc. are all out of
  scope.
- **No survivorship adjustment.** The universe is whatever tickers
  the user puts in the YAML. If those tickers are still listed
  today, the backtest is implicitly conditioned on survival.
- **No cross-sectional ranking.** Signals are independent per case;
  there is no global rebalancing across the universe at each date.
- **No out-of-sample / in-sample split.** The universe is one
  snapshot. Prompt tuning against backtest results would overfit.
- **No multi-period compounding within a case.** Each case is one
  entry, one exit. No mid-position rebalancing.
- **No alternative benchmarks.** SPY only.

## Consequences

- The harness reuses every component of the production research path
  — same graph, same retrieval, same `as_of` discipline. Drift in
  any of those automatically surfaces in the backtest output.
- The YAML universe is the single tunable input. Adding a case is a
  text edit, not a code change.
- Price data is cached on disk after first fetch. CI doesn't pay for
  network in the test suite; production backtest runs pay once per
  ticker per window and never again.
- The report calls its own caveats out at the top. Anyone reading
  it — interviewer, future-me, hiring manager — sees the
  methodological honesty before they see the numbers, which is the
  framing this project commits to.
