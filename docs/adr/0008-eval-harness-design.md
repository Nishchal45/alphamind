# 8. Eval harness design

- **Status**: accepted
- **Date**: 2026-05-22

## Context

Phase 3 added a four-node agent DAG (router → fundamentals + risk →
synthesizer → critic) with citation validation baked in at every
layer. The structural checks (specialist drops invalid cites,
synthesizer can't invent chunks) are deterministic; the
critic is the only LLM-judgement step in the citation contract.

That works fine when there are two specialists. As sentiment and
technical specialists land, prompt edits accumulate, and the corpus
grows, the question becomes: *is this change making things better
or worse?* Today the answer is "ask the LLM and see what comes out."
That's not good enough.

This ADR pins down the first slice of a measurement story.

## Decision

A small, in-repo eval harness with three properties:

1. **Reproducible**. Golden set is a YAML file checked into the
   repo; metrics are pure functions; the runner takes a compiled
   graph so tests exercise the same code path as the CLI.
2. **Cheap**. The harness produces no incremental dependencies that
   the agent layer doesn't already need, beyond `pyyaml`. Running
   against `LLM_BACKEND=echo` is free and exercises the plumbing.
3. **Scoped**. The harness *measures*; it does not gate. The CLI's
   exit code only reflects "did any case raise an exception" today.
   Threshold-based pass/fail can land later if it earns its keep.

### Metrics

Six metrics in this first slice, all returning a float in `[0, 1]`
or `None` (when the case didn't supply the hint the metric depends
on).

| Metric | Defends against | Hint required |
| --- | --- | --- |
| `citation_coverage` | Claims with no citation | — |
| `citation_validity` | Claims citing non-existent chunks | — |
| `hallucination_rate` | Critic-flagged unsupported claims | — |
| `contradiction_rate` | Critic-flagged contradictions | — |
| `topic_recall` | Synthesis dropping the actual subject | `expected_topics` |
| `chunk_recall` | Best evidence going un-cited | `required_chunk_ids` |

`citation_validity` should be ~1.0 by construction — the specialist
and synthesizer both drop invalid cites before findings leave them.
It's still on the dashboard as a regression alarm: if validity ever
slips below 1.0, the structural-check layer broke.

`hallucination_rate` and `contradiction_rate` are downstream of a
single LLM-judgement call (the critic), so they're noisier than the
other four. Treat short-term wobble in these two as expected;
long-term drift as a signal.

### Golden-set format

YAML, keyed by case `id`:

```yaml
cases:
  - id: nvda-china-2024-q4
    query: "..."
    as_of: 2024-12-31
    expected_topics: ["China", "export"]
    required_chunk_ids: [4123]
    notes: "..."
```

The `id` is the stable handle the report uses to identify a case.
Both hint fields are optional — a case with neither still gets four
metrics scored (every metric except topic / chunk recall).

`required_chunk_ids` is a coarse hint, not a strict expectation. Re-
chunking a filing renumbers chunks; treat this metric as wobbling
when the chunker changes, and refresh the IDs as part of that work.

### Why YAML and not JSON

Multi-line strings and comments. Golden cases are documents — the
notes field explains *why* this case is in the set — and YAML lets
the file double as a human-readable spec. The extra dep (`pyyaml`)
is tiny and already transitively present via langgraph.

### Why the runner doesn't build the graph

Tests want to run the harness against a stub-wired graph
(`SystemKeyedLLMClient` + fake retrieval) — that's the only way to
exercise the metrics end-to-end in CI without paying for API calls.
Production wants to run it against the real graph. Having the runner
accept a pre-built `CompiledStateGraph` means both call sites share
one implementation; only the wiring differs.

### Why per-case failure doesn't abort the suite

If case 3 of 50 raises an exception, the most useful thing to do is
run cases 4–50 and report which one broke. Aborting on the first
failure would mean every regression run loses signal on the cases
downstream of the first.

### Thresholds (added 2026-05-22)

The first iteration of this ADR shipped the harness as a thermometer
only. A follow-up adds threshold gating: `evals/thresholds.yaml`
pins one bound (`minimum` or `maximum`, never both) per metric, the
CLI takes `--thresholds`, and the exit code becomes:

| Condition | Exit code |
| --- | --- |
| At least one case raised during graph invocation | 1 |
| No case raised, but at least one threshold violated | 2 |
| Clean run | 0 |

The check is over the aggregate (mean across cases), not per-case.
Per-case spikes are interesting but noisier; the aggregate is what
CI gates on. A threshold whose metric has `n == 0` (no case supplied
the hint it depends on) is *skipped*, not failed.

Two-sided gates aren't supported. Every metric in the harness is
one-sided by intent — coverage / recall / validity want high
numbers, hallucination / contradiction rates want low — so the
loader rejects an entry that sets both `minimum` and `maximum`.

The shipped defaults are conservative on the noisier metrics (the
two critic-driven rates) and strict on the structural one
(`citation_validity` minimum=1.0). We'll tighten as the harness
gets more runs and the natural variance is visible.

## What's not covered yet

- **Historical backtest.** The README mentions an SPY backtest as
  part of Phase 6. That's a separate harness — it scores *trading
  decisions* over time, not *answer quality* on a fixed set of
  questions. Lands when the agent layer has a trade-decision output,
  which it doesn't yet.
- **Dashboards.** Reports are JSON on disk. A regression dashboard
  could plot trends; we don't have enough runs to make trends
  meaningful yet.
- **Cost-aware ranking.** `total_input_tokens` / `total_output_tokens`
  are on `CaseResult` but the aggregate report doesn't compare
  alternatives. When we have multiple model configurations to choose
  between, that comparison earns a metric of its own.

## Consequences

- Every future agent change has a single, mechanical place to ground-
  truth itself. Prompt edits, new specialists, fine-tuned models all
  flow through `make eval`.
- The citation contract isn't an honor system anymore: regressions
  show up in `citation_validity` and `chunk_recall` before they
  reach a human reader.
- The harness deliberately doesn't pretend to grade *correctness* —
  there's no oracle that knows what NVDA's bull case "should" be.
  What we measure is structural integrity, coverage, and what the
  critic catches. Treat the numbers accordingly.
