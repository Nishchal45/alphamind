# Runbook — `scripts/research.py`

The first agentic end-to-end demo of the project. Takes a research
question and runs it through a LangGraph DAG of specialised agents,
producing a structured thesis (answer + bull + bear) with cited
sources and a critic report flagging unsupported claims.

For the design rationale, see [ADR 0007](../adr/0007-agent-graph-design.md).

## What it does

```
question + as-of date
  → router (LLM): pick specialists
  → fundamentals specialist [+ others as they ship]:
       retrieve via HybridSearch (BM25 + pgvector + RRF + rerank)
       prompt LLM with domain rules
       emit cited claims
  → synthesizer (LLM): merge into bull / bear / answer with renumbered citations
  → critic (LLM): flag claims not supported by the source pool
  → stdout: router decision, per-specialist counts, thesis, sources, critic notes
```

All four specialists are registered with the graph:

- **fundamentals** — revenue, margins, segments, guidance. Augmentation
  pulls retrieval toward MD&A / financial-statement language.
- **sentiment** — qualitative tone, hedging, outlook language. Works
  from MD&A and 8-K narrative. Earnings-transcript ingestion isn't
  built yet, which is the strongest sentiment signal; the system
  prompt is explicit about that limitation.
- **risk** — Item 1A risk factors, legal proceedings, concentration
  risks. Augmentation targets risk-shaped passages.
- **technical** — no-data stub. The market-data adapter for OHLCV
  bars and corporate actions isn't built yet, and pretending to do
  technical analysis from 10-K boilerplate is actively misleading.
  This specialist is registered (so the router can route to it) but
  returns an empty report without calling the LLM. Drops the stub
  when the adapter lands.

## Prerequisites

- Same as `scripts/ask.py`: ingested filings, fetched bodies, chunks
  built. See [`ingest-edgar.md`](ingest-edgar.md) and
  [`retrieval.md`](retrieval.md).
- `.env` with `DATABASE_URL`, `REDIS_URL`, `SEC_USER_AGENT`.

To run with real models (not stubs):

```env
LLM_BACKEND=anthropic
LLM_MODEL=claude-sonnet-4-5
ANTHROPIC_API_KEY=sk-ant-...

EMBEDDING_BACKEND=gemini
GOOGLE_API_KEY=...

# Optional: real cross-encoder rerank. Costs the ~1GB torch install.
RERANKER_BACKEND=cross_encoder
```

With the defaults — `LLM_BACKEND=echo`, `EMBEDDING_BACKEND=deterministic`,
`RERANKER_BACKEND=deterministic` — the graph still runs end-to-end. Output
is meaningless but every router, specialist, synthesizer, and critic node
is exercised. Useful for smoke-testing the wiring without paying for API
calls.

## Common invocations

```bash
uv run python scripts/research.py \
  --query "What is NVDA's exposure to China revenue concentration?" \
  --as-of 2024-12-31
```

Bigger source pool per specialist:

```bash
uv run python scripts/research.py \
  --query "Bull and bear case for MSFT going into FY25" \
  --as-of 2024-09-30 \
  --top-k 12
```

## Reading the output

```
router: specialists=['fundamentals']  rationale='revenue + segment question'
specialists produced 1 report(s)
  - fundamentals: 5 claim(s), 8 citation(s)

Answer
------
NVDA's revenue is highly concentrated in the data-center segment [2][3]...

Bull case
---------
- Data-center revenue grew 86% YoY in the most recent quarter [3].
- ...

Bear case
---------
- Data-center concentration leaves NVDA exposed to a hyperscaler capex pullback [5].
- ...

Sources
-------
[1] NVDA  10-K  2024-02-21  chunk=421  section='Item 7. MD&A'
[2] NVDA  10-Q  2024-08-28  chunk=512  section='Item 7. MD&A'
...

Critic
------
notes: thesis is well-supported; no flagged claims.
no unsupported claims flagged.
```

## Failure modes

| Symptom | Likely cause | Action |
| --- | --- | --- |
| `specialists produced 0 report(s)` | Router skipped to synthesizer (rare; only happens when the router picks a specialist name not in the registry) | Check the router rationale in the printed output |
| `(technical specialist produced no findings: market-data adapter is not built yet ...)` | Expected — the technical specialist is a stub | Either ignore (the router likely shouldn't have picked it) or treat as a TODO for the market-data adapter |
| `(<specialist> specialist produced no findings: retrieval returned no candidates)` | No chunks matched even after query augmentation | Confirm chunks exist for the ticker; widen `--as-of` |
| `(fundamentals specialist produced no findings: could not parse specialist response: ...)` | LLM emitted non-JSON or wrong-shape JSON | Re-run; if the model is consistently bad at JSON, try a stronger `LLM_MODEL` |
| Critic flags everything as unsupported | LLM-judge over-strictness, or specialist hallucinated citations | Manually verify against the source list; consider tightening the specialist system prompt |
| `synthesizer fallback: ...` printed | Synthesizer's response didn't parse; thesis comes back with the failure in its answer field | Re-run; check the LLM_MODEL is configured to a model that handles structured output reliably |
| Router falls back to all specialists | Router LLM emitted bad JSON | Same as above; the all-specialists fallback is safe but more expensive |

## Operational notes

- `--as-of` is mandatory. Same reasoning as `ask.py` (and ADR 0005):
  the time-horizon filter is the project's single most important
  correctness invariant. There is no default.
- Each node has a fallback that produces a valid (if empty) state
  update. The graph does not raise out of an LLM error mid-flight.
  Backtests that fan thousands of questions through this graph
  shouldn't abort because one model response was malformed.
- The critic is a *check*, not a *gate*. An unsupported-claims report
  doesn't block the thesis. The thesis is printed first; the critic
  report follows so you can decide whether to trust it.
- Every printed answer is LLM synthesis. **Verify every citation
  against the printed source list before quoting it elsewhere.**
  LLM-generated citations have non-zero hallucination rate, which is
  precisely why the critic exists — but the critic itself is also an
  LLM, and is not perfect.
