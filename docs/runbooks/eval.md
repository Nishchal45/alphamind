# Runbook — `scripts/eval.py` / `make eval`

The first slice of the Phase 6 eval harness. Runs the research DAG
across a YAML golden set and writes a JSON report with six metrics
per case plus an aggregate. The harness *measures*; it does not gate.

See [ADR 0008](../adr/0008-eval-harness-design.md) for the design.

## What it does

```
golden set (YAML)  →
    for each EvalCase:
        invoke build_research_graph(...) with (query, as_of, top_k)
        score the finished ResearchState:
          citation_coverage     · citation_validity
          hallucination_rate    · contradiction_rate
          topic_recall          · chunk_recall
    aggregate → EvalReport →  stdout summary + JSON report on disk
```

## Prerequisites

To produce *real* scores:

```env
LLM_BACKEND=anthropic
LLM_MODEL=claude-sonnet-4-5
ANTHROPIC_API_KEY=sk-ant-...
EMBEDDING_BACKEND=gemini       # vs deterministic
RERANKER_BACKEND=cross-encoder
```

Plus a corpus: ingested filings (`scripts/ingest_edgar.py --with-bodies`)
that have been chunked and embedded.

With the default `LLM_BACKEND=echo` every case will return an empty
thesis, so the report only proves the plumbing works.

## Common invocations

The default golden set + report path:

```bash
make eval
```

Or, with explicit paths:

```bash
uv run python scripts/eval.py \
  --golden-set evals/golden_set.yaml \
  --out evals/report.json
```

Different golden set (e.g., a per-PR regression set):

```bash
uv run python scripts/eval.py \
  --golden-set evals/regression_2026q2.yaml \
  --out evals/regression_2026q2.json
```

## Reading the output

stdout:

```
Aggregate metrics
-----------------
  citation_coverage      mean=1.000  min=1.000  max=1.000  n=4
  citation_validity      mean=1.000  min=1.000  max=1.000  n=4
  hallucination_rate     mean=0.083  min=0.000  max=0.333  n=4
  contradiction_rate     mean=0.000  min=0.000  max=0.000  n=4
  topic_recall           mean=0.625  min=0.333  max=1.000  n=4
  chunk_recall           (not measured: 0 cases supplied this hint)

  cases run:    4
  cases failed: 0
```

The JSON report has the same `aggregate` plus a `per_case` list with
the full :class:`CaseResult` for each case — token usage, claim
counts, the failure string if any.

`(not measured: 0 cases supplied this hint)` means no case in the
golden set carried that field. It's an honest signal: we didn't grade
the question, so don't read it as a 0%.

## Writing a golden case

Minimal:

```yaml
cases:
  - id: my-question
    query: "What is X about Y?"
    as_of: 2024-12-31
```

Full:

```yaml
cases:
  - id: my-question
    query: "..."
    as_of: 2024-12-31
    expected_topics:
      - "China"
      - "gross margin"
    required_chunk_ids: [4123, 4218]
    notes: |
      Why this case is in the set.
```

Reminders:

- `id` must be unique. The loader rejects duplicates.
- `as_of` is `YYYY-MM-DD`. The time-horizon invariant still applies
  to every retrieval call the case makes.
- `required_chunk_ids` is the most fragile field — re-chunking a
  filing renumbers chunks. Refresh these IDs whenever the chunker
  changes.

## Failure modes

| Symptom | Likely cause | Action |
| --- | --- | --- |
| `GoldenSetError: cases[N]: ...` | Malformed YAML or missing required field | Check the loader error message; fix the case. |
| `cases failed: N` with N > 0 | A case's graph invocation raised | Inspect `per_case[i].failure` in the JSON report for the exception text. |
| `citation_validity < 1.0` in any case | The structural-check layer regressed — specialist or synthesizer is letting invalid cites through | This is a real bug. The specialist (`alphamind.agents.specialists._base._coerce_findings`) and the synthesizer (`alphamind.agents.synthesizer._coerce_claims`) are the two places that filter. |
| `hallucination_rate` spikes after a prompt edit | Synthesizer is producing more claims the critic can't support, OR the critic became stricter | Compare the critic prompts side by side; check whether the specialist findings actually back the synthesizer's reading. |
| `topic_recall = 0.0` for many cases | Synthesizer is dropping the subject of the question | Often a sign that retrieval isn't returning the relevant sections. Check the per-case `n_sources` and the chunk-id citations in the per-case report. |
| `(not measured: 0 cases supplied this hint)` | None of the cases set `expected_topics` or `required_chunk_ids` | Expected when the golden set is light on hints. Add fields when you have ground truth to assert. |

## Operational notes

- The harness uses `HybridSearch` (BM25 + dense + rerank); chunks
  without embeddings won't surface on the dense branch. Re-run
  `embed_chunks_for_filing` if recall looks suspiciously bad.
- Per-case failures don't abort the suite. The first one breaking is
  the most-common case in practice; running the rest gives you more
  signal.
- The exit code is 0 if every case ran without exception, 1
  otherwise. Metric values do not gate the exit code today (see
  ADR 0008 for why).
- Cost: one full eval run against the default golden set + Anthropic
  Sonnet is roughly 40-60k input / 4-8k output tokens. The token
  totals are on each `CaseResult` so you can budget per-case.
