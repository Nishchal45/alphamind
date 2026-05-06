# Runbook — `scripts/ask.py`

The first end-to-end demo of the project. Combines BM25 search over
already-chunked filings with a real LLM call to produce a cited answer
to a research question.

## What it does

```
question + as-of date  →  BM25 search over filing_chunks  →
                          top-k chunks formatted as numbered sources  →
                          system prompt + cited-source rules  →
                          LLM (Anthropic or echo stub)  →
                          stdout: answer + source list + cost line
```

## Prerequisites

- Phase 1 metadata ingest run for at least one ticker.
- Phase 1 body ingest run (`--with-bodies`).
- Phase 2 chunking run for the same filings (`chunk_filings_for_cik`).
- `.env` with valid `DATABASE_URL` / `REDIS_URL` / `SEC_USER_AGENT`.

To get real (non-echo) answers:

```env
LLM_BACKEND=anthropic
LLM_MODEL=claude-sonnet-4-5
ANTHROPIC_API_KEY=sk-ant-...
```

Default is `LLM_BACKEND=echo` — the script runs without an API key, but
output will be the echo stub repeating your question back.

## Common invocations

Ask about NVDA's most recent China-revenue commentary, time-bounded to
end of 2024:

```bash
uv run python scripts/ask.py \
  --query "What is NVDA saying about China revenue concentration?" \
  --as-of 2024-12-31
```

Wider question across whatever you've ingested, more sources:

```bash
uv run python scripts/ask.py \
  --query "What inventory write-down language has appeared this year?" \
  --as-of 2025-04-30 \
  --top-k 12
```

## Reading the output

```
[short answer paragraph]

[longer reasoning section that walks through specific sources]

Sources
-------
[1] NVDA  10-K  2024-02-21  accession=0001045810-24-000029  section='Item 1A. Risk Factors'
[2] NVDA  10-Q  2024-08-28  accession=0001045810-24-000200  section='Item 7. MD&A'
...

[claude-sonnet-4-5  in=4123  out=412  stop=end_turn]
```

The bracketed last line is the cost-tracking footer: model, input
tokens, output tokens, stop reason. Useful for keeping a running sense
of spend.

## Failure modes

| Symptom | Likely cause | Action |
| --- | --- | --- |
| `No chunks matched.` | No filings chunked yet, or the as-of date predates them all | Run `chunk_filings_for_cik` first; widen `--as-of` |
| `LLM call failed: anthropic api error: ...` | Bad API key, exhausted quota, or transient outage | Check `ANTHROPIC_API_KEY`; the wrapper already retries 4× on transient errors |
| Output is `echo: <your question>` | `LLM_BACKEND=echo` (the default) | Set `LLM_BACKEND=anthropic` and supply `ANTHROPIC_API_KEY` |
| Citations don't match the sources | LLM hallucinated a number; this is a known failure mode of every LLM-citation system | The critic agent (next phase) is the systematic fix; for now, manually verify each citation against the printed source list |

## Operational notes

- The `--as-of` date is required, not optional. Defaulting it to today
  would silently let lookahead bias creep into historical questions —
  the project's single most important correctness invariant.
- BM25 retrieval is used (not hybrid). Dense retrieval requires real
  embeddings, which are still on the deterministic stub. Once the real
  embedder lands, switching to `HybridSearch` is a one-line change in
  this script.
- Every printed answer is a synthesis of LLM output. **Verify every
  citation against the source list before quoting it elsewhere.**
  LLM-generated citations have non-zero hallucination rate.
