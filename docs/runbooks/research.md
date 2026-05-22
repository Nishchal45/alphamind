# Runbook — `scripts/research.py`

The first end-to-end run of the Phase 3 agent team. Same retrieval +
LLM building blocks as `scripts/ask.py`, but the answer is produced
by a LangGraph DAG: router → fundamentals specialist → synthesizer →
critic. Output is a structured bull / bear thesis with chunk-level
citations and a critic pass over the result.

## What it does

```
question + as-of date  →
    router          (classifies intent, picks specialists)        →
    fundamentals    (HybridSearch + structured findings)          →
    synthesizer     (merges findings into bull / bear claims)     →
    critic          (flags unsupported claims and contradictions) →
    stdout: intent · thesis · critique · sources · token usage
```

The router currently records a broader intent than the graph can
act on — only the fundamentals specialist is wired today. Sentiment,
technical, and risk specialists land in follow-up PRs.

## Prerequisites

Same as `scripts/ask.py`:

- Phase 1 metadata ingest for at least one ticker.
- Phase 1 body ingest (`--with-bodies`).
- Phase 2 chunking run.
- Phase 2 embedding run (`embed_chunks_for_filing`). Unlike `ask.py`,
  this script uses `HybridSearch` (BM25 + dense + rerank), so chunks
  need embeddings.
- `.env` with valid `DATABASE_URL` / `REDIS_URL` / `SEC_USER_AGENT`.

For real answers (not echo stubs):

```env
LLM_BACKEND=anthropic
LLM_MODEL=claude-sonnet-4-5
ANTHROPIC_API_KEY=sk-ant-...
```

Optional, recommended:

```env
EMBEDDING_BACKEND=gemini     # vs deterministic
RERANKER_BACKEND=cross-encoder
```

## Common invocations

Bull / bear case for NVDA's China exposure, time-bounded to EOY 2024:

```bash
uv run python scripts/research.py \
  --query "What's the bull and bear case on NVDA's China revenue concentration?" \
  --as-of 2024-12-31
```

Wider question, more retrieval depth per specialist:

```bash
uv run python scripts/research.py \
  --query "How is AAPL talking about Services growth durability?" \
  --as-of 2025-03-31 \
  --top-k 12
```

## Reading the output

```
Intent
------
  primary=fundamentals
  specialists=fundamentals, risk
  rationale=Question is about revenue mix and concentration risk.

Summary
-------
[2-3 sentence framing of the bull/bear picture]

Bull case
---------
  • [claim]  [chunk_ids]
  ...

Bear case
---------
  • [claim]  [chunk_ids]
  ...

Critique
--------
  • [unsupported] [verbatim claim]
      [why]  [chunk_ids]
  ...     or "(no issues)"

Sources
-------
  [chunk_id] NVDA  10-K  2024-02-21  section='Item 1A...'  score=0.91
  ...

Usage
-----
  router        claude-sonnet-4-5  in=512  out=124
  fundamentals  claude-sonnet-4-5  in=4123 out=512
  synthesizer   claude-sonnet-4-5  in=1042 out=287
  critic        claude-sonnet-4-5  in=4501 out=145
  total         —  in=10178 out=1068
```

Citation IDs in the bull/bear/critique blocks are chunk IDs from the
`Sources` table — follow them back to verify what's actually supported.

## Failure modes

| Symptom | Likely cause | Action |
| --- | --- | --- |
| `(synthesizer did not produce a thesis)` | Retrieval returned zero sources or every finding's citations were invalid | Confirm chunks + embeddings exist for filings dated `<= --as-of`; widen the date if needed |
| `Intent → rationale=parse_failure` | The router's model produced unparseable JSON | Real backends recover next run; with `LLM_BACKEND=echo` this is expected |
| `(critic parse error: ...)` | The critic's model produced unparseable JSON | The thesis is still printed; treat it as uncritiqued and re-run if needed |
| Bull/bear claims with `[]` for citations | Synthesizer dropped invalid cites; bug if persistent | Inspect logs for `dropping uncited claim` lines |
| Token usage section is empty for a node | Node short-circuited (no findings, no thesis, no sources) | Usually correct behaviour — but check the upstream nodes' outputs |

## Operational notes

- `--as-of` is required, not optional. Same correctness invariant as
  `ask.py`: defaulting it would silently let lookahead bias creep in.
  The retrieval pipeline enforces this at three layers (ADR 0005), but
  the CLI making the parameter required is the first line of defence.
- The script uses `HybridSearch`, so chunks without embeddings won't
  surface on the dense branch. BM25 still works on them, but recall
  suffers.
- The critic is the only LLM-judgement layer in the citation contract.
  The specialist and synthesizer drop invalid citations mechanically;
  the critic catches semantic misreadings the structural checks
  miss. It can be wrong — treat its issues as warnings, not verdicts.
- Cost: a single research run with all four nodes typically lands
  somewhere in the 8-15k input / 1-2k output token range against the
  Anthropic backend. Watch the `Usage` block when iterating on prompts.
