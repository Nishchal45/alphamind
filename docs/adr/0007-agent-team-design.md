# 7. Agent team design

- **Status**: accepted
- **Date**: 2026-05-22

## Context

Phase 3 turns the retrieval + LLM building blocks from Phases 1 and 2
into the thing AlphaMind is actually for: an agentic answer to a
research question, every claim cited, every claim checked.

The architecture sketched in the README calls for a small team of
specialised agents: a router, four specialists (fundamentals,
sentiment, technical, risk), a synthesizer, and a critic. Three design
points needed pinning down before any of that could ship:

1. How to wire the DAG.
2. How wide to make the first slice.
3. How to keep citations honest end-to-end.

## Decision

### Orchestrator: LangGraph

The DAG is built and compiled with [LangGraph][langgraph]. Each node
is an async callable returning a partial state update; LangGraph
merges updates into a single :class:`ResearchState` `TypedDict` via
per-field reducers (`operator.add` for the accumulator lists).

LangGraph was chosen over a hand-rolled `asyncio.gather` orchestrator
because of what's coming, not because the current DAG is complex
enough to need it:

- **Checkpointing** when long-running runs need to survive a process
  restart (Phase 5).
- **Tool calling** so specialists can issue retrieval queries
  iteratively rather than once at the start.
- **Streaming** so the FastAPI serving layer can emit intermediate
  agent output to the client.

Adopting LangGraph now means we don't pay a rewrite later.

### Scope of the first slice: vertical, not horizontal

This PR ships:

- the router,
- one specialist (fundamentals),
- the synthesizer,
- the critic,
- and the CLI entry-point `scripts/research.py`.

It does **not** ship the sentiment, technical, or risk specialists.
They land in follow-up PRs.

This trade-off — vertical depth over horizontal breadth — keeps the
review surface small and lets every layer of the citation contract
get exercised end-to-end before we add more producers of findings.

### Citation invariant

A claim that doesn't trace back to a real chunk is the only failure
mode the project actively defends against (everything else is just
quality). Three checks run at three layers:

1. **Source-pool check at the specialist.** The fundamentals node
   accepts only `cited_chunk_ids` that appear in the source pool it
   showed the model. Anything else is dropped before the finding
   leaves the node. Cheap, mechanical, catches the most common
   hallucination.
2. **Finding-set check at the synthesizer.** The synthesizer can only
   cite chunk ids that one of its input findings already cited.
   Inventing a new chunk to support a bull-case claim is structurally
   impossible.
3. **Semantic check at the critic.** The critic reads the thesis
   against the source pool and flags claims that pass the structural
   checks but don't actually mean what the cited chunks say. This is
   the only LLM-judgement step; it can be wrong but it's the only
   layer that catches genuine misreadings.

The first two are deterministic. The third is the soft layer and is
the one we'll evaluate hardest in Phase 6.

### Time-horizon enforcement is unchanged

`as_of` flows through `ResearchState` and is passed verbatim to the
retrieval entry-point. The pipeline's three-layer enforcement of the
time-horizon predicate (BM25, dense, and pipeline-level `as_of`
filter — see ADR 0005) is the line that holds. The agent layer never
caches across runs and never reorders chunks by anything but score,
so it can't smuggle post-horizon information into a thesis.

## What's not covered yet

- **Other specialists.** Sentiment / technical / risk specialists
  land separately. The router already records the broader intent so
  the synthesizer can flag coverage gaps in the meantime.
- **Fan-out.** Today there's exactly one specialist, so the router →
  specialist edge is a conditional that routes to a singleton. When
  more specialists land, the conditional becomes a fan-out and the
  graph runs them in parallel.
- **Tool-calling retrieval.** Specialists ask for evidence once at
  the start. A future iteration lets them iterate — read findings,
  request follow-up retrieval — which fits LangGraph's loop
  primitives cleanly.
- **Streaming.** `client.complete` is one-shot. Streaming lands
  alongside the FastAPI serving layer in Phase 5.
- **Tracing.** LangGraph has a tracing integration we deliberately
  haven't wired yet. The token-usage accumulator on state is the
  minimum the CLI needs.

## Consequences

- The graph is the dependency boundary the rest of the project
  programs against. Adding a specialist is a node + a routing-table
  entry, not an orchestration rewrite.
- Tests stay fast and deterministic: the DAG accepts an `LLMClient`
  and a retrieval function, so unit tests inject scripted stubs and
  exercise the full topology without a database.
- The contract on citations is mechanical at two of three layers,
  which makes the surface the critic has to police much smaller and
  the failure modes much easier to reason about.

[langgraph]: https://langchain-ai.github.io/langgraph/
