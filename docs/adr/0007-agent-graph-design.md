# 7. Agent graph design

- **Status**: accepted
- **Date**: 2026-05-19

## Context

ADR 0006 shipped the LLM provider abstraction. The retrieval layer
(ADR 0005) returns ranked, citation-grade chunks for a query. The
missing piece is the workflow that ties the two together: take a
question, pick the right domain to dig in, retrieve, reason, merge into
a thesis, and audit the result before it lands in front of the user.

The README pitches the agent layer as

```
router -> fundamentals, sentiment, technical, risk -> synthesizer -> critic
```

This ADR documents the design decisions that show up in the first
agent-graph PR — the skeleton plus the fundamentals specialist. The
other three specialists land in follow-ups and inherit the scaffolding.

## Decision

A LangGraph DAG with typed state, four node types, and a process-wide
factory.

### Why LangGraph

Three things matter for the workflow we want, and LangGraph hits all
three without us building scaffolding:

1. **Typed state with reducers.** Multiple specialists run in parallel
   and each one wants to append a report. LangGraph's annotated
   ``TypedDict`` lets us declare ``Annotated[list[SpecialistReport],
   merge_specialist_reports]`` and have the runtime call our reducer
   when parallel branches converge.
2. **Conditional fan-out.** The router decides which specialists run.
   A conditional edge returning a list of node names fans out only
   into those branches — we don't pay LLM cost on specialists the
   router didn't pick.
3. **Compiled graph object.** ``StateGraph.compile()`` returns a
   single async callable that drives the DAG. We don't have to
   hand-roll a topological scheduler.

We could write the same workflow in raw ``asyncio.gather`` and a few
``if`` statements. We'd save one dependency and pay for it in workflow
glue every time the graph shape changes.

### Why the state is a ``TypedDict``, not a Pydantic model

Same reasoning as :class:`alphamind.llm.base.Message` (ADR 0006):

- LangGraph's runtime expects partial-update semantics — a node
  returns ``{"thesis": ...}`` and the runtime merges. Pydantic models
  fight that by validating the whole instance on construction.
- The leaf records (``Citation``, ``Claim``, ``SpecialistReport``,
  ``Thesis``, ``CriticReport``) are frozen dataclasses. Validation
  happens at the *boundary* — the node that parses an LLM response —
  not on every read.

### Specialist scaffolding

Each specialist is a class subclassing :class:`SpecialistBase`. The
base class owns the three-step pipeline:

1. Augment the query with domain-specific terms, then retrieve via
   ``HybridSearch``.
2. Hydrate the retrieved ``chunk_id``s into ``Citation`` records via
   one Postgres query (with ``selectinload`` on filing and company so
   ticker / form lookups don't N+1).
3. Prompt the LLM with the domain system prompt and parse the JSON
   response into a :class:`SpecialistReport`.

Specialists differ in three places:

- ``name`` — the slot in :class:`SpecialistReport`.
- ``system_prompt`` — the role and rules of engagement.
- ``query_augmentation`` — terms appended before retrieval so each
  specialist pulls a domain-relevant slice of the corpus.

Everything else lives in the base class. When the next PR adds
sentiment / technical / risk, each one is ~30 lines.

#### Why query augmentation instead of section / form filters

The "obvious" alternative is to teach ``HybridSearch`` to filter by
``filing_chunks.section`` (Item 1A for risk, Item 7 for fundamentals,
etc.) and by ``filings.form`` (only 10-K / 10-Q for fundamentals, only
8-K for sentiment-of-recent-events). Two reasons we don't do that in
this PR:

- ``HybridSearch`` has a clean public API today. Growing it to take
  filter sets needs SQL changes in two retrieval branches plus the
  fusion layer, and it doesn't compose with the time-horizon predicate
  cleanly until we decide what "Item 7" means across 10-K and 10-Q
  forms.
- Query augmentation is doing meaningful work: BM25 picks up
  fundamentals terms in the lexical branch, and the dense encoder
  shifts the query vector toward semantically similar chunks. It's
  not a perfect substitute for hard filters, but it's a real
  differentiation signal that doesn't require schema changes.

Hard filters land in the follow-up that also adds the remaining
specialists, when we have enough cross-specialist usage to choose
filter shapes from data instead of guessing.

### Router: structured JSON, not tool use

The router returns ``{"specialists": [...], "rationale": "..."}``
parsed via the tolerant JSON extractor in
:mod:`alphamind.agents.json_utils`. Two reasons:

- Tool use isn't in the :class:`LLMClient` protocol yet (ADR 0006
  flagged this as deferred).
- The router's output shape is two fields. A full tool-use round-trip
  is over-engineering for that.

Fallback: if the router response can't be parsed, or the model picks
an empty / unknown set, the graph falls back to "run every available
specialist." Slower and more expensive, but correct.

### Synthesizer: renumber citations before the LLM sees them

Each specialist's citations are numbered ``[1]..[N]`` *local to that
specialist*. Two specialists can both say ``[1]`` and mean different
chunks. The synthesizer's first job is to merge those into one
deduplicated, renumbered pool and rewrite specialist citations into
the merged numbering before the LLM is asked to compose the thesis.

This keeps the contract from synthesizer to critic clean: by the time
the critic sees the thesis, every ``[N]`` in the prose refers to the
single canonical citation list on ``Thesis.citations``.

### Critic: LLM-judge with structured output, single pass

Two candidate designs:

1. **Mechanical citation-coverage check** — every sentence in
   synthesizer output must contain ``[N]``; flag any that don't.
2. **LLM judge** — the critic re-reads the thesis against the source
   pool and lists claims that aren't supported.

Mechanical coverage catches "no citation at all" but misses the
failure mode that actually matters: a citation that exists but
doesn't support the claim (the canonical hallucination). The LLM
judge gets that case. It also gets fewer false positives — bullets
that paraphrase a source correctly without exact-quote overlap.

Cost: one extra LLM call per question. Worth it.

Failure mode: the critic itself might emit malformed JSON. We
*don't* block the thesis on a critic failure — the report comes back
with ``unsupported=()`` and a ``notes`` field that surfaces the
failure. The critic is a check, not a gate.

### Fallback policy across the graph

Every node has a fallback path that produces a valid (if empty) state
update. We do not raise out of nodes for LLM / parsing failures. Two
reasons:

- A research question that finishes with "router fell back; one
  specialist returned no claims; synthesizer made a thin thesis;
  critic flagged it" is more useful than an exception.
- Backtests that fan thousands of questions through the graph
  shouldn't abort because one model response was malformed.

The narrow exception is *input* validation (empty query, missing
``as_of``) — those raise ``ValueError`` because they signal a bug in
the caller, not in the model.

## Consequences

- ``scripts/research.py`` is the first agentic end-to-end demo of the
  project: router → fundamentals → synthesizer → critic, every call
  going through the :class:`LLMClient` protocol. With
  ``LLM_BACKEND=anthropic`` and a key set, it runs against real
  models; with the defaults it runs entirely offline with stubbed
  components.
- ``scripts/ask.py`` keeps working unchanged. Two scripts, two
  workflows: ``ask.py`` is the flat BM25 + LLM smoke test; ``research.py``
  is the agentic pipeline. The boundary is explicit.
- The remaining three specialists are ~30-line subclasses of
  :class:`SpecialistBase`. The next PR can also lift query
  augmentation into section / form filters once we know what filter
  shapes the specialists actually want.
- Tool use, streaming, and cost aggregation are still on the
  follow-up list. The graph runs without them today; they fold in
  cleanly when the protocol grows.
