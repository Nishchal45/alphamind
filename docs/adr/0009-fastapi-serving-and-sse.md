# 9. FastAPI serving + Server-Sent Events

- **Status**: accepted
- **Date**: 2026-05-23

## Context

ADR 0007 shipped the agent graph behind a CLI (``scripts/research.py``).
ADR 0008 documents the in-repo eval harness. The roadmap promises a
serving layer — async streaming endpoints, a Redis cache, model
routing for cost. This ADR records the first cut: the FastAPI app and
how it streams agent-graph progress to clients. Cache and cost
routing are out of scope for this slice.

Two questions to settle:

1. How does the API expose graph progress without blocking until the
   whole run finishes? The agent team typically takes 10-30 seconds
   end-to-end. A request that returns one JSON blob at the end is bad
   UX, especially with a frontend in front of it.
2. The :class:`alphamind.llm.base.LLMClient` protocol doesn't carry a
   streaming method yet. What does "streaming" mean for this slice?

## Decision

A FastAPI app via ``create_app()`` factory, one streaming endpoint,
two probes.

### Endpoints

| Method | Path        | Purpose |
| ------ | ----------- | ------- |
| POST   | ``/research`` | Run the agent graph, stream progress as SSE. |
| GET    | ``/healthz``  | Liveness — no I/O, always ``200 {status: ok}``. |
| GET    | ``/readyz``   | Readiness — ``200`` when Postgres answers ``SELECT 1``, else ``503``. |

### Streaming model: SSE over LangGraph node updates, not tokens

LangGraph's ``CompiledStateGraph.astream(stream_mode="updates")``
yields ``{node_name: state_update}`` after each node completes. We
forward those as Server-Sent Events:

| LangGraph node | SSE event | Payload schema |
| -------------- | --------- | -------------- |
| ``router`` | ``router-intent`` | ``{primary, specialists, rationale}`` |
| ``fundamentals`` / ``sentiment`` / ``risk`` / ``technical`` | ``specialist-findings`` | ``{specialist, sources[], findings[], usage[]}`` |
| ``synthesizer`` | ``thesis`` | ``{summary, bull_case[], bear_case[]}`` |
| ``critic`` | ``critique`` | ``{issues[], parse_error}`` |
| (graph end) | ``done`` | ``{}`` |
| (graph raised) | ``error`` | ``{message}`` |

The synthesizer thesis arrives as one ``thesis`` event — not
token-by-token. Real token streaming requires growing ``LLMClient``
with a streaming method, threading the iterator through the
synthesizer node, and surfacing the deltas as a new event type. Both
land in a follow-up; the protocol grows when it's the cheapest path,
not preemptively.

### Why SSE instead of WebSockets

The data flow is one-way (server → client). WebSockets would work but
cost an upgrade handshake and a bidirectional framing layer we don't
need. SSE rides on HTTP, plays well with proxies, and JavaScript's
built-in ``EventSource`` handles the client side for free.

The trade-off: SSE doesn't support binary frames and ``EventSource``
only does GET requests (so frontends doing ``POST /research`` parse
the stream manually). Neither matters for a research run that takes
~30 seconds at most.

### Why a factory + lifespan, not a module-level app

``create_app()`` builds a fresh app per call. Two reasons:

- **Test isolation.** Each test gets its own app with its own
  dependency overrides; no global state leaks across tests. The
  in-repo eval harness already proves the dependency-injected graph
  is the right contract for tests; the API reuses it.
- **Explicit entry point.** ``uvicorn --factory
  alphamind.api.app:create_app`` documents the contract instead of
  relying on a bare ``app = FastAPI()`` somewhere in the module.

The lifespan owns three things: building the compiled research graph
once (so the LLM client + embedder factories' singleton caching has a
host that survives across requests), wiring ``HybridSearch`` behind
the agent layer's ``RetrievalFn`` contract, and disposing the engine
+ embedder HTTP clients on shutdown.

### Pydantic schemas separate from agent dataclasses

The leaf records on :class:`ResearchState` (``Source``, ``Finding``,
``RouterIntent``, ``Thesis``, ``ThesisClaim``, ``CritiqueIssue``,
``Usage``) are frozen ``slots=True`` dataclasses. The HTTP surface
uses Pydantic models that mirror them.

Why not share types?

- **Different audiences.** The dataclasses optimise for immutability
  and ergonomic Python (pydantic's validation overhead is wasted on
  internal transport). The HTTP schemas optimise for OpenAPI doc
  generation and validation at the boundary.
- **Avoids leaking internal shape.** Future refactors that change how
  the graph carries state don't break clients. The conversion lives
  in one place (:mod:`alphamind.api.schemas`) — small surface, easy
  to keep stable.

The cost is small: each leaf record gets a one-line ``from_dataclass``
helper. The duplication is intentional separation.

### Per-specialist event grouping

The agent layer's state accumulators (``sources``, ``findings``,
``usage``) are concatenated across nodes via ``operator.add``. That's
right for the *final* state, but on the stream we emit one
``specialist-findings`` event *per specialist node* — keyed by the
LangGraph node name — so clients don't have to correlate three
separate streams to know which specialist contributed what. The
``specialist`` field on the payload is the LangGraph node name; the
``sources`` / ``findings`` / ``usage`` inside are scoped to that
node's contribution.

### Error handling

Three layers, narrowest first:

1. **Pydantic validation** rejects bad requests with ``422``. Empty
   queries, missing ``as_of``, ``top_k`` out of range.
2. **Per-node failures** are handled inside the graph (every node has
   a fallback path that produces a valid state update). Clients see
   normal-looking events with empty / fallback fields and the run
   completes.
3. **Graph raising mid-stream** is caught by the SSE adapter, which
   emits a final ``error`` event and closes the stream. The HTTP
   status was already ``200`` by the time bytes started flowing, so
   the ``error`` event is the only failure signal once streaming
   begins.

## Consequences

- ``scripts/research.py`` and the API both call ``build_research_graph``
  with the same arguments. The CLI is the developer-loop entry point;
  the API is the production entry point. Behaviour parity comes for
  free.
- A new module ``alphamind.api.retrieval.build_hybrid_retrieval``
  wires ``HybridSearch`` into the agent layer's ``RetrievalFn``
  contract. ``scripts/research.py`` keeps its private copy; folding
  the two together is a follow-up if they ever diverge.
- Token streaming, Redis caching, cost routing, and auth are all
  deferred. None of them block running the API end-to-end against
  ingested data; each folds in cleanly when it becomes the most
  valuable next slice.
- The technical specialist's no-data stub is wired through: a client
  that asks a technical-flavoured question gets an empty
  ``specialist-findings`` event for ``technical`` (no findings, no
  sources, no usage). Honest, not a bug.
