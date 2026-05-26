# 9. API serving layer

- **Status**: accepted
- **Date**: 2026-05-26

## Context

The agent graph (ADR 0007) and the eval harness (ADR 0008) are useful
from the CLI but unreachable from anything else. Phase 5 puts the
research DAG behind an HTTP surface so a hiring manager, a notebook,
or a future frontend can drive it without spawning a Python process
per question.

Three design points needed pinning down before any of that could ship:

1. Web framework choice.
2. Streaming model — node-level vs. token-level, SSE vs. WebSockets.
3. Lifespan management of the heavy singletons (compiled graph,
   embedder HTTP client, reranker model, DB connection pool).

## Decision

### Framework: FastAPI

FastAPI is already implied by ``pydantic-settings`` being in the deps
tree, and the project's stack centres on async SQLAlchemy + httpx, so
an ASGI framework was the only sensible choice. Within ASGI, FastAPI
brings:

- Pydantic-driven request validation that maps cleanly to the
  ``ResearchRequest`` shape (query, as-of date, top-k).
- Lifespan context for building the compiled graph once on startup
  and disposing the embedder / reranker / engine on shutdown — the
  same hooks ``scripts/research.py`` already uses, just bound to an
  ASGI lifecycle.
- A dependency-injection model the tests can override.

Starlette directly would have worked and shaved a few imports; the
Pydantic validation and OpenAPI generation are worth the extra
dependency for portfolio readability.

### Streaming: node-level Server-Sent Events

The endpoint streams one event per LangGraph super-step via
``sse-starlette``. Each event payload is the partial state update
returned by the node that just completed (``router``,
``fundamentals``, ``risk``, ``sentiment``, ``technical``,
``synthesizer``, ``critic``). A final synthetic ``done`` event signals
end-of-stream; if anything raises before completion, the adapter
emits an ``error`` event with the exception type and message, then
closes the stream cleanly.

**Why node-level, not token-level.** Token-level streaming requires
adding a ``stream()`` method to the ``LLMClient`` protocol (ADR 0006)
and threading async iterators through every node. The agent graph
already produces user-visible structure at the node level — the
router's intent, each specialist's findings, the synthesizer's thesis
— and a UI can render those incrementally without needing partial
tokens. Token streaming is a feature for the future, after a real
frontend exists; the bytes saved by not waiting for whole-node
output are not worth the protocol burden yet.

**Why SSE, not WebSockets.** This endpoint is one-way (server →
client). SSE is half the protocol surface, works through every
reverse proxy without sticky-session config, reconnects automatically
in browsers, and consumes cleanly from ``curl -N``. WebSockets are
the right choice when the client also sends mid-stream (cancellation,
tool-use callbacks); we don't need that yet.

**Stream mode.** LangGraph's ``astream(mode="updates")`` is what we
want: each yielded value is ``{node_name: partial_state}`` so the SSE
adapter has the node identity for free. ``"values"`` mode (full state
snapshots) would be heavier and would obscure which node produced a
field.

### Serialisation: ``_to_jsonable`` over dataclasses

``ResearchState`` is composed of frozen dataclasses (``Source``,
``Finding``, ``Thesis``, ``Critique``…). The SSE adapter walks the
update dict, converts dataclasses to plain dicts via
``dataclasses.asdict``, formats ``date`` / ``datetime`` as ISO 8601,
and dumps to JSON. Pydantic models are deliberately *not* introduced
for state — see ADR 0006 on why message dataclasses stay framework-
neutral. The wire shape lives in ``alphamind.api.streaming`` and is
the only place that knows about JSON encoding rules.

### Lifespan

Startup builds the singletons in this order: embedder → reranker →
``HybridSearch`` → retrieval-fn → LLM client → compiled graph. The
compiled graph lives on ``app.state.graph``. Shutdown calls
``dispose_embedder()``, ``dispose_reranker()``, ``dispose_engine()``.

The factory ``create_app(graph=None)`` accepts an optional pre-built
graph so tests can skip the real wire-up and pass a stub graph
constructed with ``ScriptedLLMClient`` and a fake retrieval-fn — the
same pattern ``tests/unit/agents/test_graph.py`` uses.

### Shared retrieval-fn

``scripts/research.py`` builds a ``RetrievalFn`` from ``HybridSearch``
inline. The API needs the same wiring. Both now import
``alphamind.agents.retrieval.build_filing_retrieval_fn`` so there is
one canonical adapter from the retrieval layer to the agent contract.

### Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| ``GET`` | ``/healthz`` | Liveness — returns ``{"status": "ok"}`` once the graph is built. Does *not* ping Postgres or Redis; a deeper readiness probe is out of scope for v1. |
| ``POST`` | ``/research`` | Streaming research. Body: ``ResearchRequest``. Response: ``text/event-stream`` with one event per node, terminating in ``done`` or ``error``. |

### What's not covered yet

- **Authentication.** No bearer tokens, no API keys, no rate limits.
  The portfolio demo is fronted by a reverse proxy or an IP allow-
  list if exposed publicly. Auth lands when there's a frontend to
  authenticate.
- **Token-level streaming.** Deferred until the ``LLMClient`` protocol
  grows a ``stream()`` method (Phase 5+) and there's a UI that can
  render partial tokens usefully.
- **Versioning.** No ``/v1`` prefix yet — premature for a single-
  endpoint surface that has no clients. Add when the second client
  ships.
- **Readiness vs. liveness split.** ``/healthz`` is liveness-only. A
  ``/readyz`` that pings Postgres, Redis, and the configured LLM
  backend would be useful in a deployed setup; deferred until
  deployment is in scope.
- **Cancellation.** Closing the SSE stream from the client side
  doesn't cancel the in-flight LangGraph run — the upstream LLM
  calls keep going to completion. Real cancellation needs
  ``asyncio.CancelledError`` propagation through every node; out of
  scope for v1.

## Consequences

- The agent graph is a process-wide singleton. One LLM client, one
  embedder, one reranker, one compiled graph per process. Horizontal
  scaling is "more processes," not "more sessions per process" —
  appropriate for the scale this project operates at.
- Tests can drive the full HTTP path without a database or a real
  LLM: ``create_app(graph=stub_graph)`` over ``httpx.AsyncClient``
  with an ``ASGITransport`` runs in-memory and asserts on parsed SSE
  events.
- The CLI (``scripts/research.py``) and the API speak the same agent
  graph, share the same retrieval-fn factory, and surface the same
  state shapes. There is one product, two front doors.
- Operationally, the server is one ``uvicorn alphamind.api:app``
  command behind any reverse proxy. ``make api`` ships the default
  invocation; a ``Dockerfile`` for the serving image is the natural
  follow-up when deployment becomes real.
