# 8. FastAPI serving + Server-Sent Events

- **Status**: accepted
- **Date**: 2026-05-19

## Context

ADR 0007 shipped the agent graph behind a CLI (`scripts/research.py`).
The roadmap promises a serving layer — async streaming endpoints, a
Redis cache, model routing for cost. This ADR records the first cut:
the FastAPI app and how it streams agent-graph progress to clients.
Cache and cost routing are out of scope for this slice.

Two specific design questions to nail down:

1. How does the API expose graph progress without blocking until the
   whole run finishes? The agent team typically takes 10–30 seconds
   end-to-end. A request that returns one JSON blob at the end is a
   bad UX, especially once a frontend is in front of it.
2. The :class:`alphamind.llm.base.LLMClient` protocol doesn't support
   streaming completions yet (ADR 0006 deferred it explicitly). What
   does "streaming" mean for this slice?

## Decision

A FastAPI app with one streaming endpoint and two probes.

### Endpoints

| Method | Path        | Purpose |
| ------ | ----------- | ------- |
| POST   | `/research` | Run the agent graph, stream progress as SSE. |
| GET    | `/healthz`  | Liveness — no I/O, always `200 {status: ok}`. |
| GET    | `/readyz`   | Readiness — `200` when Postgres responds to `SELECT 1`, else `503`. |

### Streaming model: SSE over node-completion events, not tokens

LangGraph's `astream(stream_mode="updates")` yields
`{node_name: state_update}` after each node completes. We forward
those as Server-Sent Events:

| LangGraph node | SSE event | Payload schema |
| -------------- | --------- | -------------- |
| `router` | `router-decision` | `{specialists, rationale}` |
| `fundamentals` / `sentiment` / `risk` / `technical` | `specialist-report` | `{specialist, summary, claims[], citations[]}` |
| `synthesizer` | `thesis` | `{answer, bull[], bear[], citations[]}` |
| `critic` | `critic-report` | `{unsupported[], notes, ok}` |
| (graph end) | `done` | `{}` (sentinel) |
| (graph raised) | `error` | `{message}` |

The synthesizer's answer arrives as one `thesis` event — not
token-by-token. Real token streaming requires growing `LLMClient`
with a streaming method and threading the token iterator through the
synthesizer node. Both happen in a follow-up; the protocol grows when
it's the cheapest path, not preemptively.

### Why SSE instead of WebSockets

The data flow is one-way from server to client. WebSockets would
work but cost an upgrade handshake and a bidirectional framing layer
we don't need. SSE rides on top of HTTP, plays well with proxies and
load balancers, and JavaScript's built-in `EventSource` does the
client side for free.

The trade-off: SSE doesn't support binary frames and reconnect
semantics are weaker than WebSockets'. Neither matters for streaming
JSON events from a research run that takes ~30 seconds at most.

### Why a factory + lifespan, not a module-level app

`create_app()` builds a fresh app per call. Two reasons:

- **Test isolation.** Each test gets its own app with its own
  dependency overrides; no global state leaks across tests.
- **Explicit entry point.** `uvicorn --factory alphamind.api.app:create_app`
  documents the contract instead of relying on a bare `app =
  FastAPI()` somewhere in the module.

The lifespan owns three things: building the `ResearchGraph` once
(so the LLM client + embedder factories' singleton caching has a
host that survives across requests), and disposing the engine +
embedder HTTP clients on shutdown.

### Pydantic schemas live separate from the agent dataclasses

The leaf records on `AgentState` (`Citation`, `Claim`,
`SpecialistReport`, `Thesis`, `CriticReport`) are frozen dataclasses.
The HTTP surface uses Pydantic models that mirror them.

Why not share types? Two reasons:

- **Different audiences.** The dataclasses optimise for immutability
  and ergonomic Python (pydantic's validation overhead is wasted on
  internal transport). The HTTP schemas optimise for OpenAPI doc
  generation and JSON validation at the boundary.
- **Avoids leaking internal shape.** Future refactors that change
  how the graph carries state internally don't break clients. The
  conversion lives in one place (`alphamind.api.schemas`) and is
  easy to keep stable.

The cost: small. Each leaf record gets a ~10-line `from_dataclass`
helper. The duplication is intentional separation, not duplication
that the type system could eliminate.

### Error handling

Three layers, narrowest first:

1. **Pydantic validation** rejects bad requests with `422`. This
   covers empty queries, missing `as_of`, `top_k` out of range.
2. **Node failures** — handled inside the graph (every node has a
   fallback path that produces a valid empty update). Clients see a
   normal-looking event with empty fields and an explanatory
   summary; the run completes.
3. **Graph raising mid-stream** — caught by the SSE adapter, which
   emits a final `error` event and closes the stream. The HTTP
   status was already `200` by the time bytes started flowing, so
   there's no nice way to signal failure via status code; the
   `error` event is the signal.

## Consequences

- `scripts/research.py` and the API both call the same graph through
  the same protocol. The CLI is the developer-loop entry point; the
  API is the production entry point. Behaviour parity is automatic.
- The agent graph gained a `stream_updates(payload)` method that
  yields `(node_name, state_update)` pairs. This is the only new
  surface on the agent layer needed to support streaming.
- Token streaming, Redis caching, cost routing, and auth are all
  deferred. None of them block running the API end-to-end against
  ingested data; they fold in cleanly when each becomes the most
  valuable next slice.
- The technical specialist's no-data stub is still wired through. A
  client that asks a technical-flavoured question will get a
  `specialist-report` event with empty `claims` / `citations` and
  the documented "no market-data adapter" reason in the summary.
  This is honest, not a bug.
