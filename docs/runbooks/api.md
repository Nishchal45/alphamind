# Runbook — FastAPI service

The agent graph behind an HTTP surface. CLI (`scripts/research.py`)
remains; the API is the productised path. Design rationale in
[ADR 0008](../adr/0008-fastapi-serving-and-sse.md).

## What it does

```
POST /research { query, as_of, top_k }
  → SSE stream:
      event: router-decision     data: {...}
      event: specialist-report   data: {...}   (one per specialist)
      event: thesis              data: {...}
      event: critic-report       data: {...}
      event: done                data: {}

GET /healthz   → 200 {status: ok}                       (liveness)
GET /readyz    → 200 ok / 503 down                      (Postgres reachability)
GET /docs      → Swagger UI
GET /openapi.json
```

## Running locally

```bash
# real models / embeddings
LLM_BACKEND=anthropic ANTHROPIC_API_KEY=sk-ant-... \
EMBEDDING_BACKEND=gemini GOOGLE_API_KEY=... \
  make serve

# defaults — runs end-to-end with stubs (echo LLM, deterministic embedder)
make serve
```

`make serve` runs `uvicorn --factory alphamind.api.app:create_app
--reload --host 0.0.0.0 --port 8000`. Adjust host/port as needed.

## Calling it

```bash
curl -N -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query":"NVDA China exposure","as_of":"2024-12-31","top_k":8}'
```

The `-N` flag is critical — without it, curl buffers the SSE stream
and you see nothing until the run finishes, defeating the point.

Expected output (truncated):

```
event: router-decision
data: {"specialists":["fundamentals","risk"],"rationale":"..."}

event: specialist-report
data: {"specialist":"fundamentals","summary":"...","claims":[...],"citations":[...]}

event: specialist-report
data: {"specialist":"risk","summary":"...","claims":[...],"citations":[...]}

event: thesis
data: {"answer":"...","bull":[...],"bear":[...],"citations":[...]}

event: critic-report
data: {"unsupported":[],"notes":"well-supported","ok":true}

event: done
data: {}
```

From JavaScript:

```js
const res = await fetch("/research", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ query, as_of, top_k: 8 }),
});
// Use ReadableStream + TextDecoder, or a small SSE parser. Browsers'
// built-in EventSource only supports GET; for POST you parse the
// chunks manually.
```

## Reading the events

| Event | When fired | Payload |
| ----- | ---------- | ------- |
| `router-decision` | Right after the router runs (once) | `specialists`, `rationale` |
| `specialist-report` | After each specialist completes; parallel branches surface in completion order | `specialist`, `summary`, `claims`, `citations` |
| `thesis` | After the synthesizer (once) | `answer`, `bull`, `bear`, `citations` |
| `critic-report` | After the critic (once) | `unsupported`, `notes`, `ok` |
| `done` | Terminal sentinel after a clean run | `{}` |
| `error` | Terminal sentinel after a graph-level failure | `{message}` |

Clients should treat `done` and `error` as terminal and close the
connection. A connection that closes without either signal indicates
the network dropped — retry or surface the failure.

## Failure modes

| Symptom | Likely cause | Action |
| ------- | ------------ | ------ |
| `422 Unprocessable Entity` | Pydantic rejected the body — missing `as_of`, blank `query`, `top_k` outside [1, 50] | Read the response body; it names the offending field |
| `503` on `/readyz` | Postgres unreachable | Check `make compose-up`; confirm `DATABASE_URL` |
| Stream ends without `done` or `error` | Connection dropped (proxy timeout, client cancelled) | Retry; check proxy idle-timeout |
| `event: error` mid-stream | The graph raised. Inner-node errors are caught inside the graph, so this means input validation slipped through or a wiring bug. | Read the `message`; check server logs |
| All `specialist-report` events have empty `claims` | No chunks matched `as_of` cutoff | Re-ingest filings or widen `as_of` |
| `technical` specialist always returns empty | Expected — no-data stub until the market-data adapter exists | Ignore or treat as a TODO |

## Operational notes

- `--as-of` is mandatory (same reasoning as `ask.py` and
  `research.py`; see ADR 0005). There is no default — the API
  enforces this with a Pydantic `Field(...)` with no default.
- The lifespan builds the `ResearchGraph` once and reuses it across
  requests. Cold-start cost: the LLM client + embedder factories
  initialise on first request, not at startup. Cold-request latency
  is ~100ms higher than warm.
- Server-Sent Events stream over a long-lived HTTP connection. Any
  proxy in front of the service must allow long-lived connections
  (nginx: `proxy_read_timeout 300s; proxy_buffering off;`).
- The agent graph runs the LLM 6+ times per request (router, ~4
  specialists, synthesizer, critic). Cost per request is non-trivial
  with frontier models — Redis caching of identical
  `(query, as_of)` pairs lands in a follow-up.
- Token-level streaming of the synthesizer answer is **not** in this
  cut. The `thesis` event arrives as one chunk. The follow-up that
  grows `LLMClient` with a streaming method will surface tokens as
  incremental `thesis-delta` events.
