# Runbook — FastAPI service

The agent graph behind an HTTP surface. CLI (`scripts/research.py`)
remains; the API is the productised path. Design rationale in
[ADR 0009](../adr/0009-fastapi-serving-and-sse.md).

## What it does

```
POST /research { query, as_of, top_k }
  → SSE stream:
      event: router-intent        data: {...}
      event: specialist-findings  data: {...}   (one per specialist that ran)
      event: thesis               data: {...}
      event: critique             data: {...}
      event: done                 data: {}

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

`make serve` runs `uvicorn --factory alphamind.api.app:create_app --reload`.

## Calling it

```bash
curl -N -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query":"NVDA China exposure","as_of":"2024-12-31","top_k":8}'
```

The `-N` flag is critical — without it curl buffers the SSE stream
and nothing appears until the run finishes.

Expected output (truncated):

```
event: router-intent
data: {"primary":"fundamentals","specialists":["fundamentals","risk"],"rationale":"..."}

event: specialist-findings
data: {"specialist":"fundamentals","sources":[...],"findings":[...],"usage":[...]}

event: specialist-findings
data: {"specialist":"risk","sources":[...],"findings":[...],"usage":[...]}

event: thesis
data: {"summary":"...","bull_case":[...],"bear_case":[...]}

event: critique
data: {"issues":[],"parse_error":null}

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
// EventSource only supports GET; for POST you parse the chunks
// manually with ReadableStream + TextDecoder, or use a small SSE
// parser library.
```

## Reading the events

| Event | When | Payload |
| ----- | ---- | ------- |
| `router-intent` | After the router (once) | `primary`, `specialists`, `rationale` |
| `specialist-findings` | After each specialist node; parallel branches surface in completion order | `specialist`, `sources`, `findings`, `usage` |
| `thesis` | After the synthesizer (once) | `summary`, `bull_case`, `bear_case` |
| `critique` | After the critic (once) | `issues`, `parse_error` |
| `done` | Terminal sentinel after a clean run | `{}` |
| `error` | Terminal sentinel after a graph-level failure | `{message}` |

Clients should treat `done` and `error` as terminal and close the
connection. A connection that closes without either signal means the
network dropped — retry or surface the failure.

The `technical` specialist runs as a no-data stub today (see [ADR 0007](../adr/0007-agent-team-design.md));
its `specialist-findings` event arrives with empty `findings`,
`sources`, and `usage`. This is honest, not a bug.

## Failure modes

| Symptom | Likely cause | Action |
| ------- | ------------ | ------ |
| `422 Unprocessable Entity` | Pydantic rejected the body — missing `as_of`, blank `query`, `top_k` outside [1, 50] | Read the response body; it names the field |
| `503` on `/readyz` | Postgres unreachable | Check `make compose-up`; confirm `DATABASE_URL` |
| Stream ends without `done` or `error` | Connection dropped (proxy timeout, client cancelled) | Retry; check proxy idle-timeout |
| `event: error` mid-stream | The graph raised. Inner-node errors are caught inside the graph, so this means wiring bug or input validation slipped through | Read `message`; check server logs |
| All `specialist-findings` events have empty `findings` | No chunks matched `as_of` cutoff | Re-ingest filings or widen `as_of` |
| `technical` always empty | Expected — no-data stub until the market-data adapter exists | Ignore |
| `critique.parse_error` is non-null | Critic LLM emitted unparseable output; the thesis is not blocked but the critic didn't validate | Re-run; tighten `LLM_MODEL` |

## Operational notes

- `--as-of` is mandatory (same reasoning as `ask.py` / `research.py`;
  see ADR 0005). The API enforces this with a Pydantic `Field(...)`
  with no default.
- The lifespan builds the compiled research graph once and reuses it
  across requests. First-request latency is ~100ms higher than warm
  because the LLM client + embedder factories initialise lazily.
- Server-Sent Events stream over a long-lived HTTP connection. Any
  proxy in front of the service must allow that (nginx:
  `proxy_read_timeout 300s; proxy_buffering off;`).
- The agent graph runs the LLM 6+ times per request (router, up to
  four specialists, synthesizer, critic). Cost per request is non-
  trivial with frontier models — Redis caching of identical
  `(query, as_of)` pairs lands in a follow-up.
- Token-level streaming of the synthesizer answer is **not** in this
  cut. The `thesis` event arrives as one chunk. The follow-up that
  grows `LLMClient` with a streaming method will surface deltas
  incrementally.
