# Runbook — research API

The FastAPI serving layer exposes the agent DAG (router → specialists →
synthesizer → critic) over HTTP with Server-Sent Events streaming.
Design details in [ADR 0009](../adr/0009-api-serving-layer.md).

## What it does

```
HTTP POST /research  →  validate ResearchRequest  →
                        graph.astream(updates)  →
                        SSE event per node  →
                        terminal "done" or "error" event
```

## Prerequisites

The same data path the CLI uses:

- Phase 1 metadata ingest run for at least one ticker.
- Phase 1 body ingest (`--with-bodies`).
- Phase 2 chunking + embedding (`chunk_filings_for_cik`, `embed_chunks_for_filing`).
- `.env` with `DATABASE_URL`, `REDIS_URL`, `SEC_USER_AGENT`.

To get real (non-echo) answers:

```env
LLM_BACKEND=anthropic
LLM_MODEL=claude-sonnet-4-5
ANTHROPIC_API_KEY=sk-ant-...
EMBEDDING_BACKEND=gemini
GOOGLE_API_KEY=...
RERANKER_BACKEND=cross_encoder
```

The default backends (deterministic embedder, deterministic reranker,
echo LLM) let the server boot and respond without any keys — useful
for wiring smoke tests.

## Starting the server

```bash
make api                            # 127.0.0.1:8000
API_HOST=0.0.0.0 API_PORT=9000 make api
```

Or directly:

```bash
uv run uvicorn alphamind.api.app:app --reload --port 8000
```

The `--reload` flag is uvicorn's dev mode (watches source files);
omit it in production.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/healthz` | Liveness. Returns `{"status": "ok", "graph_ready": true}`. |
| `POST` | `/research` | Stream a research run. |
| `GET` | `/docs` | OpenAPI / Swagger UI (FastAPI default). |

### `POST /research`

Body (JSON):

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `query` | string | yes | 1-2000 chars |
| `as_of` | date (YYYY-MM-DD) | **yes** | No default — preventing lookahead bias is the project's most important correctness invariant (ADR 0005) |
| `top_k` | int | no | Default 8; range 1-32 |

Response: `text/event-stream`. Events:

| Event | When | Data shape |
| --- | --- | --- |
| `node` | Once per LangGraph super-step | `{"node": <name>, "payload": <partial state>}` |
| `done` | After the graph completes cleanly | `{}` |
| `error` | If anything raises mid-stream | `{"error": <message>, "type": <exception class>}` |

The `payload` for each `node` event is the partial state update that
node returned, JSON-encoded — chunk lists for the specialists, a
`thesis` object from the synthesizer, a `critique` from the critic,
etc. Dataclasses are flattened to plain dicts; dates are ISO 8601
strings.

## Example

```bash
curl -N -X POST http://127.0.0.1:8000/research \
  -H 'Content-Type: application/json' \
  -d '{
        "query": "What is NVDA saying about China revenue concentration?",
        "as_of": "2024-12-31",
        "top_k": 6
      }'
```

Sample output (abbreviated):

```
event: node
data: {"node": "router", "payload": {"intent": {"primary": "fundamentals", ...}}}

event: node
data: {"node": "fundamentals", "payload": {"sources": [...], "findings": [...]}}

event: node
data: {"node": "risk", "payload": {"sources": [...], "findings": [...]}}

event: node
data: {"node": "synthesizer", "payload": {"thesis": {"summary": "...", "bull_case": [...]}}}

event: node
data: {"node": "critic", "payload": {"critique": {"issues": []}}}

event: done
data: {}
```

## Failure modes

| Symptom | Likely cause | Action |
| --- | --- | --- |
| Server fails to start with `EmbedderError: ... requires GOOGLE_API_KEY` | `EMBEDDING_BACKEND=gemini` but no key | Set `GOOGLE_API_KEY` or switch to `EMBEDDING_BACKEND=deterministic` |
| `503 research graph is not initialised` | Lifespan failed during startup | Check server logs; usually a missing env var or unreachable DB |
| `422 Unprocessable Entity` | Validation failure (missing `as_of`, empty `query`, `top_k` out of range) | Fix the request body per the schema above |
| Stream emits `error` then closes | Exception inside the graph — most often the LLM provider failing, or no chunks dated before `as_of` | The event payload includes the exception class and message; check logs |
| No node events arrive at all | Reverse-proxy buffering | Disable buffering for `text/event-stream` responses (nginx: `proxy_buffering off`) |
| Connection hangs | Slow LLM call | Cancellation isn't wired yet (ADR 0009 deferred) — close the client; the upstream LLM call will eventually complete or time out |

## Operational notes

- The compiled graph is built once at app construction and lives for
  the process lifetime. Restart the server to pick up changes to
  prompts, retrieval config, or model selection.
- One process = one graph. Scale horizontally with more uvicorn
  workers; sticky sessions aren't needed.
- CORS is off by default. Set `API_CORS_ORIGINS=https://example.com`
  (comma-separated) to enable.
- Auth is not built in. Front the server with a reverse proxy or an
  IP allow-list if exposed publicly.
- The `--as-of` rule from ADR 0005 is enforced at the request schema:
  the field is required and there is no default. Don't add one.
