# AlphaMind

AlphaMind is an agentic research tool for public equities. It reads SEC filings, earnings transcripts, and news, then produces a structured thesis on a company where every claim cites the source it came from.

It's a personal project — I wanted something that would force me to learn multi-agent orchestration, retrieval at scale, and fine-tuning a small open-source model on financial text — and I got tired of LLM answers that quietly hallucinate a revenue number.

> Status: early development. See [`CHANGELOG.md`](CHANGELOG.md) for what has actually shipped.

---

## What it does

Given a question like *"What's the bull and bear case for NVDA going into Q3 2026?"*, a small team of specialised agents works the problem in parallel:

- A **router** figures out intent.
- **Fundamentals**, **sentiment**, **technical**, and **risk** agents each work over their own filtered slice of the corpus.
- A **synthesizer** merges the outputs into a structured thesis with bull and bear sides.
- A **critic** reads the output back and flags unsupported claims, internal contradictions, and hallucinations before the answer reaches me.

Every claim is linked back to a source document and a timestamp. For historical questions the system refuses to look at anything dated after the as-of date, which is how I'll keep backtests honest.

## Architecture

- **Ingestion**: SEC EDGAR (10-K / 10-Q / 8-K), earnings transcripts, news feeds, market data.
- **Retrieval**: hybrid (dense embeddings + BM25) with cross-encoder reranking and chunking tuned for filings.
- **Orchestration**: LangGraph DAG with typed state.
- **Models**: a fine-tuned open-source SLM (LoRA / QLoRA) for the domain-specific tasks; a frontier LLM for final synthesis.
- **Serving**: FastAPI with async streaming, Redis cache, model routing for cost.
- **Evaluation**: citation coverage, hallucination rate, and a historical backtest against SPY.

Full design in [`docs/architecture.md`](docs/architecture.md). Architectural decisions are in [`docs/adr/`](docs/adr).

## Development

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependencies
- Docker + Docker Compose for the local Postgres (with `pgvector`) and Redis

### Setup

```bash
make install       # install dependencies and pre-commit hooks
make compose-up    # start postgres and redis
make migrate       # apply database migrations
make healthcheck   # verify services are reachable
make test          # run the unit test suite
```

### Common commands

| Command | What it does |
| --- | --- |
| `make lint` | Ruff checks |
| `make format` | Auto-format with ruff |
| `make typecheck` | mypy, strict |
| `make test` | Unit tests |
| `make test-cov` | Tests + coverage |
| `make migrate` | Apply Alembic migrations to head |
| `make migration M="..."` | Autogenerate a new migration |
| `make healthcheck` | Ping Postgres, pgvector, and Redis |
| `make ci` | Full local CI pipeline |

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the workflow.

### Ingesting SEC filings

Once services are up and migrations applied, pull recent filings for a few tickers:

```bash
uv run python scripts/ingest_edgar.py --ticker AAPL MSFT NVDA
```

### Asking a question

After bodies are fetched (`--with-bodies`) and chunks are built, run:

```bash
LLM_BACKEND=anthropic ANTHROPIC_API_KEY=sk-ant-... \
  uv run python scripts/ask.py \
    --query "what is NVDA saying about China revenue concentration?" \
    --as-of 2024-12-31
```

This is the first end-to-end demo — BM25 retrieval over your ingested filings + a Claude call that's instructed to answer using only the cited sources. Without an API key set, the default `LLM_BACKEND=echo` returns a stub so the rest of the pipeline can still be exercised. Operational details in [`docs/runbooks/ask.md`](docs/runbooks/ask.md).

### Running the agentic pipeline

`scripts/ask.py` is the flat BM25 → LLM smoke test. For the multi-agent
pipeline — router → specialists → synthesizer → critic — use:

```bash
LLM_BACKEND=anthropic ANTHROPIC_API_KEY=sk-ant-... \
  uv run python scripts/research.py \
    --query "Bull and bear case for NVDA going into Q3 2026" \
    --as-of 2024-12-31
```

The graph is wired with the fundamentals specialist in this slice;
sentiment / technical / risk land in follow-up PRs and slot in via the
shared `SpecialistBase`. Design rationale in [`docs/adr/0007-agent-graph-design.md`](docs/adr/0007-agent-graph-design.md);
operational details in [`docs/runbooks/research.md`](docs/runbooks/research.md).

### Running the API

```bash
make serve  # uvicorn --factory alphamind.api.app:create_app --reload
curl -N -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query":"NVDA China exposure","as_of":"2024-12-31"}'
```

`POST /research` runs the agent graph and streams progress as Server-Sent
Events (one event per node completion). `GET /healthz` and `GET /readyz`
are k8s-shaped liveness / readiness probes. Design rationale in
[`docs/adr/0008-fastapi-serving-and-sse.md`](docs/adr/0008-fastapi-serving-and-sse.md);
operational details in [`docs/runbooks/api.md`](docs/runbooks/api.md).

Example output:

```
CIK          TICKER     SEEN  WRITTEN  NAME
0000320193   AAPL         12        5  Apple Inc.
0000789019   MSFT         14        6  Microsoft Corporation
0001045810   NVDA         11        4  NVIDIA Corporation
```

`SEEN` is what EDGAR returned in the recent-filings window; `WRITTEN` is how many passed the form filter and got upserted. Operational details in [`docs/runbooks/ingest-edgar.md`](docs/runbooks/ingest-edgar.md).

## Roadmap

- [x] Phase 1 — repo scaffolding, Postgres + pgvector, SEC EDGAR metadata ingestion
- [x] Phase 2 — filing-body ingestion, finance-aware chunking, embeddings, hybrid retrieval (BM25 + pgvector + RRF + cross-encoder rerank) with a hard time-horizon filter at every stage
- [ ] Phase 3 — LLM provider integration (Anthropic adapter shipped), real sentence-transformer embedder + cross-encoder rerank (shipped), LangGraph agent team (router + fundamentals specialist + synthesizer + critic shipped; remaining specialists in follow-ups)
- [ ] Phase 4 — fine-tuned SLM on financial text (LoRA / QLoRA)
- [ ] Phase 5 — FastAPI serving layer with streaming (SSE shipped), caching, cost routing (caching and cost routing in follow-ups)
- [ ] Phase 6 — backtest harness, evaluation set, public result dashboard

## Disclaimer

This is a research and education project. Output is **not financial advice**. Simulated historical performance is not a prediction of future results. Don't trade on its recommendations without independent verification and proper risk management.

## License

[MIT](LICENSE).
