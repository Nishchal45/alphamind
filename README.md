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
- [ ] Phase 2 — filing-body ingestion, chunking, embeddings, hybrid retrieval, cross-encoder rerank
- [ ] Phase 3 — LangGraph agent team (router, specialists, synthesizer, critic)
- [ ] Phase 4 — fine-tuned SLM on financial text (LoRA / QLoRA)
- [ ] Phase 5 — FastAPI serving layer with streaming, caching, cost routing
- [ ] Phase 6 — backtest harness, evaluation set, public result dashboard

## Disclaimer

This is a research and education project. Output is **not financial advice**. Simulated historical performance is not a prediction of future results. Don't trade on its recommendations without independent verification and proper risk management.

## License

[MIT](LICENSE).
