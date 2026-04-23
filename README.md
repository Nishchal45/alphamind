# AlphaMind

An agentic equity research platform that performs institutional-grade analysis of public companies by grounding every claim in primary sources — SEC filings, earnings transcripts, and news — and validating its recommendations through rigorous offline backtesting.

> Status: early development. See [`CHANGELOG.md`](CHANGELOG.md) for what has shipped.

---

## What it does

AlphaMind answers questions like *"What is the bull and bear case for NVDA heading into Q3 2026?"* by orchestrating a team of specialised LLM agents over a time-aware retrieval layer:

- A **router** classifies intent and dispatches to specialists.
- **Fundamentals**, **sentiment**, **technical**, and **risk** agents work in parallel over filtered document sets.
- A **synthesizer** merges their outputs into a structured thesis.
- A **critic** flags unsupported claims, contradictions, and hallucinations before the response reaches the user.

Every claim is traceable to a source document and timestamp. The system enforces a strict information horizon during historical analysis so it cannot accidentally use future information.

## Architecture

- **Ingestion**: SEC EDGAR (10-K / 10-Q / 8-K), earnings transcripts, news feeds, market data.
- **Retrieval**: hybrid search (semantic embeddings + BM25) with cross-encoder reranking, tuned chunking for financial documents.
- **Orchestration**: LangGraph multi-agent DAG with typed state and deterministic routing.
- **Models**: fine-tuned open-source SLM (LoRA / QLoRA) for domain tasks; frontier LLMs for synthesis.
- **Serving**: FastAPI with async streaming, Redis caching, model routing for cost and latency.
- **Evaluation**: citation coverage, hallucination rate, and historical backtest against SPY.

See [`docs/architecture.md`](docs/architecture.md) for the full design and [`docs/adr/`](docs/adr) for recorded architectural decisions.

## Development

### Prerequisites

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/) for dependency management
- Docker and Docker Compose for local services (Postgres with `pgvector`, Redis)

### Setup

```bash
make install       # install dependencies and pre-commit hooks
make compose-up    # start postgres and redis
make migrate       # apply database migrations
make healthcheck   # verify services are reachable
make test          # run the unit test suite
```

### Common commands

| Command | Description |
| --- | --- |
| `make lint` | Run ruff checks |
| `make format` | Auto-format with ruff |
| `make typecheck` | Run mypy in strict mode |
| `make test` | Run unit tests |
| `make test-cov` | Run tests with coverage report |
| `make migrate` | Apply Alembic migrations to head |
| `make migration M="..."` | Autogenerate a new migration |
| `make healthcheck` | Verify Postgres, pgvector, and Redis |
| `make ci` | Run the full local CI pipeline |

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full contribution workflow.

### Ingesting SEC filings

Once services are up and migrations are applied, pull recent filings for a few
tickers:

```bash
uv run python scripts/ingest_edgar.py --ticker AAPL MSFT NVDA
```

Full operational details live in [`docs/runbooks/ingest-edgar.md`](docs/runbooks/ingest-edgar.md).

## Disclaimer

AlphaMind is a research and education project. Its output is **not financial advice**. Simulated historical performance does not guarantee future results. Do not trade on its recommendations without independent verification and appropriate risk management.

## License

Released under the [MIT License](LICENSE).
