# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `alphamind.api` package: FastAPI serving layer with two endpoints. `GET /healthz` is the liveness probe (returns `{"status": "ok", "graph_ready": true}` once the compiled graph is bound to `app.state`). `POST /research` streams a research run as Server-Sent Events — one `node` event per LangGraph super-step, terminating in a synthetic `done` event or a structured `error` event if anything raises mid-stream. The body is validated against the `ResearchRequest` Pydantic schema (`query`, `as_of` required — no default, per ADR 0005 — and `top_k` 1-32).
- `alphamind.api.streaming.stream_research_events`: SSE adapter that consumes `graph.astream(mode="updates")` and serialises each node's partial state via a `_to_jsonable` walker (frozen dataclasses → dicts, `date`/`datetime` → ISO 8601, tuples → lists). Exceptions are caught and surfaced as `error` SSE events instead of leaking through the connection, then the stream closes cleanly.
- `create_app(graph=None)` factory in `alphamind.api.app`: production passes `None` and the factory builds the real graph from the embedder/reranker/LLM factories (mirrors `scripts/research.py`); tests pass a stub graph wired with `SystemKeyedLLMClient` and exercise the full HTTP surface via `httpx.AsyncClient + ASGITransport`. Graph construction happens at factory time so `app.state.graph` is available even when the lifespan handler doesn't fire (which it doesn't under raw httpx ASGI transport). The lifespan owns disposal only — `dispose_embedder`, `dispose_reranker`, `dispose_engine`.
- `alphamind.agents.retrieval.build_filing_retrieval_fn`: shared adapter from `HybridSearch` to the agent `RetrievalFn` contract. Previously inlined in `scripts/research.py`; the API needed the same wiring, so it moved into `src/` and both call sites import it.
- `api_host`, `api_port`, `api_cors_origins` settings (loopback / 8000 / empty by default). CORS off unless `API_CORS_ORIGINS` is set.
- `make api` target: `uvicorn alphamind.api.app:app` on `API_HOST:API_PORT`. The module-level `app` symbol is lazy so test imports don't trigger settings validation.
- `fastapi>=0.115`, `uvicorn[standard]>=0.32`, `sse-starlette>=2.1` added to `dependencies`.
- Tests under `tests/unit/api/` covering the healthz endpoint, the SSE happy path (parsed event-by-event, verifying every node fired and the synthesizer's thesis lands in the stream), input validation (`as_of` required, `query` non-empty, `top_k` in range), and the error path (a graph that raises mid-stream emits a structured `error` event with the exception class and message, no `done` event follows).
- ADR 0009 documenting the API serving design: FastAPI vs. Starlette / Litestar, SSE vs. WebSockets, why node-level streaming and not token-level yet, lifespan-as-disposal-only, and the surface explicitly deferred (auth, rate limits, token-level streaming, readiness probe, cancellation).
- Runbook `docs/runbooks/api.md` covering server start, endpoint shapes, example `curl`, and the failure-mode catalogue.

### Changed
- `scripts/research.py` now imports `build_filing_retrieval_fn` from `alphamind.agents.retrieval` instead of defining `_build_retrieval_fn` / `_hydrate` inline, so the CLI and the API surface share one production-grade adapter from `HybridSearch` to the agent contract.

### Added
- Threshold gating for the eval harness (`alphamind.eval.thresholds`). One-sided bounds (`minimum` *or* `maximum`, never both) per metric, validated at load time; the loader rejects unknown metric names, non-numeric bounds, duplicates, and two-sided entries. `evaluate_thresholds()` scores an `EvalReport` against a `Sequence[Threshold]` and skips (does not fail) a metric whose aggregate has `n == 0` — gating on something we didn't measure would be louder than useful.
- `evals/thresholds.yaml` ships with conservative defaults across all six harness metrics. `citation_validity` is pinned at `minimum: 1.0` (structural invariant — the specialist + synthesizer mechanically drop invalid cites, so anything below 1.0 is a structural-check regression); the critic-driven rates start loose and will tighten as we learn the natural variance.
- `scripts/eval.py` gains a `--thresholds` flag. `make eval` now passes it by default so the CLI gates on each run. Exit codes split: `0` clean, `1` case raised, `2` aggregate metric outside its bound.
- `alphamind.eval` package: in-repo eval harness for the research DAG. Six metrics (`citation_coverage`, `citation_validity`, `hallucination_rate`, `contradiction_rate`, `topic_recall`, `chunk_recall`) implemented as pure functions over a finished `ResearchState`. The runner accepts a pre-built compiled graph and walks a `Sequence[EvalCase]`, so tests run against a stub-wired graph while the production CLI wires the real one.
- Golden-set format and loader: YAML at `evals/golden_set.yaml`, parsed via `alphamind.eval.load_golden_set` with eager validation (missing required fields, bad `as_of` format, duplicate ids all raise `GoldenSetError`). Four seed cases covering bull/bear, fundamentals-heavy, and risk-heavy questions.
- `scripts/eval.py` + `make eval`: wires the production graph (`HybridSearch` + LLM factory) behind the runner. Writes a JSON report with per-case `CaseResult` records and an aggregate `MetricSummary` per metric; prints a human-readable summary. Exit code reflects whether any case raised, not whether metrics passed a threshold (the harness measures, it does not gate today).
- `pyyaml` added to `dependencies` and `types-pyyaml` to the dev group.
- ADR 0008 documenting the eval-harness design: metric definitions, golden-set format choices, why YAML over JSON, why the runner doesn't build the graph itself, what's not covered yet (pass/fail gates, historical backtest, dashboards).
- Runbook `docs/runbooks/eval.md` covering the CLI, output format, how to author a golden case, and the failure-mode catalogue.

- Risk specialist agent (`alphamind.agents.specialists.risk`) — second specialist in the LangGraph DAG. Same retrieval surface and citation contract as the fundamentals specialist; its system prompt biases the LLM toward Item 1A risk factors, Item 3 legal proceedings, Item 7A market-risk disclosures, and regulatory/geopolitical exposure.
- Router → specialist edge converted to a real fan-out: `_route_specialists` reads `state['intent'].specialists` and returns the list of wired specialists to run in parallel. `fundamentals` is always included as a safety net so the synthesizer always has at least one set of findings to work from. LangGraph fans in at the synthesizer once every selected specialist completes.
- Shared `alphamind.agents.specialists._base.make_specialist_node` factory: the JSON-parsing, citation-validation, and state-accumulator plumbing now lives in one place, so each concrete specialist (`fundamentals.py`, `risk.py`) is a thin wrapper binding its prompt and name. Adding sentiment / technical will be the same pattern.
- `alphamind.agents.dedupe_sources`: helper that collapses the per-specialist accumulator down to unique chunk ids while preserving first-seen order. Used by the critic before formatting and by `scripts/research.py` before printing — kept out of the state reducer to preserve the per-specialist provenance LangGraph's `operator.add` accumulator captures.

- `alphamind.agents` package: LangGraph-orchestrated research DAG with four nodes — a router that classifies intent into the fundamentals / sentiment / technical / risk taxonomy, a fundamentals specialist that runs `HybridSearch` and emits 3-7 structured findings with chunk-level citations, a synthesizer that merges findings into a bull / bear thesis, and a critic that flags unsupported claims and contradictions against the source pool. State (`ResearchState`) is a typed `TypedDict` with per-field reducers; accumulators (`sources`, `findings`, `usage`) concatenate across nodes.
- Citation invariant enforced at three layers: the specialist drops findings citing chunk ids not in its source pool; the synthesizer drops claims citing chunk ids not in the union of finding citations; the critic does the semantic pass over what's left. The structural checks are deterministic; the critic is the only LLM-judgement step.
- `langgraph` and `langchain-core` added to `dependencies`. The compiled graph is built by `build_research_graph(llm=..., retrieve=...)`, parameterised on the `LLMClient` Protocol and a `RetrievalFn` callable so the DAG can be tested end-to-end without a database.
- `scripts/research.py` — Phase 3 end-to-end CLI: `--query`, `--as-of`, `--top-k`. Prints router intent, bull/bear thesis with chunk citations, critic issues, source pool, and per-node token usage. Wires `HybridSearch` + Gemini embedder + cross-encoder reranker behind the agent graph.
- ADR 0007 documenting the agent-team design: LangGraph choice, vertical-slice scope, citation invariant, and what's deferred (sentiment/technical/risk specialists, fan-out, tool-calling retrieval, streaming, tracing).
- Runbook `docs/runbooks/research.md` covering CLI usage, output format, and failure modes.

- Initial repository scaffolding: license, README, src-layout package.
- Python tooling with ruff, mypy (strict), pytest, pre-commit, and a uv-based workflow.
- GitHub Actions CI pipeline for lint, typecheck, and test across Python 3.11 and 3.12.
- Dependabot for weekly pip and monthly github-actions updates.
- Issue templates, pull request template, and CODEOWNERS.
- Contribution guidelines, code of conduct, and an Architecture Decision Record framework.
- Local development stack via Docker Compose: PostgreSQL with `pgvector` and Redis, each with healthchecks and named volumes.
- Typed application configuration (`alphamind.config`) backed by `pydantic-settings`.
- Async SQLAlchemy 2.0 session layer with a deterministic constraint naming convention and a `session_scope()` context manager.
- Alembic wiring with a baseline migration that enables the `pgvector` extension; `make migrate`, `make migration`, `make downgrade`, `make db-reset`, and `make healthcheck` targets.
- ADR 0002 documenting the choice of PostgreSQL + `pgvector` over a dedicated vector database.
- `Company` and `Filing` ORM models with migration `0002_companies_filings`, indexed on CIK, ticker, form, and `filing_date` to support time-aware retrieval.
- Async SEC EDGAR HTTP client with identifying `User-Agent`, a token-bucket rate limiter (default 8 req/s), and tenacity retries on 429 / 5xx responses.
- Typed EDGAR response schemas (`SubmissionsResponse`, `RecentFilings`, `TickerRecord`) and `iter_filings()` helper that zips the SEC's parallel arrays into record form.
- `ingest_cik` / `ingest_ticker` service functions using Postgres `INSERT ... ON CONFLICT DO UPDATE` for idempotent upserts.
- `scripts/ingest_edgar.py` CLI accepting `--ticker`, `--cik`, `--forms`, and `--limit`, with per-item error isolation so a single bad input does not abort a batch.
- Runbook `docs/runbooks/ingest-edgar.md` covering prerequisites, common invocations, and failure modes.
- ADR 0003 documenting the EDGAR ingestion design.
- `alphamind.storage` package with a narrow `StorageBackend` protocol (`put` / `get` / `exists`), a content-addressable `LocalFilesystemStorage` implementation that shards by key prefix, and a config-driven factory so production backends can be swapped in later.
- `FilingDocument` ORM model with migration `0003_filing_documents`, recording the storage URI, SHA-256 content hash, byte size, MIME type, source URL, and fetch timestamp for each filing's primary document body.
- `EdgarClient.get_primary_document()` for fetching filing bodies from EDGAR Archives, returning `(bytes, content_type, source_url)`.
- `ingest_bodies_for_cik` service and `--with-bodies` CLI flag for `scripts/ingest_edgar.py`. Idempotent: refetched bodies whose SHA-256 matches the existing row skip the storage write and the upsert.
- ADR 0004 documenting the storage layer design.
- `alphamind.retrieval.chunking` package: HTML→text via bs4, regex-based 10-K / 10-Q section detection (Item 1A, 7, 7A, 8, etc.), and a token-aware sliding-window splitter using tiktoken's `cl100k_base`. `ChunkingPipeline` wires them together; chunks never cross section boundaries.
- `FilingChunk` ORM model with migration `0004_filing_chunks`. Includes a Postgres-generated `text_tsv` column with a GIN index for BM25 lexical retrieval and a nullable `Vector(384)` `embedding` column with an HNSW index for dense ANN.
- `chunk_filing` / `chunk_filings_for_cik` services that read filing bodies through the storage backend, run them through the chunker, and replace prior chunks atomically inside one transaction.
- `alphamind.retrieval.embeddings` package: narrow `Embedder` protocol (`dim` + `embed`), a `DeterministicHashEmbedder` for tests/dev that lets the pipeline run end-to-end without a model download, a config-driven factory (`EMBEDDING_BACKEND`), and an `embed_chunks_for_filing` service that batches inputs and is idempotent on re-runs.
- `alphamind.retrieval.search` package: BM25 (`lexical_search`), pgvector cosine ANN (`dense_search`), Reciprocal Rank Fusion (`reciprocal_rank_fusion`), a `Reranker` protocol with a `DeterministicReranker` Jaccard stub, and the end-to-end `HybridSearch` orchestrator. The `as_of` time-horizon parameter is required at every stage that touches the database.
- ADR 0005 documenting the retrieval pipeline design and the three-layer enforcement of the time-horizon invariant.
- Runbook `docs/runbooks/retrieval.md` covering chunker / embedder / search invocation and failure modes.
- `alphamind.llm` package: narrow `LLMClient` Protocol, `AnthropicLLMClient` adapter wrapping the official SDK with tenacity retries on transient errors and clean error mapping into `LLMClientError`, and an `EchoLLMClient` stub for offline development. Config-driven factory (`LLM_BACKEND`).
- `scripts/ask.py` — the project's first end-to-end demo. Runs BM25 search over `filing_chunks`, formats the top-k chunks as numbered sources, and asks the LLM to answer using only those sources with citations. `--as-of` is required, not optional.
- ADR 0006 documenting the LLM provider integration design.
- Runbook `docs/runbooks/ask.md` covering the new CLI's invocation and failure modes.
- `GeminiEmbedder`: real embedding backend calling Google's `gemini-embedding-001` REST endpoint over `httpx + tenacity + token-bucket`. Free-tier-aware defaults (20 req/s, well under the 1500 RPM ceiling), transparent slicing into the 100-input `batchEmbedContents` cap, and an `aclose()` lifecycle. Truncates the Matryoshka output to `EMBEDDING_DIM` via `outputDimensionality` and L2-renormalises so the unit-norm contract in the `Embedder` protocol still holds; uses `taskType=RETRIEVAL_DOCUMENT` for chunk encoding.
- `google_api_key` and `gemini_embedding_model` settings; `EMBEDDING_BACKEND=gemini` selects the new backend. `dispose_embedder()` factory hook for shutting the HTTP client down cleanly.
- `CrossEncoderReranker`: real reranker wrapping `sentence_transformers.CrossEncoder` behind the existing `Reranker` protocol, lazy-loading the model on first call and dispatching inference to a worker thread so the event loop stays free. `sentence-transformers` is shipped as the optional `rerank` extra (`uv sync --extra rerank`) so CI doesn't pay the ~1GB torch install cost.
- `get_reranker()` / `dispose_reranker()` factory in `alphamind.retrieval.search.reranker_factory`, mirroring the embedder factory. Picks the backend from `RERANKER_BACKEND` config.
- `reranker_backend` and `cross_encoder_model` settings (default model `cross-encoder/ms-marco-MiniLM-L-12-v2`, as ADR 0005 pins).
- Integration test suite under `tests/integration/` covering the SQL paths the unit suite can't reach: chunk persistence + generated `text_tsv` materialisation, embedding write-back into `vector(384)`, HNSW cosine ANN, `ts_rank_cd` lexical ranking, and the end-to-end `HybridSearch.search()` pipeline. The time-horizon (`as_of`) filter is exercised at every layer where it appears.
- CI gains an `integration` job that provisions Postgres + pgvector via a service container, applies migrations, and runs `pytest -m integration`. Unit tests continue to run on Python 3.11 and 3.12 in a separate job.

### Changed
- `EdgarClient` no longer sends a fixed `Accept: application/json` header — the same client now hits both JSON endpoints under `data.sec.gov` and HTML/XML bodies under `www.sec.gov/Archives`.
- README roadmap: Phase 2 is now complete (chunker, embeddings, hybrid retrieval, cross-encoder rerank).
- README roadmap: Phase 3 marked partial — router + fundamentals specialist + synthesizer + critic shipped; sentiment / technical / risk specialists still to land.

[Unreleased]: https://github.com/Nishchal45/alphamind/compare/HEAD...HEAD
