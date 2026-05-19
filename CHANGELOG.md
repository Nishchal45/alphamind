# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
- `alphamind.agents` package: LangGraph DAG (`router -> specialists -> synthesizer -> critic`) with typed state. Frozen-dataclass leaf records (`Citation`, `Claim`, `SpecialistReport`, `Thesis`, `CriticReport`) for immutability and to keep pydantic off the hot path; specialist reports merge via a concatenation reducer so parallel branches converge correctly.
- `RouterNode`: LLM-based intent classification that returns a `RouterDecision` (which specialists to run + a one-sentence rationale). Tolerant JSON extraction (`alphamind.agents.json_utils`) strips fenced blocks, balances braces, and validates schema; falls back to running every specialist if the response can't be parsed.
- `SpecialistBase` scaffold owning retrieval, hydration, prompting, parsing, and fallback construction. Specialists differ only in `name`, `system_prompt`, and `query_augmentation`. First concrete subclass: `FundamentalsSpecialist`, focused on revenue / margins / segments / guidance.
- `SynthesizerNode`: merges per-specialist citation pools into one renumbered list (deduped by `chunk_id`) before the LLM is asked to compose the bull / bear / answer thesis. Renumbering keeps the critic's view of citations canonical.
- `CriticNode`: LLM-judge with structured JSON output that flags unsupported claims by re-reading the thesis against the merged source pool. The critic is a check, not a gate — its own failures degrade to an empty `unsupported` list with the failure noted in `notes`.
- `scripts/research.py` — first agentic end-to-end demo of the project. Runs the LangGraph DAG against ingested filings with the same `--as-of` discipline as `ask.py`.
- ADR 0007 documenting the agent-graph design (LangGraph, typed state, scaffolding, the renumber-before-LLM contract, LLM-judge critic, graph-wide fallback policy).
- Runbook `docs/runbooks/research.md` covering invocation, expected output, and failure modes.

### Changed
- `EdgarClient` no longer sends a fixed `Accept: application/json` header — the same client now hits both JSON endpoints under `data.sec.gov` and HTML/XML bodies under `www.sec.gov/Archives`.
- README roadmap: Phase 2 is now complete (chunker, embeddings, hybrid retrieval, cross-encoder rerank). Phase 3 is in progress — LLM provider integration shipped, real sentence-transformer embedder + cross-encoder rerank shipped, and the LangGraph skeleton + fundamentals specialist + synthesizer + critic now ship in this slice. Remaining specialists (sentiment, technical, risk) land in follow-up PRs.

[Unreleased]: https://github.com/Nishchal45/alphamind/compare/HEAD...HEAD
