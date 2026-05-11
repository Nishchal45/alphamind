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
- `alphamind.chunking` package: `parse_filing_html` walks leaf block elements (preserving inline-XBRL and styled spans within their containing paragraph), `detect_sections` groups paragraphs by `Item N` / `Part N` headings with table-of-contents deduplication, and `chunk_text` / `chunk_filing` produce paragraph-aware overlapping chunks under a character budget.
- `FilingChunk` ORM model with migration `0004_filing_chunks`, storing per-document slices keyed on `(filing_document_id, chunk_index)` and tagged with `source_content_hash` so the service can detect stale chunks when a body changes.
- `chunk_filing_document` and `chunk_bodies_for_cik` services: idempotent persistence reading bodies from the `StorageBackend`, replacing prior chunks under a SAVEPOINT so a single failure leaves prior successes intact.
- `scripts/ingest_edgar.py` gains `--chunk` (implies `--with-bodies`) and `--force-chunk`, with a per-CIK summary table reporting documents seen, chunked, skipped, failed, and total chunks written.
- `alphamind.embeddings` package with an `Embedder` protocol, a model-free `DeterministicEmbedder` (hash-based, L2-normalised, 384-dim by default) for tests and dev, and a config-driven `get_embedder` factory so a real backend can be swapped in without touching call sites.
- `FilingChunk.embedding` (`vector(384)`), `embedding_model`, and `embedded_at` columns added via migration `0005_chunk_embeddings`, plus a partial index on `embedding_model` for cheap "needs re-embedding" lookups.
- `embed_chunks_for_document` and `embed_chunks_for_cik` services: batched, idempotent re-embed-on-model-change, dimension-validated, SAVEPOINT-isolated per document.
- `scripts/ingest_edgar.py` gains `--embed` (implies `--chunk`) and `--force-embed`, with a per-CIK summary reporting documents seen, failed, chunks embedded, and chunks skipped.
- `embedder_backend` and `embedder_dimension` settings in `alphamind.config`.
- `GeminiEmbedder`: real backend calling Google's `text-embedding-004` REST endpoint over httpx, with a token-bucket rate limiter (defaults to 20 req/s, well under the 1500 RPM free-tier ceiling), tenacity-based exponential-backoff retries on 429/5xx, transparent slicing into the 100-input `batchEmbedContents` cap, and an `aclose()` lifecycle for clean shutdown via `dispose_embedder()`.
- Migration `0006_widen_embedding_to_768` resizes `filing_chunks.embedding` from `vector(384)` to `vector(768)` to match Gemini's output; the `DeterministicEmbedder` default dimension and `Settings.embedder_dimension` default move to 768 in lockstep.
- `google_api_key` and `gemini_embedding_model` settings; ``.env.example`` documents the Google AI Studio workflow.
- `alphamind.retrieval` package implementing hybrid search over filing chunks: `dense_search` (pgvector cosine ANN), `bm25_search` (`ts_rank_cd` over a generated `text_tsv` column), and `hybrid_search` (Reciprocal Rank Fusion of the two). All three accept the same filters: `as_of_date` (no filings dated after — required for honest backtests), `cik`, `form_types`, `section_labels`.
- `RetrievalResult` dataclass carrying enough filing context (company, form, accession number, section, filing date) to cite a chunk without a second query, plus per-retriever ranks (`dense_rank`, `bm25_rank`) on hybrid results for debuggability.
- Migration `0007_retrieval_indexes` adds a generated `text_tsv tsvector` column (`to_tsvector('english', text)`, STORED) with a GIN index, plus an HNSW index on `embedding` using `vector_cosine_ops`. Alembic env.py gains an `include_object` filter so autogenerate ignores the raw-SQL-managed objects.
- `scripts/query.py` CLI: ad-hoc hybrid/dense/BM25 search with `--mode`, `--k`, `--cik`, `--form`, `--section`, and `--as-of` flags; prints company, form, accession, section, score, and per-retriever ranks for each hit.
- `alphamind.reranking` package implementing cross-encoder reranking: a `Reranker` protocol, a `DeterministicReranker` (Jaccard overlap, model-free) for tests and dev, and a `CrossEncoderReranker` wrapping `sentence-transformers` behind the optional `rerank` extra (`uv sync --extra rerank`). `rerank_results` adapts a reranker onto a list of `RetrievalResult`, replacing the score and preserving the per-retriever ranks.
- `scripts/query.py` gains `--rerank`; when set, it widens the retrieval candidate pool and applies the configured reranker before printing.
- `reranker_backend` and `cross_encoder_model` settings in `alphamind.config`.

### Changed
- `EdgarClient` no longer sends a fixed `Accept: application/json` header — the same client now hits both JSON endpoints under `data.sec.gov` and HTML/XML bodies under `www.sec.gov/Archives`.

[Unreleased]: https://github.com/Nishchal45/alphamind/compare/HEAD...HEAD
