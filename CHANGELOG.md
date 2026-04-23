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

[Unreleased]: https://github.com/Nishchal45/alphamind/compare/HEAD...HEAD
