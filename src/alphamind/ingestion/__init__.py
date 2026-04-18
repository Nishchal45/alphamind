"""Data ingestion adapters for external sources.

Each sub-package targets a single upstream system (SEC EDGAR today; news and
transcripts will follow). Adapters expose two things:

1. A transport-level client that wraps the upstream API with rate limiting,
   retries, and typed request/response schemas.
2. An ingestion service that maps fetched records into the ORM layer under a
   ``session_scope()`` context manager with idempotent upsert semantics.
"""
