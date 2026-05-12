# Architecture Decision Records

This directory captures the non-obvious architectural decisions taken in AlphaMind. Each ADR is a short Markdown file describing the context, the decision, and the consequences.

## Format

We use the [MADR 3.0](https://adr.github.io/madr/) template. Files are numbered sequentially and kebab-cased: `NNNN-short-title.md`.

## When to write an ADR

Write one when a decision:

- Is hard to reverse (cloud provider, database engine, orchestration framework).
- Introduces a non-obvious trade-off that a future maintainer will question.
- Closes off alternatives that were seriously considered.

Trivial choices (formatting, naming, small library picks) do not need ADRs.

## Lifecycle

An ADR starts in `proposed` status, moves to `accepted` when merged, and may later become `deprecated` or `superseded` by another ADR. Never delete an ADR — supersede it, so the history is preserved.

## Index

- [0001 — Record architecture decisions](0001-record-architecture-decisions.md)
- [0002 — Use PostgreSQL with pgvector for documents and embeddings](0002-postgres-pgvector-for-retrieval.md)
- [0003 — SEC EDGAR ingestion design](0003-edgar-ingestion-design.md)
- [0004 — Storage layer for filing bodies](0004-storage-layer-for-filing-bodies.md)
- [0005 — Retrieval pipeline: chunker, embedder, hybrid search](0005-retrieval-pipeline-design.md)
- [0006 — LLM provider integration](0006-llm-provider-integration.md)
