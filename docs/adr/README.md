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
