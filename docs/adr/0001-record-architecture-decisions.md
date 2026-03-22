# 1. Record architecture decisions

- **Status**: accepted
- **Date**: 2026-04-22

## Context

AlphaMind will evolve through agent-topology changes, model swaps, retrieval-strategy shifts, and deployment migrations. Future contributors — and our future selves — need to understand why each choice was made. Tribal knowledge disappears; commit messages alone are too narrow to explain trade-offs.

## Decision

We adopt lightweight Architecture Decision Records, stored under `docs/adr/`, using the MADR 3.0 format. Every non-trivial architectural decision will be captured in a numbered ADR before or alongside its implementation. The PR template requires either a linked ADR or an explicit declaration that the change has no architectural impact.

## Consequences

- Reviewers have a canonical reference for why the codebase looks the way it does.
- New contributors can read the ADR log to ramp up quickly.
- Minor overhead for each significant change, offset by reduced relitigation of settled questions.
- Deprecated or superseded ADRs remain in the repository so the decision history is preserved.
