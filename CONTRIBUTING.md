# Contributing to AlphaMind

Thanks for your interest. This document covers how to set up a development environment, the standards for code changes, and the review workflow.

## Development setup

1. Install [uv](https://docs.astral.sh/uv/) and Docker.
2. Clone the repository.
3. Run `make install` to provision the virtualenv and git hooks.
4. Run `make compose-up` to start Postgres and Redis.
5. Run `make test` to confirm the setup works.

## Branching and commits

- Base all work on `main`. Create feature branches named `<type>/<short-description>`:
  - `feat/hybrid-retrieval`
  - `fix/edgar-rate-limit`
  - `docs/evaluation-methodology`
- Write commit messages in the [Conventional Commits](https://www.conventionalcommits.org/) style.
  - `feat:` — user-facing feature
  - `fix:` — bug fix
  - `refactor:` — non-behaviour change
  - `perf:` — performance improvement
  - `docs:` — documentation only
  - `test:` — tests only
  - `build:` — build system or dependencies
  - `ci:` — CI configuration
  - `chore:` — everything else
- Keep commits atomic. Every commit should leave the tree in a working, green state.
- The commit-msg hook enforces the Conventional Commits format — it will reject non-conforming messages.

## Pull requests

- Open a draft PR early to signal direction.
- `make ci` must pass locally before marking the PR ready for review.
- Link related issues with `Closes #123` or `Relates to #456`.
- Update `CHANGELOG.md` under the `Unreleased` section for user-facing changes.
- PRs merge via squash. The squash message should read as a clean release-note line.

## Code style

- Python 3.11+. Formatting and linting are handled by `ruff`; type checking by `mypy` in strict mode. Configuration lives in [`pyproject.toml`](pyproject.toml).
- Prefer small, focused modules over kitchen-sink utilities.
- Tests: prefer integration tests for API routes, unit tests for pure logic. Write tests at the level that gives confidence, not for coverage's sake.
- Never commit secrets, API keys, or proprietary data. The expected configuration surface is documented in `.env.example`.

## Architecture decisions

Non-trivial design choices are captured as ADRs under [`docs/adr/`](docs/adr). If you are proposing a change that alters the architecture, open a PR with a new ADR before or alongside the implementation.

## Reporting issues

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.yml) or [feature request template](.github/ISSUE_TEMPLATE/feature_request.yml). Include reproduction steps and the commit SHA for bugs.
