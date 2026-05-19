"""FastAPI serving layer for AlphaMind.

Phase 5 of the project. The CLI (`scripts/research.py`) is the
developer entrypoint; this package is the productised one. An ASGI
app exposes:

- ``POST /research`` — runs the agent graph and streams progress as
  Server-Sent Events (one event per node completion). The final
  thesis arrives as a single event; token-level streaming lands in
  a follow-up that extends the :class:`alphamind.llm.base.LLMClient`
  protocol.
- ``GET /healthz`` — liveness.
- ``GET /readyz`` — readiness; verifies Postgres is reachable.

Run locally with ``make serve`` or
``uv run uvicorn --factory alphamind.api.app:create_app``.
"""

from __future__ import annotations

from alphamind.api.app import create_app

__all__ = ["create_app"]
