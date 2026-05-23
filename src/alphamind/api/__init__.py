"""FastAPI serving layer for AlphaMind.

The CLI (`scripts/research.py`) is the developer entrypoint; this
package is the productised one. An ASGI app exposes:

- ``POST /research`` — runs the agent graph and streams progress as
  Server-Sent Events (one event per node completion). The synthesizer
  thesis arrives as a single event; token-level streaming requires
  growing :class:`alphamind.llm.base.LLMClient` and lands in a
  follow-up.
- ``GET /healthz`` — liveness, no I/O.
- ``GET /readyz`` — readiness; verifies Postgres is reachable.

Run locally with ``make serve`` or
``uv run uvicorn --factory alphamind.api.app:create_app``.
"""

from __future__ import annotations

from alphamind.api.app import create_app

__all__ = ["create_app"]
