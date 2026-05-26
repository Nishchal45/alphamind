"""FastAPI serving layer for the AlphaMind research DAG.

Endpoints:

- ``GET /healthz`` — liveness.
- ``POST /research`` — Server-Sent Events stream of the research run.

See :func:`alphamind.api.app.create_app` for the factory. The
production entry-point is::

    uvicorn alphamind.api.app:app --host 0.0.0.0 --port 8000

ADR 0009 documents the framework choice, streaming model, and what
the surface deliberately doesn't cover yet (auth, rate limits,
token-level streaming, readiness probe, cancellation).
"""

from __future__ import annotations

from alphamind.api.app import create_app

__all__ = ["create_app"]
