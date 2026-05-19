"""Liveness and readiness probes.

``/healthz`` is liveness — it answers "is this process up?" and is
deliberately cheap (no I/O). Kubernetes / load balancers use this to
decide whether to restart the pod.

``/readyz`` is readiness — it answers "is this process able to serve
requests?" by checking Postgres reachability. A returning-from-cold
process should fail readiness until the DB connection works.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Response
from sqlalchemy import text

from alphamind.api.schemas import HealthResponse
from alphamind.db.session import session_scope

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse, summary="Liveness probe.")
async def healthz() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get(
    "/readyz",
    response_model=HealthResponse,
    summary="Readiness probe — verifies Postgres is reachable.",
)
async def readyz(response: Response) -> HealthResponse:
    try:
        async with session_scope() as session:
            await session.execute(text("SELECT 1"))
    except Exception as exc:
        logger.warning("readyz: database check failed: %s", exc)
        response.status_code = 503
        return HealthResponse(status="down", detail=f"database unreachable: {exc}")
    return HealthResponse(status="ok")


__all__ = ["router"]
