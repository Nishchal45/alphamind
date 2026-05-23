"""Source-pool helpers.

When more than one specialist runs in a single DAG invocation, each
calls the retrieval entry-point independently and the ``sources``
accumulator on state ends up with duplicates. The synthesizer doesn't
notice (it only sees findings), but the critic and the CLI both render
the source pool — duplicates there would clutter the output and waste
the critic's context window. The helper here keeps the dedupe logic in
one place so both callers stay consistent.
"""

from __future__ import annotations

from collections.abc import Iterable

from alphamind.agents.state import Source


def dedupe_sources(sources: Iterable[Source]) -> list[Source]:
    """Return ``sources`` with duplicates removed, preserving first-seen order.

    Two sources are considered duplicates when they share a
    ``chunk_id`` — which is the join key everything else cites by.
    """
    seen: set[int] = set()
    out: list[Source] = []
    for src in sources:
        if src.chunk_id in seen:
            continue
        seen.add(src.chunk_id)
        out.append(src)
    return out


__all__ = ["dedupe_sources"]
