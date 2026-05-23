"""Specialist agents.

Each specialist is a LangGraph node that retrieves filtered evidence
and emits a small set of structured findings with chunk-level
citations. The synthesizer merges across specialists.

Today only :mod:`alphamind.agents.specialists.fundamentals` is wired in
to the graph; sentiment, technical, and risk specialists land in
follow-up PRs.
"""

from __future__ import annotations

from alphamind.agents.specialists.fundamentals import make_fundamentals_node

__all__ = ["make_fundamentals_node"]
