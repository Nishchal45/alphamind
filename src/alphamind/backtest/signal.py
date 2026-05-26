"""Mechanical thesis-to-signal extraction.

The signal is a deterministic function of the thesis: it counts bull
and bear claims and returns a ratio in ``[-1, 1]``. See ADR 0010 for
why this is mechanical rather than LLM-rated — no extra leakage
surface, exact reproducibility, no extra cost.

The thresholding (signal → discrete position) is applied here too,
so the rest of the harness sees one consolidated ``Signal`` object
per thesis.
"""

from __future__ import annotations

from alphamind.agents.state import Thesis
from alphamind.backtest.types import Position, Signal

DEFAULT_THRESHOLD = 0.20


def extract_signal(
    thesis: Thesis | None,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> Signal:
    """Compute a :class:`Signal` from a synthesizer thesis.

    A ``None`` thesis (synthesizer never produced one) and an
    empty-thesis case (no claims either way) both resolve to a flat
    signal with score ``0.0``. That's how the rest of the harness
    distinguishes "the system had no view" from "the system was
    bullish or bearish."
    """

    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    if thesis is None:
        return Signal(score=0.0, position="flat", n_bull=0, n_bear=0)

    n_bull = len(thesis.bull_case)
    n_bear = len(thesis.bear_case)

    if n_bull == 0 and n_bear == 0:
        return Signal(score=0.0, position="flat", n_bull=0, n_bear=0)

    score = (n_bull - n_bear) / (n_bull + n_bear)
    position: Position
    if score > threshold:
        position = "long"
    elif score < -threshold:
        position = "short"
    else:
        position = "flat"

    return Signal(score=score, position=position, n_bull=n_bull, n_bear=n_bear)


__all__ = ["DEFAULT_THRESHOLD", "extract_signal"]
