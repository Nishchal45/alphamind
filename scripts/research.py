"""Run the multi-agent research graph end-to-end.

Usage:

    uv run python scripts/research.py \\
        --query "what is NVDA's exposure to China revenue concentration?" \\
        --as-of 2024-12-31

Pipeline (see ADR 0007):

1. router classifies the query and selects specialists.
2. each selected specialist retrieves chunks via HybridSearch (BM25 +
   pgvector + RRF + rerank), prompts an LLM with its domain rules, and
   returns cited claims.
3. synthesizer merges the specialist reports into a bull/bear/answer
   thesis with a single renumbered citation pool.
4. critic re-reads the thesis and flags unsupported claims.

``--as-of`` is required — same reason as ``scripts/ask.py`` (see ADR
0005): defaulting to today silently introduces lookahead bias in any
backtest that forgets to pass it.

Without provider keys set the project defaults to ``LLM_BACKEND=echo``
and ``EMBEDDING_BACKEND=deterministic`` — the graph still runs, but the
LLM and embeddings are stubs. Use real backends for real output:

    LLM_BACKEND=anthropic ANTHROPIC_API_KEY=sk-ant-... \\
    EMBEDDING_BACKEND=gemini GOOGLE_API_KEY=... \\
    RERANKER_BACKEND=cross_encoder \\
    uv run python scripts/research.py --query "..." --as-of 2024-12-31
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime

from alphamind.agents import GraphInput, get_research_graph
from alphamind.agents.state import Thesis
from alphamind.config import get_settings
from alphamind.db.session import dispose_engine
from alphamind.retrieval.embeddings.factory import dispose_embedder

logger = logging.getLogger("alphamind.research")

DEFAULT_TOP_K = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="The research question.")
    parser.add_argument(
        "--as-of",
        required=True,
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        help="Time horizon (YYYY-MM-DD). No filings dated after this are used.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Sources fed to each specialist (default: {DEFAULT_TOP_K}).",
    )
    return parser.parse_args()


def _configure_logging() -> None:
    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _print_thesis(thesis: Thesis) -> None:
    print()
    print("Answer")
    print("------")
    print(thesis.answer or "(empty)")

    if thesis.bull:
        print()
        print("Bull case")
        print("---------")
        for b in thesis.bull:
            print(f"- {b}")

    if thesis.bear:
        print()
        print("Bear case")
        print("---------")
        for b in thesis.bear:
            print(f"- {b}")


def _print_sources(thesis: Thesis) -> None:
    if not thesis.citations:
        return
    print()
    print("Sources")
    print("-------")
    for c in thesis.citations:
        section = c.section or "—"
        print(
            f"[{c.ordinal}] {c.ticker}  {c.form}  "
            f"{c.filing_date.isoformat()}  chunk={c.chunk_id}  section={section!r}"
        )


async def _run(args: argparse.Namespace) -> int:
    _configure_logging()

    graph = get_research_graph()
    state = await graph.invoke(
        GraphInput(query=args.query, as_of=args.as_of, top_k=args.top_k)
    )

    decision = state.get("router_decision")
    if decision is not None:
        print(
            f"router: specialists={list(decision.specialists)}  "
            f"rationale={decision.rationale!r}"
        )

    reports = state.get("specialist_reports", [])
    print(f"specialists produced {len(reports)} report(s)")
    for r in reports:
        print(f"  - {r.specialist}: {len(r.claims)} claim(s), {len(r.citations)} citation(s)")

    thesis = state.get("thesis")
    if thesis is None:
        print("no thesis produced", file=sys.stderr)
        return 1

    _print_thesis(thesis)
    _print_sources(thesis)

    critic = state.get("critic_report")
    if critic is not None:
        print()
        print("Critic")
        print("------")
        print(f"notes: {critic.notes or '(none)'}")
        if critic.unsupported:
            print(f"unsupported claims ({len(critic.unsupported)}):")
            for u in critic.unsupported:
                print(f"  - {u.claim}")
                print(f"      reason: {u.reason}")
        else:
            print("no unsupported claims flagged.")

    return 0


def main() -> None:
    args = parse_args()
    try:
        code = asyncio.run(_run(args))
    finally:
        # Clean up async resources owned by the singletons.
        asyncio.run(_cleanup())
    sys.exit(code)


async def _cleanup() -> None:
    await dispose_embedder()
    await dispose_engine()


if __name__ == "__main__":
    main()
