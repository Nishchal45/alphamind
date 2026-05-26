"""Run the full agent DAG against an ingested SEC corpus.

Usage:

    LLM_BACKEND=anthropic ANTHROPIC_API_KEY=sk-ant-... \
        uv run python scripts/research.py \
            --query "What is NVDA's exposure to China revenue concentration?" \
            --as-of 2024-12-31

The DAG runs router → fundamentals → synthesizer → critic and prints:

- the router's intent + rationale,
- the synthesized bull / bear thesis with chunk-id citations,
- the critic's flagged issues,
- the source pool (so citations can be traced back),
- aggregate token usage per node.

The ``--as-of`` time horizon is required. See ADR 0005 and ADR 0007 for
why backtests refuse to default it.

With the default ``LLM_BACKEND=echo`` every node will degrade to a
parse-failure path, so this CLI is most useful pointed at a real
backend. It still runs end-to-end with the echo client, which is the
point — useful for smoke-testing the wiring without API costs.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from typing import Any

from alphamind.agents import build_research_graph
from alphamind.agents._sources import dedupe_sources
from alphamind.agents.retrieval import build_filing_retrieval_fn
from alphamind.agents.state import Source, Thesis
from alphamind.config import get_settings
from alphamind.db.session import dispose_engine
from alphamind.llm.factory import get_llm_client
from alphamind.retrieval.embeddings.factory import dispose_embedder, get_embedder
from alphamind.retrieval.search import HybridSearch, get_reranker

logger = logging.getLogger("alphamind.research")

DEFAULT_TOP_K = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="The research question.")
    parser.add_argument(
        "--as-of",
        required=True,
        help="Time horizon (YYYY-MM-DD). No filings dated after this are used.",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Sources retrieved per specialist (default: {DEFAULT_TOP_K}).",
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
    print("Summary")
    print("-------")
    print(thesis.summary)
    print()
    print("Bull case")
    print("---------")
    if not thesis.bull_case:
        print("(none)")
    for claim in thesis.bull_case:
        cites = ",".join(str(c) for c in claim.cited_chunk_ids)
        print(f"  • {claim.claim}  [{cites}]")
    print()
    print("Bear case")
    print("---------")
    if not thesis.bear_case:
        print("(none)")
    for claim in thesis.bear_case:
        cites = ",".join(str(c) for c in claim.cited_chunk_ids)
        print(f"  • {claim.claim}  [{cites}]")


def _print_critique(state: dict[str, Any]) -> None:
    critique = state.get("critique")
    print()
    print("Critique")
    print("--------")
    if critique is None or (not critique.issues and critique.parse_error is None):
        print("(no issues)")
        return
    if critique.parse_error is not None:
        print(f"(critic parse error: {critique.parse_error!r})")
        return
    for issue in critique.issues:
        cites = ",".join(str(c) for c in issue.cited_chunk_ids) or "—"
        print(f"  • [{issue.kind}] {issue.claim}")
        print(f"      {issue.detail}  [{cites}]")


def _print_sources(sources: list[Source]) -> None:
    print()
    print("Sources")
    print("-------")
    if not sources:
        print("(none)")
        return
    for src in sources:
        section = src.section or "—"
        print(
            f"  [{src.chunk_id}] {src.ticker}  {src.form}  "
            f"{src.filing_date.isoformat()}  section={section!r}  score={src.score:.3f}"
        )


def _print_usage(state: dict[str, Any]) -> None:
    usage = state.get("usage") or []
    print()
    print("Usage")
    print("-----")
    if not usage:
        print("(none)")
        return
    total_in = sum(u.input_tokens for u in usage)
    total_out = sum(u.output_tokens for u in usage)
    for u in usage:
        print(f"  {u.node:<13} {u.model}  in={u.input_tokens}  out={u.output_tokens}")
    print(f"  {'total':<13} —  in={total_in}  out={total_out}")


async def _run(args: argparse.Namespace) -> int:
    _configure_logging()

    embedder = get_embedder()
    reranker = get_reranker()
    search = HybridSearch(embedder=embedder, reranker=reranker)
    retrieve = build_filing_retrieval_fn(search)
    client = get_llm_client()

    graph = build_research_graph(llm=client, retrieve=retrieve)
    result: dict[str, Any] = await graph.ainvoke(
        {
            "query": args.query,
            "as_of": args.as_of,
            "top_k": args.top_k,
        }
    )

    intent = result.get("intent")
    print("Intent")
    print("------")
    if intent is None:
        print("(router did not produce an intent)")
    else:
        print(f"  primary={intent.primary}")
        print(f"  specialists={', '.join(intent.specialists)}")
        print(f"  rationale={intent.rationale}")

    thesis = result.get("thesis")
    if thesis is None:
        print("(synthesizer did not produce a thesis)", file=sys.stderr)
        return 1

    _print_thesis(thesis)
    _print_critique(result)
    _print_sources(dedupe_sources(result.get("sources", [])))
    _print_usage(result)
    return 0


def main() -> None:
    args = parse_args()
    try:
        code = asyncio.run(_run(args))
    finally:
        asyncio.run(dispose_embedder())
        asyncio.run(dispose_engine())
    sys.exit(code)


if __name__ == "__main__":
    main()
