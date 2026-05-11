"""Ad-hoc hybrid-search CLI over the ingested filing corpus.

Usage:

    uv run python scripts/query.py "what is the bull case for nvidia gpus" --k 5
    uv run python scripts/query.py "supply chain risk" --form 10-K --as-of 2025-01-01
    uv run python scripts/query.py "revenue guidance" --cik 320193 --mode dense
    uv run python scripts/query.py "share buybacks" --mode bm25
    uv run python scripts/query.py "ai chip demand" --rerank

Results are printed in reading order with citation context — company,
form, accession number, section, and a short text preview. ``--mode``
defaults to ``hybrid``; pass ``dense`` or ``bm25`` to inspect either
retriever in isolation. ``--rerank`` runs the configured cross-encoder
over the candidate pool before printing.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import textwrap
from datetime import date

from alphamind.config import get_settings
from alphamind.db.session import dispose_engine, session_scope
from alphamind.embeddings.factory import dispose_embedder, get_embedder
from alphamind.reranking import dispose_reranker, get_reranker, rerank_results
from alphamind.retrieval import (
    DEFAULT_CANDIDATES,
    DEFAULT_TOP_K,
    RetrievalResult,
    bm25_search,
    dense_search,
    hybrid_search,
)

logger = logging.getLogger("alphamind.query")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Natural-language search query.")
    parser.add_argument(
        "--mode",
        choices=("hybrid", "dense", "bm25"),
        default="hybrid",
        help="Which retriever to run. Default: hybrid (RRF over dense + BM25).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of results to return. Default: {DEFAULT_TOP_K}.",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=DEFAULT_CANDIDATES,
        help=(f"Per-retriever candidate pool for hybrid mode. Default: {DEFAULT_CANDIDATES}."),
    )
    parser.add_argument(
        "--cik",
        default=None,
        help="Restrict to a single CIK (10-digit; leading zeros optional).",
    )
    parser.add_argument(
        "--form",
        nargs="+",
        default=None,
        help="Restrict to specific form types (e.g. 10-K 10-Q).",
    )
    parser.add_argument(
        "--section",
        nargs="+",
        default=None,
        help="Restrict to specific section labels (e.g. item_1a item_7).",
    )
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=None,
        help="Drop filings dated after this YYYY-MM-DD (for honest backtests).",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help=(
            "Rerank the candidate pool with the configured reranker before "
            "printing. Pulls more candidates than --k when used so the final "
            "ordering has something to reshuffle."
        ),
    )
    return parser.parse_args()


def _configure_logging() -> None:
    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _print_results(results: list[RetrievalResult], *, mode: str) -> None:
    if not results:
        print("(no results)")
        return

    print(f"\n{len(results)} result(s) [mode={mode}]\n")
    for i, r in enumerate(results, start=1):
        ranks = []
        if r.dense_rank is not None:
            ranks.append(f"dense#{r.dense_rank}")
        if r.bm25_rank is not None:
            ranks.append(f"bm25#{r.bm25_rank}")
        rank_str = f" ({', '.join(ranks)})" if ranks else ""

        header = (
            f"[{i}] {r.ticker or '-':<6} {r.form:<6} "
            f"{r.filing_date.isoformat()} {r.accession_number} "
            f"§ {r.section_label} score={r.score:.4f}{rank_str}"
        )
        print(header)
        preview = textwrap.shorten(r.text.replace("\n", " "), width=220, placeholder=" …")
        print(f"    {preview}\n")


async def _run(args: argparse.Namespace) -> int:
    _configure_logging()
    form_types = frozenset(args.form) if args.form else None
    section_labels = frozenset(args.section) if args.section else None

    # When reranking we want a wider candidate pool than ``--k`` so the
    # reranker has room to reorder — otherwise it can only confirm what
    # retrieval already returned.
    pool_size = max(args.k, args.candidates) if args.rerank else args.k

    async with session_scope() as session:
        if args.mode == "dense":
            results = await dense_search(
                session=session,
                embedder=get_embedder(),
                query=args.query,
                k=pool_size,
                as_of_date=args.as_of,
                cik=args.cik,
                form_types=form_types,
                section_labels=section_labels,
            )
        elif args.mode == "bm25":
            results = await bm25_search(
                session=session,
                query=args.query,
                k=pool_size,
                as_of_date=args.as_of,
                cik=args.cik,
                form_types=form_types,
                section_labels=section_labels,
            )
        else:
            results = await hybrid_search(
                session=session,
                embedder=get_embedder(),
                query=args.query,
                k=pool_size,
                dense_candidates=args.candidates,
                bm25_candidates=args.candidates,
                as_of_date=args.as_of,
                cik=args.cik,
                form_types=form_types,
                section_labels=section_labels,
            )

    if args.rerank:
        results = await rerank_results(
            reranker=get_reranker(),
            query=args.query,
            results=results,
            top_k=args.k,
        )

    _print_results(results, mode=args.mode)
    return 0


async def _shutdown() -> None:
    await dispose_reranker()
    await dispose_embedder()
    await dispose_engine()


def main() -> None:
    args = parse_args()
    try:
        code = asyncio.run(_run(args))
    finally:
        asyncio.run(_shutdown())
    sys.exit(code)


if __name__ == "__main__":
    main()
