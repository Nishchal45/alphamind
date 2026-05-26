"""Run the backtest harness over a YAML universe.

Usage:

    LLM_BACKEND=anthropic ANTHROPIC_API_KEY=sk-ant-... make backtest

or directly:

    uv run python scripts/backtest.py \\
        --universe evals/backtest_universe.yaml \\
        --report-md  docs/eval/backtest.md \\
        --report-json evals/backtest_report.json \\
        --chart docs/eval/backtest_equity.png

Per ADR 0010, this harness is a sanity check that the system's
expressed view is correlated with subsequent price action. It is
not a measurement of tradable alpha. The report's first paragraph
restates these caveats so readers form an honest mental model
before they see any number.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import date
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from alphamind.agents import build_research_graph
from alphamind.agents.state import Source
from alphamind.backtest.loader import load_universe
from alphamind.backtest.prices import YahooPriceSource
from alphamind.backtest.report import write_chart, write_json, write_markdown
from alphamind.backtest.runner import run_backtest
from alphamind.config import get_settings
from alphamind.db.session import dispose_engine, session_scope
from alphamind.llm.factory import get_llm_client
from alphamind.models.filing import Filing
from alphamind.models.filing_chunk import FilingChunk
from alphamind.retrieval.embeddings.factory import dispose_embedder, get_embedder
from alphamind.retrieval.search import HybridSearch, get_reranker

logger = logging.getLogger("alphamind.backtest")

DEFAULT_CACHE_DIR = Path("data/prices")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--universe",
        required=True,
        type=Path,
        help="Path to the universe YAML.",
    )
    parser.add_argument(
        "--report-md",
        required=True,
        type=Path,
        help="Output path for the markdown report.",
    )
    parser.add_argument(
        "--report-json",
        required=True,
        type=Path,
        help="Output path for the JSON sidecar.",
    )
    parser.add_argument(
        "--chart",
        type=Path,
        default=None,
        help="Optional output path for the equity-curve PNG.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Directory for cached price CSVs (default: {DEFAULT_CACHE_DIR}).",
    )
    return parser.parse_args()


def _configure_logging() -> None:
    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# TODO: when PR #29 (FastAPI) lands, both this script and scripts/research.py
# can drop their inlined adapters and import build_filing_retrieval_fn from
# alphamind.agents.retrieval. The function below is the same shape as the one
# that PR introduces.
async def _hydrate(
    session: AsyncSession,
    chunk_ids: list[int],
    *,
    as_of: date,
) -> dict[int, tuple[str, str]]:
    if not chunk_ids:
        return {}
    stmt = (
        select(FilingChunk)
        .where(FilingChunk.id.in_(chunk_ids))
        .where(FilingChunk.filing_date <= as_of)
        .options(selectinload(FilingChunk.filing).selectinload(Filing.company))
    )
    rows = (await session.execute(stmt)).scalars().all()
    out: dict[int, tuple[str, str]] = {}
    for row in rows:
        ticker = row.filing.company.ticker or row.filing.company.cik
        out[row.id] = (ticker, row.filing.form)
    return out


def _build_retrieval_fn(search: HybridSearch):  # type: ignore[no-untyped-def]
    async def retrieve(*, query: str, as_of: date, top_k: int) -> list[Source]:
        async with session_scope() as session:
            hits = await search.search(session, query=query, as_of=as_of, top_k=top_k)
            metadata = await _hydrate(session, [h.chunk_id for h in hits], as_of=as_of)
        sources: list[Source] = []
        for hit in hits:
            ticker, form = metadata.get(hit.chunk_id, (str(hit.filing_id), "—"))
            sources.append(
                Source(
                    chunk_id=hit.chunk_id,
                    filing_id=hit.filing_id,
                    ticker=ticker,
                    form=form,
                    filing_date=hit.filing_date,
                    section=hit.section,
                    text=hit.text,
                    score=hit.score,
                )
            )
        return sources

    return retrieve


async def _run(args: argparse.Namespace) -> int:
    _configure_logging()

    universe = load_universe(args.universe)
    logger.info("loaded %d cases from %s", len(universe.cases), args.universe)

    embedder = get_embedder()
    reranker = get_reranker()
    search = HybridSearch(embedder=embedder, reranker=reranker)
    retrieve = _build_retrieval_fn(search)
    llm = get_llm_client()
    graph = build_research_graph(llm=llm, retrieve=retrieve)

    prices = YahooPriceSource(cache_dir=args.cache_dir)

    report = await run_backtest(universe, graph=graph, prices=prices)

    write_markdown(report, args.report_md, chart_path=args.chart)
    write_json(report, args.report_json)
    if args.chart is not None:
        write_chart(report, args.chart)

    summary = report.summary
    if summary is not None:
        logger.info(
            "backtest done: n_active=%d/%d hit_rate=%s mean_alpha=%s",
            summary.n_active,
            summary.n_cases,
            f"{summary.hit_rate:.3f}" if summary.hit_rate is not None else "n/a",
            f"{summary.mean_alpha:+.3%}" if summary.mean_alpha is not None else "n/a",
        )

    n_errors = sum(1 for r in report.results if r.error is not None)
    return 1 if n_errors == len(report.results) and report.results else 0


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
