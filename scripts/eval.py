"""Run the eval harness against a golden set.

Usage:

    LLM_BACKEND=anthropic ANTHROPIC_API_KEY=sk-ant-... \
        uv run python scripts/eval.py \
            --golden-set evals/golden_set.yaml \
            --out evals/report.json

Wires the production research graph (HybridSearch + LLM factory)
behind the same runner that the unit tests exercise with stubs. The
output is a JSON report with per-case metrics and an aggregate
summary.

With the default ``LLM_BACKEND=echo`` every case will return an empty
thesis (echo can't produce structured JSON), so the report is only
useful as a plumbing smoke-test. For real scores, point at a real
backend.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import sys
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from alphamind.agents import build_research_graph
from alphamind.agents.state import Source
from alphamind.config import get_settings
from alphamind.db.session import dispose_engine, session_scope
from alphamind.eval import load_golden_set, run_eval
from alphamind.eval.types import EvalReport
from alphamind.llm.factory import get_llm_client
from alphamind.models.filing import Filing
from alphamind.models.filing_chunk import FilingChunk
from alphamind.retrieval.embeddings.factory import dispose_embedder, get_embedder
from alphamind.retrieval.search import HybridSearch, get_reranker
from sqlalchemy import select
from datetime import date

logger = logging.getLogger("alphamind.eval.cli")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden-set",
        default="evals/golden_set.yaml",
        type=Path,
        help="Path to the golden-set YAML (default: evals/golden_set.yaml).",
    )
    parser.add_argument(
        "--out",
        default="evals/report.json",
        type=Path,
        help="Where to write the JSON report (default: evals/report.json).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Top-k retrieval depth per specialist (default: 8).",
    )
    return parser.parse_args()


def _configure_logging() -> None:
    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


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
    return {
        row.id: (row.filing.company.ticker or row.filing.company.cik, row.filing.form)
        for row in rows
    }


def _build_retrieval_fn(search: HybridSearch) -> Any:
    async def retrieve(*, query: str, as_of: date, top_k: int) -> list[Source]:
        async with session_scope() as session:
            hits = await search.search(session, query=query, as_of=as_of, top_k=top_k)
            metadata = await _hydrate(session, [h.chunk_id for h in hits], as_of=as_of)
        return [
            Source(
                chunk_id=h.chunk_id,
                filing_id=h.filing_id,
                ticker=metadata.get(h.chunk_id, (str(h.filing_id), "—"))[0],
                form=metadata.get(h.chunk_id, (str(h.filing_id), "—"))[1],
                filing_date=h.filing_date,
                section=h.section,
                text=h.text,
                score=h.score,
            )
            for h in hits
        ]

    return retrieve


def _serialise(report: EvalReport) -> dict[str, Any]:
    """Turn the dataclass tree into JSON-safe primitives."""
    return {
        "n_cases": report.n_cases,
        "n_failed": report.n_failed,
        "aggregate": [dataclasses.asdict(m) for m in report.aggregate],
        "per_case": [dataclasses.asdict(c) for c in report.per_case],
    }


def _print_summary(report: EvalReport) -> None:
    print()
    print("Aggregate metrics")
    print("-----------------")
    for metric in report.aggregate:
        if metric.n == 0:
            print(f"  {metric.name:<22} (not measured: 0 cases supplied this hint)")
            continue
        print(
            f"  {metric.name:<22} mean={metric.mean:.3f}  "
            f"min={metric.minimum:.3f}  max={metric.maximum:.3f}  n={metric.n}"
        )
    print()
    print(f"  cases run:    {report.n_cases}")
    print(f"  cases failed: {report.n_failed}")


async def _run(args: argparse.Namespace) -> int:
    _configure_logging()

    cases = load_golden_set(args.golden_set)
    logger.info("loaded %d eval case(s) from %s", len(cases), args.golden_set)

    embedder = get_embedder()
    reranker = get_reranker()
    search = HybridSearch(embedder=embedder, reranker=reranker)
    retrieve = _build_retrieval_fn(search)
    client = get_llm_client()

    graph = build_research_graph(llm=client, retrieve=retrieve)
    report = await run_eval(graph, cases, top_k=args.top_k)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_serialise(report), indent=2, default=str), encoding="utf-8")
    logger.info("wrote eval report → %s", args.out)

    _print_summary(report)
    return 0 if report.n_failed == 0 else 1


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
