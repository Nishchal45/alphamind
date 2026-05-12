"""Ask a research question grounded in already-ingested SEC filings.

Usage:

    uv run python scripts/ask.py \
        --query "what is NVDA saying about China revenue concentration?" \
        --as-of 2024-12-31

The script runs a BM25 search over ``filing_chunks`` filtered by the
as-of date, builds a prompt that lists each retrieved chunk as a
numbered source, and calls the configured LLM. The model is instructed
to answer using only the supplied sources and to cite by number.

Defaults:

- 8 retrieved chunks (override with ``--top-k``).
- Date filter required — no default for the time horizon, on purpose
  (see ADR 0005). A backtest at horizon 2023-06 that accidentally
  retrieves chunks from 2024 silently produces alpha that didn't
  exist; making ``--as-of`` mandatory turns that footgun off.

To get real answers (not echoes), set:

    LLM_BACKEND=anthropic
    ANTHROPIC_API_KEY=sk-ant-...

The default ``LLM_BACKEND=echo`` exists so the rest of the pipeline can
be exercised without API costs.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import date, datetime

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from alphamind.config import get_settings
from alphamind.db.session import dispose_engine, session_scope
from alphamind.llm import LLMClientError, SystemMessage, UserMessage
from alphamind.llm.factory import get_llm_client
from alphamind.models.filing import Filing
from alphamind.models.filing_chunk import FilingChunk
from alphamind.retrieval.search.lexical import lexical_search

logger = logging.getLogger("alphamind.ask")

DEFAULT_TOP_K = 8

SYSTEM_PROMPT = """\
You are a research assistant for institutional equity analysis.

Rules:
1. Answer using ONLY the numbered sources provided. If the sources do
   not contain enough information, say so plainly.
2. Cite every claim with the source number in square brackets, e.g. [3].
   Multiple citations look like [1][4].
3. Do not speculate. Do not introduce facts that aren't in the sources.
4. Quote sparingly. Paraphrase in your own voice.
5. Output two sections: a short answer (2-4 sentences) and a longer
   reasoning section that walks through the key sources.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query",
        required=True,
        help="The research question.",
    )
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
        help=f"Number of source chunks to feed the LLM (default: {DEFAULT_TOP_K}).",
    )
    return parser.parse_args()


def _configure_logging() -> None:
    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


async def _hydrate_chunks(
    chunk_ids: list[int],
    *,
    as_of: date,
) -> list[FilingChunk]:
    """Load chunks + parent filings + companies in one query."""
    if not chunk_ids:
        return []
    async with session_scope() as session:
        stmt = (
            select(FilingChunk)
            .where(FilingChunk.id.in_(chunk_ids))
            .where(FilingChunk.filing_date <= as_of)
            .options(selectinload(FilingChunk.filing).selectinload(Filing.company))
        )
        result = await session.execute(stmt)
        chunks = list(result.scalars())
    # Preserve the lexical-search ranking.
    order = {cid: i for i, cid in enumerate(chunk_ids)}
    chunks.sort(key=lambda c: order.get(c.id, len(order)))
    return chunks


def _format_sources(chunks: list[FilingChunk]) -> str:
    """Build the [1]/[2]/... source block for the prompt."""
    lines: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        ticker = chunk.filing.company.ticker or chunk.filing.company.cik
        section = chunk.section or "—"
        header = (
            f"[{i}] {ticker} — {chunk.filing.form} — "
            f"{chunk.filing.filing_date.isoformat()} — {section}"
        )
        lines.append(header)
        lines.append(chunk.text)
        lines.append("")  # blank line between sources
    return "\n".join(lines).rstrip()


def _print_sources(chunks: list[FilingChunk]) -> None:
    print()
    print("Sources")
    print("-------")
    for i, chunk in enumerate(chunks, start=1):
        ticker = chunk.filing.company.ticker or chunk.filing.company.cik
        section = chunk.section or "—"
        print(
            f"[{i}] {ticker}  {chunk.filing.form}  "
            f"{chunk.filing.filing_date.isoformat()}  "
            f"accession={chunk.filing.accession_number}  section={section!r}"
        )


async def _run(args: argparse.Namespace) -> int:
    _configure_logging()

    async with session_scope() as session:
        hits = await lexical_search(
            session,
            query=args.query,
            as_of=args.as_of,
            limit=args.top_k,
        )

    if not hits:
        print(
            f"No chunks matched. Confirm filings are ingested + chunked, "
            f"and that filing_date <= {args.as_of} for at least one filing.",
            file=sys.stderr,
        )
        return 1

    chunks = await _hydrate_chunks([h.chunk_id for h in hits], as_of=args.as_of)
    if not chunks:
        print("Hydration returned no chunks (concurrent deletion?).", file=sys.stderr)
        return 1

    sources_block = _format_sources(chunks)
    prompt = (
        f"Question: {args.query}\n"
        f"As-of date: {args.as_of.isoformat()}\n\n"
        f"Sources:\n\n{sources_block}\n"
    )

    client = get_llm_client()
    try:
        response = await client.complete(
            [SystemMessage(SYSTEM_PROMPT), UserMessage(prompt)],
            max_tokens=1024,
        )
    except LLMClientError as exc:
        print(f"LLM call failed: {exc}", file=sys.stderr)
        return 2

    print(response.content.rstrip())
    _print_sources(chunks)
    print()
    print(
        f"[{response.model}  in={response.input_tokens}  out={response.output_tokens}  "
        f"stop={response.stop_reason}]"
    )
    return 0


def main() -> None:
    args = parse_args()
    try:
        code = asyncio.run(_run(args))
    finally:
        asyncio.run(dispose_engine())
    sys.exit(code)


if __name__ == "__main__":
    main()
