"""Ingest recent SEC filings for one or more tickers or CIKs.

Usage:

    uv run python scripts/ingest_edgar.py --ticker AAPL MSFT NVDA
    uv run python scripts/ingest_edgar.py --cik 320193 789019 --forms 10-K 10-Q
    uv run python scripts/ingest_edgar.py --ticker AAPL --limit 5

The script exits with code 1 if any ticker fails to resolve or any request
raises. Per-item errors are logged but do not abort sibling items so that
a single bad ticker does not kill an overnight backfill.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from alphamind.config import get_settings
from alphamind.db.session import dispose_engine
from alphamind.ingestion.edgar.client import EdgarClient
from alphamind.ingestion.edgar.service import IngestResult, ingest_cik, ingest_ticker

logger = logging.getLogger("alphamind.ingest_edgar")

DEFAULT_FORMS = ("10-K", "10-Q", "8-K")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--ticker", nargs="+", help="Ticker symbols to ingest.")
    target.add_argument("--cik", nargs="+", help="CIK numbers to ingest.")
    parser.add_argument(
        "--forms",
        nargs="+",
        default=list(DEFAULT_FORMS),
        help=f"Form types to keep (default: {' '.join(DEFAULT_FORMS)}). Use 'ALL' to disable the filter.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Keep only the N most recent matching filings per company.",
    )
    return parser.parse_args()


def _configure_logging() -> None:
    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _build_form_filter(forms: list[str]) -> frozenset[str] | None:
    if any(f.upper() == "ALL" for f in forms):
        return None
    return frozenset(forms)


async def _run(args: argparse.Namespace) -> int:
    _configure_logging()
    forms = _build_form_filter(args.forms)

    results: list[IngestResult] = []
    exit_code = 0

    async with EdgarClient() as client:
        if args.ticker:
            for ticker in args.ticker:
                try:
                    result = await ingest_ticker(
                        client,
                        ticker,
                        form_types=forms,
                        limit=args.limit,
                    )
                    results.append(result)
                except Exception:
                    logger.exception("ingest failed for ticker=%s", ticker)
                    exit_code = 1
        elif args.cik:
            for cik in args.cik:
                try:
                    result = await ingest_cik(
                        client,
                        cik,
                        form_types=forms,
                        limit=args.limit,
                    )
                    results.append(result)
                except Exception:
                    logger.exception("ingest failed for cik=%s", cik)
                    exit_code = 1

    print()
    print(f"{'CIK':<12} {'TICKER':<8} {'SEEN':>6} {'WRITTEN':>8}  NAME")
    for r in results:
        print(
            f"{r.cik:<12} {(r.ticker or '-'):<8} "
            f"{r.filings_seen:>6} {r.filings_written:>8}  {r.name}"
        )
    return exit_code


def main() -> None:
    args = parse_args()
    try:
        code = asyncio.run(_run(args))
    finally:
        asyncio.run(dispose_engine())
    sys.exit(code)


if __name__ == "__main__":
    main()
