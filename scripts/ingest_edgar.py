"""Ingest recent SEC filings for one or more tickers or CIKs.

Usage:

    uv run python scripts/ingest_edgar.py --ticker AAPL MSFT NVDA
    uv run python scripts/ingest_edgar.py --cik 320193 789019 --forms 10-K 10-Q
    uv run python scripts/ingest_edgar.py --ticker AAPL --limit 5
    uv run python scripts/ingest_edgar.py --ticker AAPL --with-bodies
    uv run python scripts/ingest_edgar.py --ticker AAPL --with-bodies --chunk
    uv run python scripts/ingest_edgar.py --ticker AAPL --chunk --embed

The script exits with code 1 if any ticker fails to resolve or any request
raises. Per-item errors are logged but do not abort sibling items so that
a single bad ticker does not kill an overnight backfill.

When ``--with-bodies`` is set, each filing's primary document is fetched
from EDGAR Archives, written to the configured storage backend, and
recorded in the ``filing_documents`` table. Bodies whose SHA-256 already
matches what is stored are skipped without rewriting.

When ``--chunk`` is set, every persisted filing document for each CIK is
read back from storage, split into retrieval-ready chunks, and inserted
into ``filing_chunks``. Documents whose stored chunks already match the
current body hash are skipped (use ``--force-chunk`` to re-chunk anyway).

When ``--embed`` is set, every chunk that is missing an embedding (or was
embedded under a different model) is encoded by the configured embedder
and written back. ``--embed`` implies ``--chunk``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass

from alphamind.chunking.service import ChunkBatchResult, chunk_bodies_for_cik
from alphamind.config import get_settings
from alphamind.db.session import dispose_engine
from alphamind.embeddings.factory import dispose_embedder, get_embedder
from alphamind.embeddings.service import EmbedBatchResult, embed_chunks_for_cik
from alphamind.ingestion.edgar.client import EdgarClient
from alphamind.ingestion.edgar.service import (
    BodyIngestResult,
    IngestResult,
    ingest_bodies_for_cik,
    ingest_cik,
    ingest_ticker,
)
from alphamind.storage.base import StorageBackend
from alphamind.storage.factory import get_storage

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
        help=(
            f"Form types to keep (default: {' '.join(DEFAULT_FORMS)}). "
            "Use 'ALL' to disable the filter."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Keep only the N most recent matching filings per company.",
    )
    parser.add_argument(
        "--with-bodies",
        action="store_true",
        help=(
            "After ingesting metadata, fetch each filing's primary document "
            "body and persist it via the storage backend."
        ),
    )
    parser.add_argument(
        "--chunk",
        action="store_true",
        help=(
            "After persisting bodies, chunk each filing document and write "
            "rows to filing_chunks. Implies --with-bodies."
        ),
    )
    parser.add_argument(
        "--force-chunk",
        action="store_true",
        help="Re-chunk filings even when stored chunks match the current hash.",
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help=(
            "After chunking, encode each chunk with the configured embedder "
            "and write vectors to filing_chunks. Implies --chunk."
        ),
    )
    parser.add_argument(
        "--force-embed",
        action="store_true",
        help="Re-embed chunks even when stored vectors match the current model.",
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


@dataclass
class _PipelineState:
    """Mutable accumulators for the per-CIK pipeline."""

    body_results: list[BodyIngestResult]
    chunk_results: list[ChunkBatchResult]
    embed_results: list[EmbedBatchResult]
    exit_code: int = 0


async def _process_bodies_and_chunks(
    *,
    client: EdgarClient,
    storage: StorageBackend | None,
    cik: str,
    label: str,
    forms: frozenset[str] | None,
    args: argparse.Namespace,
    state: _PipelineState,
) -> None:
    """Fetch and persist bodies for a CIK, then chunk them when requested.

    ``label`` is the user-facing identifier (ticker or raw CIK) used in log
    messages so failures can be traced back to the input that produced them.
    """

    if storage is None:
        return

    try:
        state.body_results.append(
            await ingest_bodies_for_cik(
                client,
                storage,
                cik,
                form_types=forms,
                limit=args.limit,
            )
        )
    except Exception:
        logger.exception("body ingest failed for %s cik=%s", label, cik)
        state.exit_code = 1
        return

    want_chunks = args.chunk or args.embed
    if not want_chunks:
        return

    try:
        state.chunk_results.append(
            await chunk_bodies_for_cik(
                storage,
                cik,
                form_types=forms,
                limit=args.limit,
                force=args.force_chunk,
            )
        )
    except Exception:
        logger.exception("chunking failed for %s cik=%s", label, cik)
        state.exit_code = 1
        return

    if not args.embed:
        return

    try:
        state.embed_results.append(
            await embed_chunks_for_cik(
                cik,
                embedder=get_embedder(),
                form_types=forms,
                limit=args.limit,
                force=args.force_embed,
            )
        )
    except Exception:
        logger.exception("embedding failed for %s cik=%s", label, cik)
        state.exit_code = 1


async def _run(args: argparse.Namespace) -> int:
    _configure_logging()
    forms = _build_form_filter(args.forms)

    # --chunk and --embed both require persisted bodies upstream.
    want_bodies = args.with_bodies or args.chunk or args.embed

    results: list[IngestResult] = []
    state = _PipelineState(body_results=[], chunk_results=[], embed_results=[])
    storage = get_storage() if want_bodies else None

    async with EdgarClient() as client:
        inputs = [(t, "ticker") for t in (args.ticker or [])]
        inputs += [(c, "cik") for c in (args.cik or [])]

        for value, kind in inputs:
            try:
                if kind == "ticker":
                    result = await ingest_ticker(client, value, form_types=forms, limit=args.limit)
                else:
                    result = await ingest_cik(client, value, form_types=forms, limit=args.limit)
                results.append(result)
            except Exception:
                logger.exception("ingest failed for %s=%s", kind, value)
                state.exit_code = 1
                continue

            await _process_bodies_and_chunks(
                client=client,
                storage=storage,
                cik=result.cik,
                label=f"{kind}={value}",
                forms=forms,
                args=args,
                state=state,
            )

    body_results = state.body_results
    chunk_results = state.chunk_results
    embed_results = state.embed_results
    exit_code = state.exit_code

    print()
    print(f"{'CIK':<12} {'TICKER':<8} {'SEEN':>6} {'WRITTEN':>8}  NAME")
    for r in results:
        print(
            f"{r.cik:<12} {(r.ticker or '-'):<8} "
            f"{r.filings_seen:>6} {r.filings_written:>8}  {r.name}"
        )

    if body_results:
        print()
        print(f"{'CIK':<12} {'BODIES_SEEN':>12} {'WRITTEN':>8} {'UNCHANGED':>10} {'FAILED':>7}")
        for b in body_results:
            print(
                f"{b.cik:<12} {b.bodies_seen:>12} {b.bodies_written:>8} "
                f"{b.bodies_unchanged:>10} {b.bodies_failed:>7}"
            )

    if chunk_results:
        print()
        print(
            f"{'CIK':<12} {'DOCS_SEEN':>10} {'CHUNKED':>8} "
            f"{'SKIPPED':>8} {'FAILED':>7} {'CHUNKS':>8}"
        )
        for c in chunk_results:
            print(
                f"{c.cik:<12} {c.documents_seen:>10} {c.documents_chunked:>8} "
                f"{c.documents_skipped:>8} {c.documents_failed:>7} {c.chunks_written:>8}"
            )

    if embed_results:
        print()
        print(f"{'CIK':<12} {'DOCS_SEEN':>10} {'FAILED':>7} {'EMBEDDED':>9} {'SKIPPED':>8}")
        for e in embed_results:
            print(
                f"{e.cik:<12} {e.documents_seen:>10} {e.documents_failed:>7} "
                f"{e.chunks_embedded:>9} {e.chunks_skipped:>8}"
            )

    return exit_code


async def _shutdown() -> None:
    """Close cached async resources before the event loop tears down."""

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
