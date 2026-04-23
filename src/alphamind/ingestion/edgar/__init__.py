"""SEC EDGAR ingestion adapter.

Covers the public EDGAR JSON endpoints:

- ``/files/company_tickers.json``     — ticker-to-CIK mapping
- ``/submissions/CIK{cik}.json``      — recent filings metadata for a company

Document bodies (10-K / 10-Q / 8-K HTML or XBRL) are fetched from
``/Archives/edgar/data/{cik}/{accession}/...`` in a later phase.
"""

from alphamind.ingestion.edgar.client import (
    DEFAULT_RATE_PER_SECOND,
    EdgarClient,
    TokenBucketLimiter,
)
from alphamind.ingestion.edgar.schemas import (
    NormalizedFiling,
    RecentFilings,
    SubmissionsResponse,
    TickerRecord,
    iter_filings,
    parse_ticker_map,
)

__all__ = [
    "DEFAULT_RATE_PER_SECOND",
    "EdgarClient",
    "NormalizedFiling",
    "RecentFilings",
    "SubmissionsResponse",
    "TickerRecord",
    "TokenBucketLimiter",
    "iter_filings",
    "parse_ticker_map",
]
