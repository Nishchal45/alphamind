"""Unit tests for EDGAR response schemas."""

from __future__ import annotations

from datetime import date

from alphamind.ingestion.edgar.schemas import (
    NormalizedFiling,
    SubmissionsResponse,
    iter_filings,
    parse_ticker_map,
)


def _apple_submissions_payload() -> dict[str, object]:
    return {
        "cik": 320193,
        "name": "Apple Inc.",
        "tickers": ["AAPL"],
        "exchanges": ["Nasdaq"],
        "sic": "3571",
        "sicDescription": "Electronic Computers",
        "filings": {
            "recent": {
                "accessionNumber": [
                    "0000320193-24-000123",
                    "0000320193-24-000101",
                ],
                "filingDate": ["2024-11-01", "2024-08-02"],
                "reportDate": ["2024-09-28", ""],
                "form": ["10-K", "10-Q"],
                "primaryDocument": [
                    "aapl-20240928.htm",
                    "aapl-20240629.htm",
                ],
                "primaryDocDescription": ["10-K", "10-Q"],
            }
        },
    }


def test_parse_submissions_response_normalises_cik_and_metadata() -> None:
    response = SubmissionsResponse.model_validate(_apple_submissions_payload())

    assert response.cik == "0000320193"
    assert response.name == "Apple Inc."
    assert response.primary_ticker == "AAPL"
    assert response.primary_exchange == "Nasdaq"
    assert response.sic == "3571"
    assert response.sic_description == "Electronic Computers"


def test_iter_filings_zips_parallel_arrays_and_handles_blank_report_date() -> None:
    response = SubmissionsResponse.model_validate(_apple_submissions_payload())

    filings = list(iter_filings(response))

    assert filings == [
        NormalizedFiling(
            accession_number="0000320193-24-000123",
            filing_date=date(2024, 11, 1),
            report_date=date(2024, 9, 28),
            form="10-K",
            primary_document="aapl-20240928.htm",
            primary_doc_description="10-K",
        ),
        NormalizedFiling(
            accession_number="0000320193-24-000101",
            filing_date=date(2024, 8, 2),
            report_date=None,
            form="10-Q",
            primary_document="aapl-20240629.htm",
            primary_doc_description="10-Q",
        ),
    ]


def test_parse_ticker_map_flattens_indexed_dict() -> None:
    payload = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
    }

    records = parse_ticker_map(payload)

    assert len(records) == 2
    assert records[0].ticker == "AAPL"
    assert records[0].cik == "0000320193"
    assert records[1].ticker == "MSFT"
    assert records[1].cik == "0000789019"


def test_unknown_fields_are_ignored() -> None:
    payload = _apple_submissions_payload()
    payload["unknownField"] = "ignore me"
    payload["filings"]["recent"]["unknownArray"] = []  # type: ignore[index]

    response = SubmissionsResponse.model_validate(payload)

    assert response.name == "Apple Inc."
