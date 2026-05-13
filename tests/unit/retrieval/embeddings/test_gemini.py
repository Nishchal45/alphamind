"""Unit tests for :class:`GeminiEmbedder`.

The embedder talks to Google's REST API; we mock the wire with respx so
no network call ever leaves the process. Tests cover the happy path,
exact request shape, batching beyond ``MAX_BATCH_SIZE``, retry on 429,
parsing failures, and the L2 re-normalisation after Matryoshka truncation.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable

import httpx
import pytest
import respx

from alphamind.retrieval.embeddings.base import EmbedderError
from alphamind.retrieval.embeddings.gemini import (
    DOCUMENT_TASK_TYPE,
    GEMINI_API_BASE,
    MAX_BATCH_SIZE,
    GeminiEmbedder,
)

pytestmark = pytest.mark.asyncio

_API_KEY = "test-key"
_MODEL = "gemini-embedding-001"
_URL = f"{GEMINI_API_BASE}/models/{_MODEL}:batchEmbedContents"


def _payload(*vectors: Iterable[float]) -> dict[str, object]:
    return {"embeddings": [{"values": list(v)} for v in vectors]}


def _norm(vec: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


async def test_embed_returns_vectors_in_input_order() -> None:
    embedder = GeminiEmbedder(api_key=_API_KEY, dim=3, rate=1000)
    try:
        async with respx.mock(assert_all_called=True) as mock:
            # Two unnormalised vectors; the embedder must normalise both.
            mock.post(_URL).respond(
                json=_payload([1.0, 0.0, 0.0], [0.0, 2.0, 0.0]),
            )

            vectors = await embedder.embed(["alpha", "beta"])

        # Order preserved; both unit-norm.
        assert len(vectors) == 2
        assert math.isclose(_norm(vectors[0]), 1.0, rel_tol=1e-6)
        assert math.isclose(_norm(vectors[1]), 1.0, rel_tol=1e-6)
        # Direction preserved (first basis vector stays on x-axis).
        assert vectors[0][0] > 0.99
    finally:
        await embedder.aclose()


async def test_embed_sends_api_key_and_correct_request_shape() -> None:
    embedder = GeminiEmbedder(api_key=_API_KEY, dim=4, rate=1000)
    try:
        async with respx.mock(assert_all_called=True) as mock:
            route = mock.post(_URL).respond(json=_payload([1.0, 0.0, 0.0, 0.0]))

            await embedder.embed(["hello"])

            request = route.calls.last.request
            assert request.url.params["key"] == _API_KEY
            body = json.loads(request.content)
            assert body == {
                "requests": [
                    {
                        "model": f"models/{_MODEL}",
                        "content": {"parts": [{"text": "hello"}]},
                        "taskType": DOCUMENT_TASK_TYPE,
                        "outputDimensionality": 4,
                    }
                ]
            }
    finally:
        await embedder.aclose()


async def test_embed_empty_input_does_not_call_api() -> None:
    embedder = GeminiEmbedder(api_key=_API_KEY, dim=3)
    try:
        async with respx.mock(assert_all_called=False) as mock:
            route = mock.post(_URL)
            vectors = await embedder.embed([])

        assert vectors == []
        assert route.call_count == 0
    finally:
        await embedder.aclose()


async def test_embed_slices_into_multiple_batches_when_over_max() -> None:
    n = MAX_BATCH_SIZE + 5
    embedder = GeminiEmbedder(api_key=_API_KEY, dim=2, rate=1000)
    try:
        async with respx.mock(assert_all_called=True) as mock:
            route = mock.post(_URL).mock(
                side_effect=[
                    httpx.Response(200, json=_payload(*[[1.0, 0.0]] * MAX_BATCH_SIZE)),
                    httpx.Response(200, json=_payload(*[[0.0, 1.0]] * 5)),
                ]
            )

            vectors = await embedder.embed([f"t{i}" for i in range(n)])

        assert route.call_count == 2
        assert len(vectors) == n
        assert math.isclose(_norm(vectors[0]), 1.0, rel_tol=1e-6)
        assert math.isclose(_norm(vectors[-1]), 1.0, rel_tol=1e-6)
    finally:
        await embedder.aclose()


async def test_embed_retries_on_429_then_succeeds() -> None:
    embedder = GeminiEmbedder(
        api_key=_API_KEY,
        dim=2,
        rate=1000,
        max_retries=3,
    )
    try:
        async with respx.mock(assert_all_called=True) as mock:
            route = mock.post(_URL).mock(
                side_effect=[
                    httpx.Response(429),
                    httpx.Response(200, json=_payload([0.5, 0.5])),
                ]
            )

            vectors = await embedder.embed(["please"])

        assert route.call_count == 2
        assert math.isclose(_norm(vectors[0]), 1.0, rel_tol=1e-6)
    finally:
        await embedder.aclose()


async def test_embed_raises_embedder_error_on_persistent_4xx() -> None:
    embedder = GeminiEmbedder(
        api_key=_API_KEY,
        dim=2,
        rate=1000,
        max_retries=2,
    )
    try:
        async with respx.mock(assert_all_called=True) as mock:
            mock.post(_URL).respond(status_code=400, text="bad request")

            with pytest.raises(EmbedderError, match="gemini embed failed"):
                await embedder.embed(["nope"])
    finally:
        await embedder.aclose()


async def test_embed_raises_embedder_error_when_response_shape_is_wrong() -> None:
    embedder = GeminiEmbedder(api_key=_API_KEY, dim=2, rate=1000)
    try:
        async with respx.mock(assert_all_called=True) as mock:
            # Asked for 2 vectors but Google returned 1.
            mock.post(_URL).respond(json={"embeddings": [{"values": [0.1, 0.0]}]})

            with pytest.raises(EmbedderError, match="expected 2 embeddings"):
                await embedder.embed(["a", "b"])
    finally:
        await embedder.aclose()


async def test_zero_norm_vector_is_returned_unchanged() -> None:
    """A pathological all-zeros vector cannot be normalised; leave it alone
    rather than dividing by zero."""
    embedder = GeminiEmbedder(api_key=_API_KEY, dim=3, rate=1000)
    try:
        async with respx.mock(assert_all_called=True) as mock:
            mock.post(_URL).respond(json=_payload([0.0, 0.0, 0.0]))

            vectors = await embedder.embed(["empty"])

        assert vectors == [[0.0, 0.0, 0.0]]
    finally:
        await embedder.aclose()


async def test_dim_property_matches_constructor() -> None:
    embedder = GeminiEmbedder(api_key=_API_KEY, dim=128)
    try:
        assert embedder.dim == 128
    finally:
        await embedder.aclose()


async def test_rejects_empty_api_key() -> None:
    with pytest.raises(ValueError, match="api_key must be non-empty"):
        GeminiEmbedder(api_key="", dim=2)


async def test_rejects_non_positive_dim() -> None:
    with pytest.raises(ValueError, match="dim must be positive"):
        GeminiEmbedder(api_key=_API_KEY, dim=0)
