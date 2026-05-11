"""Unit tests for :class:`GeminiEmbedder`.

The embedder talks to Google's REST API; we mock the wire with respx so
no network call ever leaves the process. Tests cover the happy path, the
request shape Google expects, batching beyond ``MAX_BATCH_SIZE``, retry
on 429, and parsing failures.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

import httpx
import pytest
import respx

from alphamind.embeddings.base import EmbedderError
from alphamind.embeddings.gemini import (
    GEMINI_API_BASE,
    MAX_BATCH_SIZE,
    GeminiEmbedder,
)

pytestmark = pytest.mark.asyncio

_API_KEY = "test-key"
_MODEL = "text-embedding-004"
_URL = f"{GEMINI_API_BASE}/models/{_MODEL}:batchEmbedContents"


def _payload(*vectors: Iterable[float]) -> dict[str, object]:
    return {"embeddings": [{"values": list(v)} for v in vectors]}


async def test_embed_returns_vectors_in_input_order() -> None:
    embedder = GeminiEmbedder(api_key=_API_KEY, dimension=3, rate=1000)
    try:
        async with respx.mock(assert_all_called=True) as mock:
            mock.post(_URL).respond(
                json=_payload([0.1, 0.2, 0.3], [0.4, 0.5, 0.6]),
            )

            vectors = await embedder.embed(["alpha", "beta"])

        assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    finally:
        await embedder.aclose()


async def test_embed_sends_api_key_and_correct_request_shape() -> None:
    embedder = GeminiEmbedder(api_key=_API_KEY, dimension=3, rate=1000)
    try:
        async with respx.mock(assert_all_called=True) as mock:
            route = mock.post(_URL).respond(json=_payload([0.0, 0.0, 0.0]))

            await embedder.embed(["hello"])

            request = route.calls.last.request
            assert request.url.params["key"] == _API_KEY
            body = json.loads(request.content)
            assert body == {
                "requests": [
                    {
                        "model": f"models/{_MODEL}",
                        "content": {"parts": [{"text": "hello"}]},
                    }
                ]
            }
    finally:
        await embedder.aclose()


async def test_embed_empty_input_does_not_call_api() -> None:
    embedder = GeminiEmbedder(api_key=_API_KEY, dimension=3)
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
    embedder = GeminiEmbedder(api_key=_API_KEY, dimension=2, rate=1000)
    try:
        async with respx.mock(assert_all_called=True) as mock:
            route = mock.post(_URL).mock(
                side_effect=[
                    httpx.Response(200, json=_payload(*[[1.0, 1.0]] * MAX_BATCH_SIZE)),
                    httpx.Response(200, json=_payload(*[[2.0, 2.0]] * 5)),
                ]
            )

            vectors = await embedder.embed([f"t{i}" for i in range(n)])

        assert route.call_count == 2
        assert len(vectors) == n
        assert vectors[0] == [1.0, 1.0]
        assert vectors[-1] == [2.0, 2.0]
    finally:
        await embedder.aclose()


async def test_embed_retries_on_429_then_succeeds() -> None:
    embedder = GeminiEmbedder(
        api_key=_API_KEY,
        dimension=2,
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
        assert vectors == [[0.5, 0.5]]
    finally:
        await embedder.aclose()


async def test_embed_raises_embedder_error_on_persistent_4xx() -> None:
    embedder = GeminiEmbedder(
        api_key=_API_KEY,
        dimension=2,
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
    embedder = GeminiEmbedder(api_key=_API_KEY, dimension=2, rate=1000)
    try:
        async with respx.mock(assert_all_called=True) as mock:
            mock.post(_URL).respond(json={"embeddings": [{"values": [0.1]}]})

            # We asked for 2 vectors but Google returned 1.
            with pytest.raises(EmbedderError, match="expected 2 embeddings"):
                await embedder.embed(["a", "b"])
    finally:
        await embedder.aclose()


async def test_model_name_includes_model_path() -> None:
    embedder = GeminiEmbedder(api_key=_API_KEY, dimension=2)
    try:
        assert embedder.model_name == f"gemini:{_MODEL}"
        assert embedder.dimension == 2
    finally:
        await embedder.aclose()


async def test_rejects_empty_api_key() -> None:
    with pytest.raises(ValueError, match="api_key must be non-empty"):
        GeminiEmbedder(api_key="", dimension=2)


async def test_rejects_non_positive_dimension() -> None:
    with pytest.raises(ValueError, match="dimension must be positive"):
        GeminiEmbedder(api_key=_API_KEY, dimension=0)
