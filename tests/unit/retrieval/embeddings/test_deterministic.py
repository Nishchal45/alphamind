"""Tests for the deterministic hash-seeded embedder."""

from __future__ import annotations

import math

import pytest

from alphamind.models.filing_chunk import EMBEDDING_DIM
from alphamind.retrieval.embeddings.base import Embedder
from alphamind.retrieval.embeddings.deterministic import DeterministicHashEmbedder

pytestmark = pytest.mark.asyncio


async def test_satisfies_embedder_protocol() -> None:
    embedder = DeterministicHashEmbedder()
    assert isinstance(embedder, Embedder)


async def test_default_dim_matches_filing_chunk_column() -> None:
    embedder = DeterministicHashEmbedder()
    assert embedder.dim == EMBEDDING_DIM

    vectors = await embedder.embed(["hello world"])
    assert len(vectors) == 1
    assert len(vectors[0]) == EMBEDDING_DIM


async def test_vectors_are_unit_norm() -> None:
    embedder = DeterministicHashEmbedder()

    vectors = await embedder.embed(["alpha", "beta", "gamma delta"])

    for vec in vectors:
        norm = math.sqrt(sum(x * x for x in vec))
        assert math.isclose(norm, 1.0, abs_tol=1e-5)


async def test_same_text_yields_identical_vector() -> None:
    embedder = DeterministicHashEmbedder()

    a = await embedder.embed(["risk factors"])
    b = await embedder.embed(["risk factors"])

    assert a == b


async def test_different_text_yields_different_vector() -> None:
    embedder = DeterministicHashEmbedder()

    a, b = await embedder.embed(["risk factors", "management discussion"])

    assert a != b


async def test_preserves_input_order_in_batch() -> None:
    embedder = DeterministicHashEmbedder()

    inputs = ["one", "two", "three", "four"]
    vectors = await embedder.embed(inputs)

    # Hashing inputs individually should match.
    individual = []
    for text in inputs:
        result = await embedder.embed([text])
        individual.append(result[0])

    assert vectors == individual


async def test_rejects_non_positive_dim() -> None:
    with pytest.raises(ValueError):
        DeterministicHashEmbedder(dim=0)
    with pytest.raises(ValueError):
        DeterministicHashEmbedder(dim=-1)
