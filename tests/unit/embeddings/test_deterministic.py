"""Unit tests for the deterministic hash-based embedder.

These tests pin the contract the rest of the system relies on:

- Output dimension equals the declared dimension.
- Vectors are L2-normalised so cosine similarity is well defined.
- Encoding is stable across calls for identical inputs.
- Texts that share more tokens have higher cosine similarity than
  unrelated texts (weak but non-trivial structure).
"""

from __future__ import annotations

import math

import pytest

from alphamind.embeddings.deterministic import (
    DEFAULT_DIMENSION,
    DeterministicEmbedder,
)

pytestmark = pytest.mark.asyncio


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


async def test_dimension_and_model_name_match_declared_values() -> None:
    embedder = DeterministicEmbedder(dimension=64)

    assert embedder.dimension == 64
    assert embedder.model_name == "deterministic-hash-64"

    vectors = await embedder.embed(["hello world"])
    assert len(vectors) == 1
    assert len(vectors[0]) == 64


async def test_default_dimension_matches_module_constant() -> None:
    embedder = DeterministicEmbedder()

    assert embedder.dimension == DEFAULT_DIMENSION
    # Pinned to 768 to match Gemini's text-embedding-004 and the schema.
    assert DEFAULT_DIMENSION == 768


async def test_vectors_are_l2_normalised() -> None:
    embedder = DeterministicEmbedder(dimension=128)

    [vec] = await embedder.embed(["this is a non-trivial sentence with several words"])
    norm = math.sqrt(sum(v * v for v in vec))

    assert math.isclose(norm, 1.0, rel_tol=1e-9)


async def test_encoding_is_deterministic_across_calls() -> None:
    embedder = DeterministicEmbedder(dimension=64)

    [first] = await embedder.embed(["AlphaMind chunk text"])
    [second] = await embedder.embed(["AlphaMind chunk text"])

    assert first == second


async def test_order_is_preserved_within_a_batch() -> None:
    embedder = DeterministicEmbedder(dimension=32)

    texts = ["alpha", "beta", "gamma"]
    vectors = await embedder.embed(texts)

    # Compare against per-text encodings to confirm position is stable.
    individual = [(await embedder.embed([t]))[0] for t in texts]
    assert vectors == individual


async def test_shared_tokens_yield_higher_similarity_than_unrelated_text() -> None:
    embedder = DeterministicEmbedder(dimension=512)

    [a, b, c] = await embedder.embed(
        [
            "revenue from cloud services grew this quarter",
            "revenue from cloud services declined this quarter",
            "the cat sat quietly on the windowsill at dusk",
        ]
    )

    sim_ab = _cosine(a, b)  # heavy token overlap
    sim_ac = _cosine(a, c)  # essentially no overlap

    assert sim_ab > sim_ac


async def test_empty_text_yields_zero_vector() -> None:
    embedder = DeterministicEmbedder(dimension=32)

    [vec] = await embedder.embed([""])

    assert vec == [0.0] * 32


async def test_rejects_non_positive_dimension() -> None:
    with pytest.raises(ValueError, match="dimension must be positive"):
        DeterministicEmbedder(dimension=0)
