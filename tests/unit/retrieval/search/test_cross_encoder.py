"""Unit tests for :class:`CrossEncoderReranker`.

The real model is heavy (~130 MB) and pulls torch, so unit tests stub
``sentence_transformers.CrossEncoder`` rather than load it. End-to-end
verification (real model, real GPU/CPU inference) belongs in an
integration test gated on the ``rerank`` extra.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from alphamind.retrieval.search.cross_encoder import (
    DEFAULT_MODEL,
    CrossEncoderReranker,
)
from alphamind.retrieval.search.rerank import RerankCandidate, RerankerError

pytestmark = pytest.mark.asyncio


class _FakeCrossEncoder:
    """Stand-in for sentence_transformers.CrossEncoder used in tests."""

    def __init__(self, model_name: str, *, scores: list[float] | None = None) -> None:
        self.model_name = model_name
        self.predict_calls: list[tuple[list[list[str]], int]] = []
        self._scores = scores

    def predict(self, pairs: list[list[str]], *, batch_size: int = 32) -> list[float]:
        self.predict_calls.append((pairs, batch_size))
        if self._scores is not None:
            if len(self._scores) != len(pairs):
                raise RuntimeError("test setup: scores must match pair count")
            return list(self._scores)
        # Default: score by query/text length ratio so order is predictable.
        return [float(len(t)) for _, t in pairs]


def _install_fake_module(monkeypatch: pytest.MonkeyPatch, ce_class: Any) -> None:
    """Make ``import sentence_transformers`` resolve to a fake module."""
    fake_mod = types.ModuleType("sentence_transformers")
    fake_mod.CrossEncoder = ce_class  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)


async def test_rerank_returns_scores_in_descending_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_module(monkeypatch, _FakeCrossEncoder)
    reranker = CrossEncoderReranker()

    candidates = [
        RerankCandidate(chunk_id=1, text="short"),
        RerankCandidate(chunk_id=2, text="a much longer passage here"),
        RerankCandidate(chunk_id=3, text="medium length"),
    ]

    hits = await reranker.rerank("query", candidates)

    # Default scoring uses len(text); chunk 2 wins, then 3, then 1.
    assert [h.chunk_id for h in hits] == [2, 3, 1]


async def test_rerank_preserves_chunk_id_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_scores = [0.1, 0.9, 0.5]
    _install_fake_module(
        monkeypatch,
        lambda model_name: _FakeCrossEncoder(model_name, scores=fake_scores),
    )
    reranker = CrossEncoderReranker()

    hits = await reranker.rerank(
        "q",
        [
            RerankCandidate(chunk_id=100, text="a"),
            RerankCandidate(chunk_id=200, text="b"),
            RerankCandidate(chunk_id=300, text="c"),
        ],
    )

    assert {h.chunk_id: h.score for h in hits} == {100: 0.1, 200: 0.9, 300: 0.5}
    # Top hit is chunk 200 (highest score).
    assert hits[0].chunk_id == 200


async def test_empty_candidates_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def factory(model_name: str) -> _FakeCrossEncoder:
        nonlocal called
        called = True
        return _FakeCrossEncoder(model_name)

    _install_fake_module(monkeypatch, factory)
    reranker = CrossEncoderReranker()

    hits = await reranker.rerank("q", [])

    assert hits == []
    # The model is never loaded for an empty input.
    assert called is False


async def test_model_loads_lazily_and_only_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances: list[_FakeCrossEncoder] = []

    def factory(model_name: str) -> _FakeCrossEncoder:
        inst = _FakeCrossEncoder(model_name)
        instances.append(inst)
        return inst

    _install_fake_module(monkeypatch, factory)
    reranker = CrossEncoderReranker(model_name="some/model")

    # Construction must NOT load the model.
    assert instances == []

    await reranker.rerank("q", [RerankCandidate(chunk_id=1, text="x")])
    await reranker.rerank("q", [RerankCandidate(chunk_id=2, text="y")])

    # Loaded once and cached across calls.
    assert len(instances) == 1
    assert instances[0].model_name == "some/model"
    # Two predict() calls.
    assert len(instances[0].predict_calls) == 2


async def test_batch_size_is_forwarded_to_predict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances: list[_FakeCrossEncoder] = []

    def factory(model_name: str) -> _FakeCrossEncoder:
        inst = _FakeCrossEncoder(model_name)
        instances.append(inst)
        return inst

    _install_fake_module(monkeypatch, factory)
    reranker = CrossEncoderReranker(batch_size=7)

    await reranker.rerank("q", [RerankCandidate(chunk_id=1, text="x")])

    assert instances[0].predict_calls[0][1] == 7


async def test_score_length_mismatch_raises_reranker_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class WrongLengthCE:
        def __init__(self, model_name: str) -> None: ...

        def predict(self, pairs: list[list[str]], *, batch_size: int = 32) -> list[float]:
            return [0.5]  # always one, regardless of input

    _install_fake_module(monkeypatch, WrongLengthCE)
    reranker = CrossEncoderReranker()

    with pytest.raises(RerankerError, match="returned 1 scores for 3 pairs"):
        await reranker.rerank(
            "q",
            [
                RerankCandidate(chunk_id=1, text="a"),
                RerankCandidate(chunk_id=2, text="b"),
                RerankCandidate(chunk_id=3, text="c"),
            ],
        )


async def test_predict_exception_becomes_reranker_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BoomCE:
        def __init__(self, model_name: str) -> None: ...

        def predict(self, pairs: list[list[str]], *, batch_size: int = 32) -> list[float]:
            raise RuntimeError("CUDA OOM")

    _install_fake_module(monkeypatch, BoomCE)
    reranker = CrossEncoderReranker()

    with pytest.raises(RerankerError, match="cross-encoder inference failed: CUDA OOM"):
        await reranker.rerank("q", [RerankCandidate(chunk_id=1, text="x")])


async def test_missing_sentence_transformers_raises_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate the extra not being installed: nuke any cached import and
    # ensure a fresh import attempt fails.
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    reranker = CrossEncoderReranker()

    with pytest.raises(RerankerError, match="uv sync --extra rerank"):
        await reranker.rerank("q", [RerankCandidate(chunk_id=1, text="x")])


async def test_default_model_matches_adr_pin() -> None:
    # ADR 0005 names ``ms-marco-MiniLM-L-12-v2`` as the default; the
    # constant is the source of truth used by both the class and the
    # factory.
    assert DEFAULT_MODEL == "cross-encoder/ms-marco-MiniLM-L-12-v2"


async def test_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        CrossEncoderReranker(batch_size=0)
