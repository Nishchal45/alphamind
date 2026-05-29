"""Tests for the fine-tune evaluator: matching + metrics + the run shell."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from alphamind.llm.base import LLMClientError, LLMResponse, Message
from alphamind.training.evaluate import (
    MatchResult,
    aggregate,
    evaluate_model,
    jaccard,
    match_claims,
)
from alphamind.training.types import Claim, TrainingExample


def _claims(*texts: str) -> tuple[Claim, ...]:
    return tuple(Claim(text=t) for t in texts)


# --- jaccard ---------------------------------------------------------------


def test_jaccard_identical() -> None:
    assert jaccard("revenue grew 12 percent", "revenue grew 12 percent") == 1.0


def test_jaccard_disjoint() -> None:
    assert jaccard("alpha beta", "gamma delta") == 0.0


def test_jaccard_partial() -> None:
    # {a,b,c} vs {b,c,d}: intersection 2, union 4 → 0.5
    assert jaccard("a b c", "b c d") == pytest.approx(0.5)


def test_jaccard_both_empty() -> None:
    assert jaccard("", "") == 1.0


def test_jaccard_one_empty() -> None:
    assert jaccard("something", "") == 0.0


# --- match_claims ----------------------------------------------------------


def test_match_perfect() -> None:
    pred = _claims("China is 17% of revenue", "Gross margin reached 73%")
    gold = _claims("Gross margin reached 73%", "China is 17% of revenue")
    result = match_claims(pred, gold)
    assert result.matched == 2
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0


def test_match_partial_recall() -> None:
    pred = _claims("China is 17% of revenue")
    gold = _claims("China is 17% of revenue", "Gross margin reached 73%")
    result = match_claims(pred, gold)
    assert result.matched == 1
    assert result.precision == 1.0
    assert result.recall == pytest.approx(0.5)


def test_match_partial_precision() -> None:
    pred = _claims("China is 17% of revenue", "totally unrelated fabricated claim xyz")
    gold = _claims("China is 17% of revenue")
    result = match_claims(pred, gold)
    assert result.matched == 1
    assert result.precision == pytest.approx(0.5)
    assert result.recall == 1.0


def test_match_one_to_one_no_double_counting() -> None:
    # Two near-identical predictions shouldn't both match a single gold.
    pred = _claims("revenue grew strongly", "revenue grew strongly")
    gold = _claims("revenue grew strongly")
    result = match_claims(pred, gold)
    assert result.matched == 1
    assert result.n_predicted == 2


def test_match_threshold_excludes_weak_overlap() -> None:
    pred = _claims("the company sells chips")
    gold = _claims("management discussed dividend policy at length")
    result = match_claims(pred, gold, threshold=0.5)
    assert result.matched == 0


def test_match_empty_prediction_against_empty_gold() -> None:
    result = match_claims((), ())
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0


def test_match_empty_prediction_against_nonempty_gold() -> None:
    result = match_claims((), _claims("a claim"))
    assert result.precision == 1.0  # nothing wrong emitted
    assert result.recall == 0.0  # but missed the gold


# --- aggregate -------------------------------------------------------------


def test_aggregate_micro_average() -> None:
    results = [
        MatchResult(matched=2, n_predicted=2, n_gold=2),
        MatchResult(matched=1, n_predicted=3, n_gold=2),
    ]
    report = aggregate(results, n_valid_json=2, n_total=2)
    # pooled: matched 3, pred 5, gold 4
    assert report.precision == pytest.approx(3 / 5)
    assert report.recall == pytest.approx(3 / 4)
    assert report.valid_json_rate == 1.0


def test_aggregate_valid_json_rate() -> None:
    results = [MatchResult(matched=0, n_predicted=0, n_gold=1)]
    report = aggregate(results, n_valid_json=0, n_total=1)
    assert report.valid_json_rate == 0.0


def test_aggregate_empty_set() -> None:
    report = aggregate([], n_valid_json=0, n_total=0)
    assert report.valid_json_rate == 0.0
    assert report.n_examples == 0


# --- evaluate_model (run shell) --------------------------------------------


@dataclass
class KeyedCandidate:
    """Candidate model stub keyed by chunk-text substring."""

    by_marker: dict[str, str] = field(default_factory=dict)
    raise_for: set[str] = field(default_factory=set)
    default_model: str = "candidate-stub"

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system: str | None = None,
    ) -> LLMResponse:
        user = messages[-1].content
        for marker in self.raise_for:
            if marker in user:
                raise LLMClientError("candidate boom")
        content = "{}"
        for marker, resp in self.by_marker.items():
            if marker in user:
                content = resp
                break
        return LLMResponse(
            content=content,
            model=model or self.default_model,
            input_tokens=5,
            output_tokens=5,
            stop_reason="end_turn",
        )


async def test_evaluate_model_perfect_predictions() -> None:
    examples = [
        TrainingExample(chunk_id=1, chunk_text="ALPHA", claims=_claims("Revenue grew 12 percent")),
    ]
    candidate = KeyedCandidate(
        by_marker={"ALPHA": '{"claims": [{"claim": "Revenue grew 12 percent"}]}'}
    )
    report = await evaluate_model(examples, llm=candidate)
    assert report.valid_json_rate == 1.0
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.f1 == 1.0


async def test_evaluate_model_unparseable_prediction_counts_against_json_rate() -> None:
    examples = [
        TrainingExample(chunk_id=1, chunk_text="ALPHA", claims=_claims("a claim")),
    ]
    candidate = KeyedCandidate(by_marker={"ALPHA": "the model rambled, no json here"})
    report = await evaluate_model(examples, llm=candidate)
    assert report.valid_json_rate == 0.0
    assert report.recall == 0.0  # missed the gold claim


async def test_evaluate_model_handles_candidate_errors() -> None:
    examples = [
        TrainingExample(chunk_id=1, chunk_text="EXPLODE", claims=_claims("a claim")),
    ]
    candidate = KeyedCandidate(raise_for={"EXPLODE"})
    report = await evaluate_model(examples, llm=candidate)
    assert report.valid_json_rate == 0.0
    assert report.recall == 0.0
