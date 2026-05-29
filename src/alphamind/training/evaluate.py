"""Claim-extraction evaluation: matching + precision / recall / F1.

The metric functions are pure and fully tested. :func:`evaluate_model`
runs predictions through any ``LLMClient`` (the base model, the
fine-tuned ``LocalLLMClient``, or a stub in tests) and scores them
against the gold targets — so the same evaluator works for the
base-vs-fine-tuned comparison ADR 0011 calls for.

Claim matching is greedy one-to-one on token-overlap (Jaccard over
lowercased token sets) above a threshold. ADR 0011 explains the
choice: exact-string is too strict for free-form claims, embedding
similarity drags a model into the metric, token overlap is a
defensible, deterministic middle ground.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from alphamind.llm.base import LLMClient, LLMClientError
from alphamind.training.formatting import build_extraction_messages, parse_claims
from alphamind.training.types import Claim, TrainingExample

logger = logging.getLogger(__name__)

DEFAULT_MATCH_THRESHOLD = 0.5

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def jaccard(a: str, b: str) -> float:
    """Jaccard overlap of the token sets of two claim strings."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


@dataclass(frozen=True, slots=True)
class MatchResult:
    """The matched-pair count for one (predicted, gold) comparison."""

    matched: int
    n_predicted: int
    n_gold: int

    @property
    def precision(self) -> float:
        # No predictions → no false positives → vacuously perfect
        # precision (sklearn's zero_division=1 convention). The penalty
        # for a silent model lands on recall, not precision.
        if self.n_predicted == 0:
            return 1.0
        return self.matched / self.n_predicted

    @property
    def recall(self) -> float:
        if self.n_gold == 0:
            return 1.0  # nothing to recall; a correct empty answer scores 1.0
        return self.matched / self.n_gold

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)


def match_claims(
    predicted: Sequence[Claim],
    gold: Sequence[Claim],
    *,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> MatchResult:
    """Greedy one-to-one matching of predicted claims to gold claims.

    For each predicted claim, take the best-overlapping unused gold
    claim; if the overlap clears ``threshold`` it's a match and that
    gold claim is consumed. Greedy (not optimal assignment) is fine at
    the handful-of-claims-per-chunk scale and keeps the metric simple
    and deterministic.
    """
    used: set[int] = set()
    matched = 0
    for pred in predicted:
        best_idx = -1
        best_score = threshold
        for i, g in enumerate(gold):
            if i in used:
                continue
            score = jaccard(pred.text, g.text)
            if score >= best_score:
                best_score = score
                best_idx = i
        if best_idx >= 0:
            used.add(best_idx)
            matched += 1
    return MatchResult(matched=matched, n_predicted=len(predicted), n_gold=len(gold))


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Aggregate metrics over an evaluation set."""

    n_examples: int
    valid_json_rate: float
    precision: float
    recall: float
    f1: float


def aggregate(results: Sequence[MatchResult], *, n_valid_json: int, n_total: int) -> EvalReport:
    """Roll per-example match results into an :class:`EvalReport`.

    Precision / recall / F1 are micro-averaged (pool matched, predicted,
    gold across all examples) so a chunk with many claims weighs more
    than one with few — the right call for an extraction task.
    """
    total_matched = sum(r.matched for r in results)
    total_pred = sum(r.n_predicted for r in results)
    total_gold = sum(r.n_gold for r in results)

    precision = 1.0 if total_pred == 0 else total_matched / total_pred
    recall = 1.0 if total_gold == 0 else total_matched / total_gold
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    return EvalReport(
        n_examples=n_total,
        valid_json_rate=(n_valid_json / n_total) if n_total else 0.0,
        precision=precision,
        recall=recall,
        f1=f1,
    )


async def evaluate_model(
    examples: Sequence[TrainingExample],
    *,
    llm: LLMClient,
    model: str | None = None,
    max_tokens: int = 1024,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> EvalReport:
    """Run ``llm`` over the validation examples and score the predictions.

    A prediction that doesn't parse counts against ``valid_json_rate``
    and contributes zero matched claims (so it also drags precision /
    recall). A model that errors on a call is treated the same as an
    unparseable prediction — it produced no usable output.
    """
    results: list[MatchResult] = []
    n_valid_json = 0

    for ex in examples:
        messages = build_extraction_messages(ex.chunk_text)
        try:
            response = await llm.complete(messages, model=model, max_tokens=max_tokens)
        except LLMClientError as exc:
            logger.warning("chunk %s: candidate call failed: %s", ex.chunk_id, exc)
            results.append(MatchResult(matched=0, n_predicted=0, n_gold=len(ex.claims)))
            continue

        predicted = parse_claims(response.content)
        if predicted is None:
            results.append(MatchResult(matched=0, n_predicted=0, n_gold=len(ex.claims)))
            continue

        n_valid_json += 1
        results.append(match_claims(predicted, ex.claims, threshold=threshold))

    return aggregate(results, n_valid_json=n_valid_json, n_total=len(examples))


__all__ = [
    "DEFAULT_MATCH_THRESHOLD",
    "EvalReport",
    "MatchResult",
    "aggregate",
    "evaluate_model",
    "jaccard",
    "match_claims",
]
