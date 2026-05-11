"""Deterministic token-overlap reranker for tests and local development.

Like :class:`alphamind.embeddings.DeterministicEmbedder`, this is a real
:class:`Reranker` that produces stable, sensible-enough scores without a
model. The score is the Jaccard similarity of the query and passage token
sets — texts that share more tokens with the query win. This is enough
structure to exercise rerank plumbing while a real cross-encoder is set
up and to keep unit tests reproducible.
"""

from __future__ import annotations

import asyncio
import re

from alphamind.reranking.base import RerankedPassage

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class DeterministicReranker:
    """Jaccard-overlap reranker. Model-free; useful for tests."""

    @property
    def model_name(self) -> str:
        return "deterministic-jaccard"

    async def rerank(
        self,
        query: str,
        passages: list[tuple[int, str]],
    ) -> list[RerankedPassage]:
        return await asyncio.to_thread(self._rerank_sync, query, passages)

    def _rerank_sync(
        self,
        query: str,
        passages: list[tuple[int, str]],
    ) -> list[RerankedPassage]:
        query_tokens = set(_TOKEN_RE.findall(query.lower()))
        if not query_tokens:
            return [RerankedPassage(chunk_id=cid, score=0.0) for cid, _ in passages]

        scored: list[tuple[int, RerankedPassage]] = []
        for input_index, (chunk_id, text) in enumerate(passages):
            passage_tokens = set(_TOKEN_RE.findall(text.lower()))
            if not passage_tokens:
                score = 0.0
            else:
                intersection = len(query_tokens & passage_tokens)
                union = len(query_tokens | passage_tokens)
                score = intersection / union if union else 0.0
            scored.append((input_index, RerankedPassage(chunk_id=chunk_id, score=score)))

        # Sort by score desc, then by original input order so ties are
        # broken deterministically (matches the protocol contract).
        scored.sort(key=lambda pair: (-pair[1].score, pair[0]))
        return [pair[1] for pair in scored]


__all__ = ["DeterministicReranker"]
