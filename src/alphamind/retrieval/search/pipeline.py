"""Hybrid search pipeline.

Wires together: embedder (for the dense branch), lexical search, dense
search, RRF fusion, candidate hydration, cross-encoder rerank, final
top-k truncation.

The time-horizon ``as_of`` parameter is required, not optional. Defaulting
it to ``date.today()`` would be footgun: a backtest at horizon 2023-06-01
that accidentally retrieves chunks from 2024 silently produces alpha that
isn't there. Forcing the caller to pass the horizon makes the contract
explicit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from alphamind.models.filing_chunk import FilingChunk
from alphamind.retrieval.embeddings.base import Embedder
from alphamind.retrieval.search.dense import dense_search
from alphamind.retrieval.search.fusion import reciprocal_rank_fusion
from alphamind.retrieval.search.lexical import lexical_search
from alphamind.retrieval.search.rerank import (
    RerankCandidate,
    Reranker,
)


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One result row returned to the caller."""

    chunk_id: int
    filing_id: int
    filing_date: date
    section: str | None
    text: str
    score: float


class HybridSearch:
    """End-to-end retrieval orchestrator: embed → BM25 + dense → RRF → rerank → top-k."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        reranker: Reranker,
        candidate_pool_size: int = 50,
        rrf_k: int = 60,
        rerank_pool_size: int = 25,
    ) -> None:
        if candidate_pool_size <= 0:
            raise ValueError("candidate_pool_size must be positive")
        if rerank_pool_size <= 0:
            raise ValueError("rerank_pool_size must be positive")
        if rerank_pool_size > candidate_pool_size:
            raise ValueError("rerank_pool_size cannot exceed candidate_pool_size")

        self._embedder = embedder
        self._reranker = reranker
        self._candidate_pool_size = candidate_pool_size
        self._rrf_k = rrf_k
        self._rerank_pool_size = rerank_pool_size

    async def search(
        self,
        session: AsyncSession,
        *,
        query: str,
        as_of: date,
        top_k: int = 10,
    ) -> list[SearchHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not query.strip():
            return []

        # 1. Run lexical and dense branches in series. Could be parallel,
        #    but each is one round-trip to Postgres and the rate limiter
        #    is on the embedder, not the DB — keep it simple.
        query_vectors = await self._embedder.embed([query])
        if not query_vectors:
            return []

        lexical_hits = await lexical_search(
            session,
            query=query,
            as_of=as_of,
            limit=self._candidate_pool_size,
        )
        dense_hits = await dense_search(
            session,
            query_vector=query_vectors[0],
            as_of=as_of,
            limit=self._candidate_pool_size,
        )

        # 2. Fuse ranked lists. Restrict the rerank input to the top
        #    ``rerank_pool_size`` candidates so the cross-encoder doesn't
        #    pay for tail material that won't make the cut anyway.
        fused = reciprocal_rank_fusion(
            [lexical_hits, dense_hits],
            k=self._rrf_k,
            limit=self._rerank_pool_size,
        )
        if not fused:
            return []

        # 3. Hydrate candidate text + filing metadata in one query.
        candidates = await self._hydrate(session, [h.chunk_id for h in fused], as_of)
        if not candidates:
            return []

        # 4. Rerank, then truncate to top_k.
        reranked = await self._reranker.rerank(
            query,
            [RerankCandidate(chunk_id=c.id, text=c.text) for c in candidates],
        )
        rerank_index = {hit.chunk_id: hit.score for hit in reranked}

        candidates_by_id = {c.id: c for c in candidates}
        ordered_ids = [hit.chunk_id for hit in reranked][:top_k]

        return [
            SearchHit(
                chunk_id=cid,
                filing_id=candidates_by_id[cid].filing_id,
                filing_date=candidates_by_id[cid].filing_date,
                section=candidates_by_id[cid].section,
                text=candidates_by_id[cid].text,
                score=rerank_index[cid],
            )
            for cid in ordered_ids
            if cid in candidates_by_id
        ]

    async def _hydrate(
        self,
        session: AsyncSession,
        chunk_ids: Sequence[int],
        as_of: date,
    ) -> list[FilingChunk]:
        if not chunk_ids:
            return []
        stmt = select(FilingChunk).where(
            FilingChunk.id.in_(chunk_ids),
            # Belt-and-suspenders: every branch already filters by
            # filing_date, but if a future code path bypasses that, this
            # guarantees the time-horizon invariant holds at the
            # pipeline level too.
            FilingChunk.filing_date <= as_of,
        )
        result = await session.execute(stmt)
        return list(result.scalars())
