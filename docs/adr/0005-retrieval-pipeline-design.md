# 5. Retrieval pipeline: chunker, embedder, hybrid search

- **Status**: accepted
- **Date**: 2026-05-04

## Context

Phase 1 delivers metadata + raw filing bodies. Phase 2 has to turn those
bodies into something the agent layer can query, with two non-negotiable
properties:

1. **No lookahead.** Backtests fix an analysis horizon; the retrieval
   layer must refuse to surface anything dated after that horizon. A
   chunk from a 2024-Q3 10-K cannot appear when the as-of date is
   2024-Q1, even by accident.
2. **Sane recall on financial prose.** Filings paraphrase heavily —
   "inventory write-down", "excess and obsolete reserve", "NRV
   adjustment" all mean roughly the same thing. Pure lexical search
   misses the synonyms; pure dense search promotes boilerplate. Hybrid
   is required.

## Decision

A four-stage pipeline, with each stage a swappable module:

| Stage | What it does |
| --- | --- |
| Chunker | HTML → plain text → section-aware token-bounded chunks |
| Embedder | Chunks → unit-norm dense vectors |
| Search | Query → BM25 + dense candidates → RRF → cross-encoder rerank → top-k |
| Time-horizon | `WHERE filing_date <= :as_of`, applied at every stage that touches the DB |

### Chunker

- HTML rendered via BeautifulSoup + lxml. Block-level tags become
  newlines, inline whitespace collapses, NFKC normalises smart quotes
  and non-breaking spaces.
- Section detection is regex over canonical 10-K / 10-Q Item headings.
  Filings without standard formatting fall back to a single
  ``"Preamble"`` span — chunking still works, section labels are just
  ``None``. The detector dedupes table-of-contents references by keeping
  the first occurrence in document order.
- Token counting uses tiktoken's ``cl100k_base`` vocabulary. Chunks are
  decoded back to character spans rather than token offsets, so the
  stored ``char_start`` / ``char_end`` survive future tokenizer changes.
- Default chunk size is 512 tokens with 15% overlap. Chunks below 64
  tokens are dropped — page numbers and signature lines pollute the
  index without adding signal.
- Chunks never cross section boundaries: each section is split
  independently, so the recorded ``section`` label is honest.

### Embedder

- Narrow protocol: ``dim`` + ``embed(texts) -> list[list[float]]``. Two
  attributes total. That's the entire surface area the embed service
  and search pipeline depend on.
- First implementation is :class:`DeterministicHashEmbedder`: SHA-256 of
  the input seeds a NumPy RNG, samples a Gaussian, L2-normalises. It
  is **not a real embedder** — same string maps to same vector, but
  semantically related strings don't end up close. It exists so the
  pipeline can be exercised end-to-end without a 130 MB model download.
- Real backend (``SentenceTransformerEmbedder`` over ``bge-small-en-v1.5``)
  lands in Phase 3. The factory in ``embeddings.factory`` swaps it in
  via ``EMBEDDING_BACKEND``; no other code changes.
- The embed service batches inputs (default 32) and is idempotent: a
  chunk that already has a non-null embedding is skipped unless the
  caller asks for ``force=True``.

### Search

The crown jewel and the place where lookahead bias can sneak in if
you're not careful. The pipeline:

1. Query → embedder → query vector.
2. Two branches in parallel: BM25 (``ts_rank_cd`` against the
   ``text_tsv`` GIN index) and dense (cosine via the HNSW index).
   Each branch applies ``filing_date <= :as_of`` in its own ``WHERE``
   so the planner can push the predicate under the index scan.
3. Reciprocal Rank Fusion. RRF was chosen over weighted score
   combination because BM25 scores and cosine similarities live on
   completely different scales — comparing them directly lets one
   branch dominate based on score magnitude rather than relevance.
   RRF only reads ranks, which is calibration-free.
4. Hydrate the top ``rerank_pool_size`` (default 25) candidates from
   the DB. ``HybridSearch._hydrate`` re-applies the time-horizon
   filter as a belt-and-suspenders check.
5. Cross-encoder rerank. The reranker is a Protocol; the first
   implementation is a Jaccard-overlap stub for tests. A real
   cross-encoder (``ms-marco-MiniLM-L-12-v2``) lands in Phase 3.
6. Truncate to ``top_k``.

### Time-horizon enforcement

Three layers. By design, redundant:

1. **Schema-level.** ``filing_chunks.filing_date`` is denormalised from
   ``filings`` and indexed. Every retrieval query filters on it.
2. **Query-level.** ``lexical_search`` and ``dense_search`` accept
   ``as_of`` as a required parameter. There is no default.
3. **Pipeline-level.** ``HybridSearch._hydrate`` re-applies
   ``filing_date <= :as_of`` when fetching candidate text. If a future
   code path bypasses the branch-level filters, this catches it.

Why three layers: lookahead is silent. A bug here doesn't blow up — it
fabricates alpha. Three independent enforcement points mean a single
mistake during refactoring doesn't break the invariant.

## Why not alternatives

### Why not a vector-DB-only approach (Qdrant, Weaviate)

Already covered in ADR 0002. Short version: time-horizon filtering wants
to be a SQL predicate that can be pushed under the ANN scan. Putting the
metadata in a separate system from the vector index turns that into a
cross-system join, which is operationally ugly and prevents the
predicate from short-circuiting the index.

### Why not langchain / llama-index for chunking

Both have finance-aware chunkers. Both also drag in transitive
dependency trees that are an order of magnitude larger than the chunker
itself. Three small modules (``text``, ``sections``, ``splitter``) +
``ChunkingPipeline`` is ~200 lines, easy to test, easy to extend.

### Why not BM25 only

Tried it on a small set of filings. Recall on paraphrased queries is
poor; "inventory write-down" misses chunks that talk about "excess and
obsolete reserves". Hybrid recovers those without giving up exact-term
matches.

### Why not dense only

Boilerplate disclaimers ("forward-looking statements") embed close to
many real questions because they share vocabulary. BM25 doesn't fall
for that — exact term frequency matters. Hybrid keeps the strengths of
both.

### Why RRF instead of weighted sum

Calibration. ``ts_rank_cd`` returns values typically in ``[0, 1]`` but
with very different distributions per query; cosine returns values in
``[-1, 1]``. Mixing them with weights would require per-query
normalisation. RRF avoids the problem entirely by ignoring score
magnitude.

## Consequences

- One pipeline serves both production retrieval (Phase 3 agent calls)
  and backtest replay (Phase 6 evaluation harness). The only
  difference is the ``as_of`` parameter.
- Adding new content types (earnings transcripts, news) means a new
  chunker for that source format and a row template in
  ``filing_chunks`` (or a sibling table). The search pipeline doesn't
  care.
- Swapping the embedder or reranker is a config change. The Protocols
  are the contract; concrete implementations are loaded by factory.
- The deterministic embedder + reranker mean the project boots and runs
  end-to-end without downloading a model. Real semantic quality
  obviously requires real models — that's Phase 3.
- ``filing_chunks`` is the table that grows fastest as the corpus
  expands. HNSW maintenance and tsvector reindexing dominate write
  cost. Acceptable at MVP scale; revisit when the chunk count crosses
  ~10 M.
