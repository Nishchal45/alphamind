"""Retrieval layer for AlphaMind.

Phase 2 of the project. Contains:

- :mod:`alphamind.retrieval.chunking` — turns filing bodies into searchable chunks.
- :mod:`alphamind.retrieval.embeddings` — encodes chunks as dense vectors.
- :mod:`alphamind.retrieval.search` — hybrid (dense + lexical) retrieval with
  cross-encoder reranking and a hard time-horizon filter that prevents
  lookahead bias during historical analysis.
"""
