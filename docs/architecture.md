# Architecture

This document summarises the intended architecture of AlphaMind. For the reasoning behind individual decisions, see [`adr/`](adr).

## Component overview

```
┌──────────────┐   ┌──────────────┐   ┌─────────────────┐
│  Ingestion   │──▶│  Retrieval   │──▶│  Agent graph    │──▶ API / UI
│  (EDGAR,     │   │  (pgvector,  │   │  (LangGraph)    │
│   news,      │   │   BM25,      │   │                 │
│   market)    │   │   reranker)  │   │                 │
└──────────────┘   └──────────────┘   └─────────────────┘
         │                 │                    │
         ▼                 ▼                    ▼
    PostgreSQL         Embeddings         Evaluation +
    (primary)          + hybrid index     backtest harness
```

## Layers

### Ingestion
- SEC EDGAR adapters for 10-K, 10-Q, and 8-K filings.
- News and earnings-transcript adapters, each carrying source attribution and publication timestamp.
- Market data adapter covering historical prices and corporate actions.

### Storage
- PostgreSQL with the `pgvector` extension is the single source of truth for documents, chunks, embeddings, and metadata.
- Redis handles short-lived state: embedding and response caches, rate-limit counters, background-job coordination.

### Retrieval
- Hybrid search combining semantic (dense embeddings) and lexical (BM25) candidates.
- A cross-encoder reranker produces the final ordering.
- Time-aware filtering prevents lookahead bias during historical analysis — no chunk whose source post-dates the analysis horizon is returned.

### Agent graph
- Router → specialists (fundamentals, sentiment, technical, risk) → synthesizer → critic.
- Typed LangGraph state. Deterministic routing with LLM-based classification guarded by schema validation.

### Evaluation
- Citation coverage and hallucination rate on a labelled eval set.
- Historical backtest harness that fixes the information horizon and reports hit rate, Sharpe, and alpha versus SPY.
- Failures are preserved and surfaced in the README so regressions are visible.

### Serving
- FastAPI with async streaming endpoints.
- Model routing — a cheap model handles classification and formatting, a frontier model handles synthesis.
- OpenTelemetry traces spanning retrieval, agent calls, and token accounting.

## Target non-functional properties

- **Reproducibility**: given the same inputs and document snapshot, outputs are stable within a defined tolerance.
- **Auditability**: every claim in an agent's output is linked to retrievable source material.
- **Cost awareness**: token usage and inference cost are measured per request and budgeted per agent.
- **Safety**: guardrails inject disclaimers, block jailbreak attempts, and prevent financial-advice framing in responses.
