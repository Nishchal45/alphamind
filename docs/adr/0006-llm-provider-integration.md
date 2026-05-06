# 6. LLM provider integration

- **Status**: accepted
- **Date**: 2026-05-08

## Context

Phase 3 needs the agent layer to call a frontier LLM. Two design points
matter:

1. Agents shouldn't depend on a specific provider's SDK quirks. The
   router, specialists, synthesizer, and critic all need to call
   ``client.complete(messages)`` and not care whether the bytes go to
   Anthropic, OpenAI, or a self-hosted model.
2. Tests and offline development need to run end-to-end without an API
   key. CI can't hit a paid endpoint on every PR.

## Decision

A narrow ``LLMClient`` Protocol with two attributes:

- ``default_model: str``
- ``async complete(messages, *, model, max_tokens, temperature, system) -> LLMResponse``

Concrete implementations:

| Backend | Class | Use case |
| --- | --- | --- |
| Anthropic | ``AnthropicLLMClient`` | Production. Wraps ``anthropic.AsyncAnthropic``. |
| Echo | ``EchoLLMClient`` | Tests, offline dev. Returns ``"echo: <last user message>"``. |

The factory in ``alphamind.llm.factory`` reads ``LLM_BACKEND`` from
config and returns a singleton. ``LLM_BACKEND=echo`` is the default —
the project boots and runs end-to-end without an API key, same as the
deterministic embedder and reranker.

### Why a thin wrapper around the SDK

Two specific responsibilities live in the wrapper, not in agent code:

1. **Provider quirks.** Anthropic takes ``system`` as a top-level
   parameter, separate from the messages array. OpenAI folds it into
   the messages array. ``LLMClient.complete`` takes ``system`` as a
   keyword argument and each adapter handles the wiring. Without this,
   every agent node would have provider-specific branching.
2. **Retries and error mapping.** Tenacity wraps ``messages.create``
   with exponential backoff on the documented retryable errors
   (``RateLimitError``, ``APIConnectionError``, ``APITimeoutError``,
   ``InternalServerError``). Non-retryable ``APIError`` is wrapped in
   ``LLMClientError`` so call sites only need to catch one exception
   type.

### Why ``Message`` is a frozen dataclass and not a Pydantic model

Three reasons:

- The protocol is provider-agnostic; we want a transport that's
  trivially serialisable but not tied to a particular validator.
- Pydantic validation overhead matters when you're constructing
  thousands of messages per second (which the agent graph will, once
  it's parallel).
- A frozen dataclass is the simplest way to enforce immutability —
  agent code passes message lists around and shouldn't be mutating
  them mid-conversation.

### Why ``EchoLLMClient`` instead of mocking the API at every call site

Same reason ``DeterministicHashEmbedder`` exists. Tests need a
deterministic transport that satisfies the Protocol. Mocking
``anthropic.AsyncAnthropic`` everywhere bloats test setup and couples
test code to SDK shape. Echo is one ~30-line class that returns
predictable output and counts tokens crudely so the accounting code in
the agent graph still has something to track.

### What's not covered yet

- **Streaming.** ``messages.create(stream=True)`` returns an async
  iterator of events. Streaming matters when the FastAPI serving layer
  is in place (Phase 5). Out of scope for this PR.
- **Tool use.** Anthropic's tool-calling protocol is the natural way to
  let the agent layer query the retrieval pipeline. Lands when the
  agent graph itself does (next slice of Phase 3).
- **Cost tracking.** ``LLMResponse`` already carries ``input_tokens``
  and ``output_tokens``; an aggregator that sums them across a run
  lands when we have something to budget against.
- **Other providers.** OpenAI, Gemini, self-hosted via vLLM. The
  protocol allows them; nobody needs them yet, so they aren't built.

## Consequences

- Every agent node, every prompt template, every part of the synthesis
  layer depends on the Protocol, not on the SDK. Swapping providers is
  a config change.
- Tests stay fast and deterministic. CI doesn't pay for API calls.
- Offline development works — set ``LLM_BACKEND=echo`` and exercise
  the full agent graph without an API key.
- ``scripts/ask.py`` becomes the first end-to-end demo of the project:
  BM25 search → cited prompt → real LLM answer. With
  ``LLM_BACKEND=anthropic`` and a key set, this is the first thing in
  the project that does what AlphaMind is for.
