# 12. Local LLM backend (vLLM)

- **Status**: accepted
- **Date**: 2026-05-27

## Context

ADR 0011 fine-tunes a small model for claim extraction. To use that
model, the project needs a fourth `LLMClient` backend (alongside
`anthropic`, `echo`) that serves a local Hugging Face model + LoRA
adapter. ADR 0006 anticipated this: *"Other providers. OpenAI,
Gemini, self-hosted via vLLM. The protocol allows them; nobody needs
them yet, so they aren't built."* Now one is needed.

## Decision

### Serving engine: vLLM

vLLM over raw `transformers.generate`, TGI, or llama.cpp:

- **Throughput.** The claim-extraction step fires per-chunk over a
  large pool; vLLM's continuous batching is exactly the workload.
- **LoRA support.** vLLM serves a base model with a hot-swappable
  LoRA adapter, which is precisely the artefact ADR 0011 produces —
  no merge-and-reload step required.
- **OpenAI-compatible server option.** vLLM can run as a standalone
  server; the in-process `LLM` class is what we wrap here, but the
  server path is available for a future deployment split.

The trade-off is that vLLM is CUDA/Linux-centric and a heavy
dependency. It lives behind an optional `serve-local` extra, imported
lazily, exactly like `sentence-transformers` behind `rerank` (ADR
0005) — the project installs and tests without it.

### Shape: `LocalLLMClient` implementing the existing Protocol

`LocalLLMClient` satisfies the `LLMClient` Protocol (ADR 0006)
unchanged. It:

- Lazily constructs a vLLM `LLM` on first `complete()` (model load is
  expensive; don't pay it at import or construction).
- Renders the `Message` list through the base model's chat template
  (via the tokenizer) so the prompt matches what the model was
  fine-tuned on.
- Maps vLLM's output back into `LLMResponse`, populating
  `input_tokens` / `output_tokens` from the request output so the
  agent graph's usage accounting keeps working across backends.
- Raises `LLMClientError` (the one exception type the call sites
  already catch) when vLLM is absent or generation fails.

The prompt-rendering and response-mapping logic is pure and tested;
the model construction and the `generate()` call are the only parts
that need GPU + weights, and they're isolated behind the lazy import.

### Config + factory

Three settings:

- `llm_backend = "local"` selects it (joins `anthropic` / `echo` in
  the `LLMBackend` literal).
- `slm_base_model` — the Hugging Face base model id (default
  `Qwen/Qwen2.5-1.5B-Instruct`, matching ADR 0011).
- `slm_adapter_path` — filesystem path to the trained LoRA adapter.
  Optional: unset serves the base model untuned (useful for an
  apples-to-apples base-vs-fine-tuned eval).

The factory in `alphamind.llm.factory` gains the `local` branch,
mirroring the existing ones.

### Where it plugs in

Two consumers:

1. **The fine-tune evaluator** (ADR 0011) runs predictions through an
   `LLMClient`; pointing it at `LocalLLMClient` evaluates the trained
   model, pointing it at the base gives the baseline.
2. **The specialist claim-extraction step** (future): route per-chunk
   extraction to `local`, keep synthesis/critique on `anthropic`.
   That routing isn't wired in this ADR — it's a follow-up once the
   fine-tune has real eval numbers justifying the swap.

## What's not covered yet

- **No streaming.** Same as the other backends today (ADR 0006);
  lands with the serving layer's streaming work.
- **No tensor-parallel / multi-GPU config.** Single-GPU defaults.
  vLLM exposes the knobs; we don't surface them until needed.
- **No automatic base-vs-adapter routing.** The specialist still
  calls one configured client. Per-step model routing (cheap local
  for extraction, frontier for synthesis) is the payoff this backend
  enables but is deliberately a separate change.
- **No served-process mode.** We wrap the in-process `LLM`. The
  OpenAI-compatible server deployment is a Phase 5 concern.

## Consequences

- The agent graph gains a self-hosted model option with zero changes
  to node code — backend selection is config, the Protocol is the
  contract (ADR 0006 paying off exactly as designed).
- The project stays installable and CI-green without CUDA: the heavy
  path is an optional extra, lazily imported, with a clean error when
  absent.
- The cost-routing vision from the README (cheap local model for the
  high-volume step, frontier model for judgement) has its
  load-bearing piece in place. Wiring the per-step routing is the
  next increment.
