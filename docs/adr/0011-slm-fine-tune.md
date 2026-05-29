# 11. Fine-tuned SLM for claim extraction

- **Status**: accepted
- **Date**: 2026-05-27

## Context

The README has promised a fine-tuned small language model since day
one: *"a fine-tuned open-source SLM (LoRA / QLoRA) for the
domain-specific tasks; a frontier LLM for final synthesis."* Phase 4
makes that real.

The motivating problem is cost and latency, not capability. The
specialist nodes (ADR 0007) each make a frontier-model call to turn a
pool of filing chunks into structured findings. That's the highest-
volume LLM call in the system — it fires once per specialist per
question, over a large source pool. If a small local model can do the
*claim-extraction* step at acceptable quality, the frontier model is
reserved for synthesis and criticism, where judgement actually
matters.

Three design points needed pinning down:

1. What task to fine-tune for.
2. What base model and adapter method.
3. How to build the dataset, and how to evaluate the result.

This ADR is about the **design and the offline training
infrastructure**. The actual fine-tune run requires a GPU and is a
manual step documented in `docs/runbooks/fine-tune.md`; this repo
ships the pipeline that produces a checkpoint, not a checkpoint.

## Decision

### Task: per-chunk claim extraction

The fine-tune target is a single, sharply-scoped task:

> Given one filing chunk, output the material claims it contains as a
> JSON list of `{"claim": "<one sentence>"}` objects.

This is deliberately narrower than what a specialist does. A
specialist takes a *query* plus a *pool* of chunks and emits findings
that cite chunk ids across the pool. That cross-chunk citation logic
stays in the orchestration layer (and stays deterministic — ADR 0007).
The model only has to learn the local skill: read one passage of
financial prose, surface the claims worth extracting, emit valid
JSON. The citation is implicit — the claim came from the chunk it was
extracted from.

Narrowing the task this way has three payoffs:

- **Learnable on a small model.** A 1-2B model can learn "extract
  salient claims from this passage as JSON" far more reliably than
  "reason over a pool and assign citations."
- **Cheap to evaluate.** Claim-level precision / recall against a
  held-out set is well-defined (see below).
- **Drop-in.** The specialist's claim-extraction step becomes "call
  the local model per chunk, collect claims, attach the source chunk
  id." The orchestration contract (`Finding` with `cited_chunk_ids`)
  is unchanged.

### Base model: Qwen2.5-1.5B-Instruct

Chosen over Llama-3.2-1B-Instruct and Phi-3.5-mini primarily on
**license** and **size/quality trade-off**:

- **License.** Qwen2.5-1.5B ships under Apache-2.0. For a portfolio
  project that reads as production engineering, a permissive,
  no-strings license is worth more than a fraction of a benchmark
  point. Llama's community license carries acceptable-use and
  naming conditions that are noise in this context.
- **Size.** 1.5B params fits a QLoRA fine-tune on a single 16 GB GPU
  (Colab T4, with care) and an inference deployment on a 24 GB A10.
- **Instruction-tuned base.** The `-Instruct` variant already follows
  a chat template and emits JSON when asked, so the fine-tune is
  teaching *domain behaviour*, not *basic instruction-following from
  scratch*.

The base model is a config value (`SLM_BASE_MODEL`), so swapping to
Llama-3.2-1B or a larger Qwen is a settings change, not a code change.

### Adapter: QLoRA (4-bit base + LoRA)

QLoRA over full fine-tuning, for the obvious reasons — full
fine-tuning a 1.5B model needs far more VRAM than the project's target
hardware, and a domain-narrowing task like this does not need to move
every weight. LoRA config baked into the trainer defaults:

- 4-bit NF4 quantised base (bitsandbytes).
- LoRA rank `r=16`, `alpha=32`, dropout `0.05`.
- Target modules: the attention projections (`q_proj`, `k_proj`,
  `v_proj`, `o_proj`) plus the MLP projections (`gate_proj`,
  `up_proj`, `down_proj`).
- One epoch over the dataset to start; the runbook notes how to bump
  it.

Every one of these is a field on `TrainConfig` with the above as
defaults, so the runbook can override without touching code.

### Dataset: frontier-model distillation, spot-checked

The training data is **distilled** from the frontier model. The
dataset builder:

1. Pulls filing chunks (the same `filing_chunks` rows the retrieval
   layer indexes).
2. Runs each through the frontier model (`LLMClient`, ADR 0006) with
   the extraction prompt.
3. Captures the JSON output as the target for that chunk.

The frontier model is the teacher; the SLM is the student. This is
the standard distillation recipe and it's honest about what it is —
the SLM is learning to imitate the frontier model's claim extraction
cheaply, not to exceed it.

**Quality control.** The builder validates that every captured target
is parseable JSON in the expected schema and drops the ones that
aren't (a teacher mistake shouldn't become a training target). A
sample is spot-checked by hand — the runbook says how many and what
to look for. The dataset is versioned as JSONL on disk under
`data/training/` (gitignored, like the price cache and the storage
blobs).

Because the builder takes an `LLMClient`, it's testable end-to-end
against a scripted stub — no frontier API call in the test suite.

### Evaluation: claim-level precision / recall + valid-JSON rate

The fine-tuned model is scored on a held-out split with three numbers:

- **valid_json_rate** — fraction of predictions that parse as the
  expected schema. A structural floor; a model that can't emit JSON
  isn't usable downstream regardless of content quality.
- **precision** — of the claims the model emitted, how many match a
  gold claim.
- **recall** — of the gold claims, how many the model recovered.

Claim matching is **token-overlap above a threshold** (Jaccard over
lowercased token sets, default `0.5`). Exact-string matching is too
strict for free-form claims; embedding similarity would drag a model
dependency into the metric. Token overlap is a defensible middle
ground for a v1 and is a pure, deterministic, fully-tested function.

The F1 of precision and recall is the headline number the runbook
tells you to track across base-vs-fine-tuned and across epochs.

### The GPU boundary

What this repo ships:

- The dataset builder, runnable anywhere (needs the frontier API
  for real data; runs against a stub in tests).
- The trainer: LoRA/QLoRA config assembly + example formatting (pure,
  tested) wrapping a `train()` that lazily imports `torch`, `peft`,
  `trl`, `transformers`. Heavy imports are an optional `train` extra.
- The evaluator: claim-matching + P/R/F1 (pure, tested) plus a shell
  that runs predictions through any `LLMClient`.
- The local inference backend (ADR 0012).

What this repo does **not** ship, because it needs a GPU:

- Trained adapter weights.
- Real eval numbers.

Those are produced by running `make train-slm` / `make eval-slm` on
a GPU, documented step by step in the runbook. The CI suite exercises
every pure-logic path; the GPU path is a manual, reproducible
procedure.

## What's not covered yet

- **No RLHF / DPO.** Plain supervised fine-tuning on distilled
  targets. Preference tuning is a later concern, if ever.
- **No multi-task head.** One task (claim extraction). Sentiment
  scoring, section classification, etc. would each be their own
  fine-tune or a multi-task dataset — out of scope.
- **No quantisation of the merged adapter for deployment.** The
  serving path (ADR 0012) loads base + adapter; a merged-and-quantised
  artefact for faster cold start is a follow-up.
- **No active-learning loop.** The dataset is built once. Feeding the
  critic's rejections back as hard negatives is a compelling next
  step but not built.

## Consequences

- The specialist claim-extraction step gets a cheap local path while
  synthesis and criticism stay on the frontier model — the
  cost/quality split the README promised.
- The orchestration contract is unchanged: a fine-tuned model plugs
  in behind the same `LLMClient` Protocol (ADR 0012), selected by
  config. The agent graph doesn't know or care which model answered.
- Every pure-logic component (dataset building, formatting, metrics)
  is unit-tested with stubs. The GPU steps are thin, lazily-imported
  wrappers — the repo stays installable and testable without CUDA.
- Reproducing the fine-tune is a documented `make` sequence, not
  tribal knowledge. Anyone with a GPU and a frontier API key can
  rebuild the dataset, train, and evaluate.
