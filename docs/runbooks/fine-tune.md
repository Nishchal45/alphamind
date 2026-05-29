# Runbook — fine-tuning the claim-extraction SLM

End-to-end procedure for distilling a training set, running the QLoRA
fine-tune, and evaluating the result. Design in
[ADR 0011](../adr/0011-slm-fine-tune.md) (the fine-tune) and
[ADR 0012](../adr/0012-local-llm-backend.md) (serving it).

> **This is a GPU procedure.** The repo ships the *pipeline*, not a
> trained checkpoint or eval numbers. The pure pieces (dataset
> building, formatting, metrics) run in CI; the `train()` loop and
> vLLM inference need a CUDA GPU and are run manually, here.

## The pipeline at a glance

```
ingested filing_chunks
   │  scripts/build_training_set.py   (frontier model = teacher)
   ▼
data/training/{train,val}.jsonl
   │  scripts/train_slm.py            (QLoRA, GPU, `train` extra)
   ▼
checkpoints/claim-extractor/adapter
   │  scripts/eval_slm.py             (LLM_BACKEND=local, GPU, `serve-local` extra)
   ▼
evals/slm_report.json   (precision / recall / F1 / valid-JSON rate)
```

## 1. Build the training set

Needs a frontier model as the teacher and ingested + chunked filings
in the DB. Runs anywhere (no GPU):

```bash
LLM_BACKEND=anthropic ANTHROPIC_API_KEY=sk-ant-... \
  uv run python scripts/build_training_set.py --limit 2000 --out-dir data/training
```

This distils each chunk into `{"claims": [...]}` targets, drops the
teacher's unparseable outputs, splits deterministically by hashed
chunk id, and writes `data/training/train.jsonl` and `val.jsonl`.

**Spot-check before training.** Open ~20 random lines of `train.jsonl`
and confirm the targets are sane — claims grounded in the chunk text,
no hallucinated numbers, empty lists only where the passage really is
boilerplate. The fine-tune can only be as good as the teacher's
labels; this is the cheapest place to catch a bad teacher prompt.

`data/training/` is gitignored — training sets are not committed.

## 2. Fine-tune (GPU)

On a CUDA box (a single 16 GB GPU is enough for the 1.5B base at
4-bit; 24 GB is comfortable):

```bash
uv sync --extra train
uv run python scripts/train_slm.py \
  --train data/training/train.jsonl \
  --val   data/training/val.jsonl \
  --output-dir checkpoints/claim-extractor
```

Defaults (ADR 0011): Qwen2.5-1.5B-Instruct base, 4-bit NF4 QLoRA,
rank 16 / alpha 32, 1 epoch. Overrides:

| Flag | Default | Notes |
| --- | --- | --- |
| `--base-model` | `Qwen/Qwen2.5-1.5B-Instruct` | Any HF causal-LM id |
| `--epochs` | `1.0` | Bump to 2-3 if val F1 is still climbing |
| `--lora-r` | `16` | Higher = more capacity, more VRAM |
| `--learning-rate` | `2e-4` | |
| `--no-4bit` | (off) | Full-precision base; needs far more VRAM |

The adapter lands in `checkpoints/claim-extractor/adapter`.
`checkpoints/` is gitignored.

If the `train` extra isn't installed the script exits with code 2 and
a clear message — it does not half-run.

## 3. Evaluate (GPU)

Score the fine-tuned model on the held-out split. The evaluator runs
through whatever `LLM_BACKEND` selects, so the *same command* scores
the base model, the fine-tuned model, or the frontier teacher — change
only the env:

```bash
uv sync --extra serve-local

# fine-tuned model
LLM_BACKEND=local SLM_ADAPTER_PATH=checkpoints/claim-extractor/adapter \
  uv run python scripts/eval_slm.py --val data/training/val.jsonl --out evals/slm_report.json

# untuned base (the baseline to beat)
LLM_BACKEND=local \
  uv run python scripts/eval_slm.py --val data/training/val.jsonl

# frontier teacher (an upper bound on this metric)
LLM_BACKEND=anthropic ANTHROPIC_API_KEY=sk-ant-... \
  uv run python scripts/eval_slm.py --val data/training/val.jsonl
```

The report carries four numbers (ADR 0011):

- **valid_json_rate** — structural floor; should be ~1.0 after a
  successful fine-tune.
- **precision** — of the claims emitted, how many match a gold claim.
- **recall** — of the gold claims, how many were recovered.
- **f1** — the headline number to track base-vs-fine-tuned.

The fine-tune is "working" when the fine-tuned model's F1 clears the
untuned base by a meaningful margin while keeping valid_json_rate at
~1.0 — i.e. it learned the domain behaviour, not just JSON formatting.

## 4. (Later) route the agents to the local model

ADR 0012's payoff — running the high-volume claim-extraction step on
the cheap local model while synthesis / criticism stay on the frontier
model — is a separate change. It's deliberately not wired here: do it
once the eval numbers justify the swap.

## Failure modes

| Symptom | Cause | Action |
| --- | --- | --- |
| `build_training_set`: "No chunks found" | Nothing ingested / chunked | Run the ingest + chunk pipeline first |
| Targets look wrong on spot-check | Teacher prompt or model too weak | Fix `EXTRACTION_SYSTEM`, rebuild; or use a stronger teacher |
| `train_slm` exits code 2 | `train` extra not installed | `uv sync --extra train` on a CUDA box |
| CUDA OOM during training | Batch / seq-len / rank too high for the GPU | Lower `--lora-r`, the batch size in `TrainConfig`, or `max_seq_length` |
| `eval_slm`: `LLMClientError: serve-local extra ...` | vLLM not installed | `uv sync --extra serve-local` |
| Fine-tuned F1 ≤ base F1 | Under-trained, bad data, or task saturated | More epochs, more / cleaner data, or accept the base is already good enough |
| valid_json_rate < 1.0 after training | Model drifting from the JSON schema | More epochs, or check the chat template renders the assistant turn correctly |
