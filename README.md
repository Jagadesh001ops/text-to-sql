# Text-to-SQL: A 3B Model as a Production Service

> *How far can a 3B model go as a production text-to-SQL service when it's fine-tuned, checked against the database it queries, and served on one GPU, and what does each step cost in accuracy, latency, and dollars?*

1. **Prompting → fine-tuning.** **Qwen2.5-3B-Instruct**, zero-shot and few-shot, then fine-tuned with **QLoRA** on Spider.
2. **Execution feedback.** The generated SQL is run against the real database; failures trigger a retry, or several candidates are generated and voted on by their results.
3. **Serving.** **vLLM + FastAPI** on a single GPU, measured for latency, throughput and cost per 1K queries.

### Summary

On Spider dev (1,034 questions, execution accuracy):

- **Fine-tuning is the big step:** 55.9% zero-shot → **71.4%**. The best prompt-only setup I found (8 random examples, mean of 3 seeds) reaches 62.1% against 71.1% for the fine-tuned model on the same vLLM engine.
- **Checking against the database is a small, real step:** retrying failed queries gives +1.2 points (72.5%); voting over 8 candidates by execution result gives +2.9 (**74.3%**).
- **One 8 GB GPU can serve it:** 2.0 s p95 and about 17 queries/s at 16 concurrent requests, 59 GPU-seconds per 1,000 queries (about $0.04 at an assumed $2.50/GPU-hour).
- **Data needs:** about 1,000 training examples recover most of the gain (67.8%); 7,000 reach 71.2%. One of two 1,000-example runs collapsed into empty outputs for unknown reasons.
- Several pre-registered hypotheses were **not** supported; they are reported as such (see [Limitations](#limitations-and-how-to-read-these-numbers)).

---

## Results

**Exec Acc** = the predicted SQL returns the same result set as the gold SQL on the real SQLite database (Spider test-suite convention: duplicates count, row order matters only under a top-level `ORDER BY`, column order doesn't).

| Configuration | Exec Acc | Easy (248) | Medium (446) | Hard (172) | Extra (168) | Latency p95 | Cost / 1K queries |
|---|---|---|---|---|---|---|---|
| Zero-shot | 55.9% | 76.2% | 57.8% | 41.3% | 35.7% | — | — |
| 3-shot | 60.8% | 82.3% | 64.1% | 50.6% | 31.0% | — | — |
| **Fine-tuned (QLoRA)** | **71.4%** | 88.7% | 75.8% | 61.0% | 44.6% | 2.0 s | 59 GPU-s ($0.04) |
| + retry, error fed back (≤2) | 72.5% | 89.1% | 76.9% | 63.4% | 45.8% | 4.2 s | 82 GPU-s ($0.06) |
| + retry, resample (≤2) | 72.9% | 88.7% | 77.1% | 63.4% | 48.2% | — | — |
| **+ voting (8 candidates)** | **74.3%** | 88.3% | 79.4% | 65.7% | 48.8% | — | — |

All rows are Qwen2.5-3B-Instruct. Latency and cost are measured on the served system (vLLM + FastAPI, one 8 GB laptop GPU, 16 concurrent requests); a dash means that configuration was not served. Dollars use an *assumed* $2.50/GPU-hour; GPU-seconds is the price-independent figure. Tiers are computed from the gold SQL with Spider's hardness rules (`eval/hardness.py`).

- Fine-tuning gains +15.5 points over zero-shot in all tiers (+8.9 on extra-hard) and cuts the share of SQL that fails to run from 18–19% to 10.5%. Accuracy still falls steadily with query complexity.
- 3-shot prompting adds +4.9 points overall. Its 4.8-point drop on extra-hard is 8 of 168 questions, did not reproduce with other example sets, and is not reported as an effect.

**Setup:** greedy decoding, bf16, max 256 new tokens. QLoRA (4-bit NF4 base, LoRA r=16 on q/k/v/o, 7.4M trainable parameters) on Spider's 7,000 training examples, whose databases do not overlap dev; the adapter is merged into the bf16 base and prompted zero-shot. The shipped checkpoint is step 400, the lowest validation loss; the end-of-training checkpoint (step 1299) scores 69.3%. The fine-tuned model ran on Hazel's RTX 6000 Pro; retry and voting ran on a laptop RTX 5070 and reuse the greedy first answers (20 of 20 identical across the two GPUs).

### Prompting alone: how many examples, and which ones?

Base model, 0–16 examples in the prompt, chosen **randomly** or by a coverage-greedy **diverse** rule, 3 seeds per cell, one engine (vLLM). Overall accuracy, mean ± sd:

| Examples | 0 | 1 | 3 | 6 | 8 | 16 |
|---|---|---|---|---|---|---|
| random | 55.2 | 57.2 ± 1.9 | 59.7 ± 2.4 | 59.9 ± 2.1 | **62.1 ± 1.5** | 61.8 ± 2.4 |
| diverse | 55.2 | 55.9 ± 2.0 | 55.2 ± 4.4 | 55.7 ± 4.6 | 61.4 ± 2.2 | 61.0 ± 2.0 |

The original fixed 3-shot prompt scores 59.9 on this engine; the fine-tuned model scores **71.1**. Per-tier numbers: `results/fewshot_sweep.json`.

Hypotheses, pre-registered in [`docs/experiment_plan.md`](docs/experiment_plan.md):

- **H1.1** (rises to 3 examples, then flattens): supported for random (+4.5, then +0.2), not for diverse.
- **H1.2** (diverse beats random by ≥2 points on hard + extra): **not supported**; differences of +0.2, −0.7, +0.6 are inside the seed spread.
- **H1.3** (random 3-shot reproduces the extra-hard drop): **not supported**; random 3-shot did not drop.
- **H1.4** (random plateau persists to 8 and 16; added post-hoc, predicted first): **not supported as written** (+2.2 and +1.9 over 6, only 1.0–1.5 standard errors).
- **H1.5** (diverse converges to random as examples grow): **supported** (gap −4.5 at 3 examples, −0.8 at 16).

So more examples help slowly and stop around 8; the best single prompt (64.5%) is still about 7 points below fine-tuning, which gains roughly 2–3× more on every tier. The "diversity" idea was not supported. Three seeds detect only differences of roughly 4+ points.

### Execution feedback: retry and voting

The database acts as a verifier. Of the 1,034 first attempts, **109 (10.5%) failed to run**.

| | Exec Acc | Error rate | Extra generation / query* |
|---|---|---|---|
| Greedy (fine-tuned) | 71.4% | 10.5% | — |
| Retry, error fed back (≤2) | 72.5% (+1.2) | 6.7% | ~1.0 s |
| Retry, resample at T=0.7 (≤2) | 72.9% (+1.5) | 7.4% | ~1.0 s |
| **Vote over 8 candidates** | **74.3% (+2.9)** | **5.5%** | ~4.9 s |

\*Laptop RTX 5070, Hugging Face `generate`, one query at a time; one greedy answer takes about 3 s. Not served latencies. Retry and voting are single runs; the two retry styles differ by 4 questions and are not distinguishable.

- **Retry** gets 40 (feedback) or 33 (resample) of the 109 failures running, but only 12 and 16 of those become correct; about three quarters of the remaining errors run and return the wrong result, which retrying cannot see.
- **Voting** generates 8 candidates (greedy + 7 samples at T=0.7), executes them all and returns the result most candidates agree on (failed candidates get no vote; ties go to greedy). It changed 78 answers and came out 30 questions ahead, with the gain in medium (+3.6), hard (+4.7) and extra (+4.2); easy is flat. It costs about 5× the extra generation time of a retry for about twice the gain.

### Serving: latency, throughput and cost

Fine-tuned model (adapter merged, bf16) behind **vLLM 0.34 + FastAPI** on an **RTX 5070 Laptop GPU (8 GB)** in WSL2. All 1,034 dev questions per pass at a fixed number of requests in flight, after warm-up; no request failed.

| Requests in flight | Mode | Throughput | p50 | p95 | GPU-s / 1K | $ / 1K |
|---|---|---|---|---|---|---|
| 1 | generate only | 1.3 QPS | 0.56 s | 1.7 s | 757 | $0.53 |
| 4 | generate only | 5.6 QPS | 0.53 s | 1.6 s | 179 | $0.12 |
| **16** | **generate only** | **16.9 QPS** | 0.70 s | 2.0 s | **59** | **$0.04** |
| 64 | generate only | 17.5 QPS | 3.4 s | 4.9 s | 57 | $0.04 |
| 16 | execute + retry (≤2) | 12.2 QPS | 0.74 s | 4.2 s | 82 | $0.06 |

- **Served accuracy matches offline:** 71.1% generate-only and 72.6% with retries.
- **Batching is the biggest cost lever:** 16 in flight costs about 13× less GPU time per query than 1 in flight, with p95 barely moving.
- **The verifier** costs about 40% more GPU time and doubles p95 (2.0 → 4.2 s) for +1.6 points; only the ~11% of queries that fail are generated again, so the median is unchanged.
- **Throughput saturates near 16 in flight** because the 5.8 GiB model leaves only a 0.5 GiB KV cache on this card. The shape of the curve is the transferable result, not the absolute numbers.

**Setup:** `scripts/run_wsl_benchmark.sh` (vLLM `--enforce-eager`, `--max-num-seqs 16`, 512 MiB KV cache, 2,048-token context). One run per level. The GPU is shared with the Windows desktop. Voting and resampled retries were not served.

### Fine-tuning: how much training data?

QLoRA on random subsets of the 7,000 examples, same recipe; each model merged and evaluated zero-shot through vLLM, the same path for every row (the 7,000 model re-evaluated).

| Training examples | Easy | Medium | Hard | Extra | Overall | Empty outputs |
|---|---|---|---|---|---|---|
| 0 (base, zero-shot) | 75.8% | 56.7% | 44.2% | 32.1% | 55.2% | |
| 1,000 (subset seed 0) | 65.3% | 53.1% | 47.1% | 31.5% | **51.6%** | 22% |
| 1,000 (subset seed 1) | 85.9% | 72.7% | 55.8% | 40.5% | 67.8% | 0% |
| 3,000 | 86.7% | 70.6% | 55.8% | 40.5% | 67.1% | 0% |
| 7,000 | 88.7% | 75.8% | 60.5% | 44.1% | 71.2% | 0% |

- **The first 1,000-example run broke, the second did not.** The seed-0 model returns an empty answer (mostly runs of whitespace) for 225 of 1,034 questions; on the other 809 it scores 65.8%. Its best validation loss (0.266) equals the 3,000 model's, so loss-based selection did not catch it. It is not a bf16-merge artifact (the unmerged adapter on the 4-bit base is also empty, under a different generation path) and not simple under-training (later checkpoints, steps 155 and 186 of 187, still return 215 and 207 empties, against a pre-registered prediction of under 10%). The two subsets are statistically alike. **I did not find the cause.**
- **H2.1** (gain(1,000) ≥ 0.6 and gain(3,000) ≥ 0.85 of the 0→7,000 improvement): **not supported**; gain(3,000) is 0.75, and gain(1,000) is −0.23 (seed 0) or 0.79 (seed 1). **H2.2** (hard and extra gain more than easy from data beyond 1,000): **not supported.**
- **What I would claim:** about 1,000 well-behaved examples get most of the gain over prompting (67.8% vs 55.2%) and 7,000 add 3–4 points. The 1,000 vs 3,000 shape is unresolved, and small-data training looks unstable in this recipe.

**Setup:** `scripts/run_wsl_learning_curve.sh`, `run_wsl_followup.sh`, `run_wsl_ckpt_scan.sh`. The new models use micro-batch 2 × 8 (the 7,000 model used 4 × 4), the same effective batch of 16, learning rate, schedule and 3 epochs. Raw numbers: `results/learning_curve.json`, `results/learning_curve_followup.json`.

---

## Limitations and how to read these numbers

- **No held-out test set.** The dev set selects the checkpoint (by validation loss, not accuracy) and reports the result, so headline numbers are slightly optimistic.
- **Mostly single runs.** Retry, voting and training-size runs are one run each (the example-count sweep has 3 seeds per cell). Differences under about 1–1.5 points are noise; extra-hard is 168 questions, so one question is 0.6 points.
- **Two engines.** The main table uses Hugging Face generation; the example-count sweep, training-size runs and served numbers use vLLM, which reads the base model 0.7–1.0 points lower. Compare cells only within one engine (the 7,000 model: 71.4% Hugging Face, 71.1–71.2% vLLM).
- **Serving numbers are for this laptop.** 8 GB GPU shared with the desktop, 512 MiB KV cache, assumed GPU price. Voting and resampled retries were evaluated offline only.
- **One benchmark.** Spider is small SQLite databases with clean schemas; it says nothing about messy production schemas.
- **Post-hoc work is labelled.** The 8- and 16-shot runs, the 1,000-example reseed and the checkpoint scan were added after seeing earlier results; the plan records predictions made before each run (Addenda 2–6). Supported: H1.1 (partly) and H1.5. Not supported: H1.2–H1.4, H2.1, H2.2. An earlier explanation for the 3-shot extra-hard drop was withdrawn.
- **Unexplained failure.** The collapsed 1,000-example run (above) is unexplained; checkpoint selection by loss should not be trusted without an output check. The shipped 7,000 model has no empty outputs.

---

## Reproduce

```bash
pip install -r requirements.txt                 # GPU with 8+ GB VRAM for training/serving
python scripts/prepare_data.py                  # Spider -> data/processed/{train,dev}.jsonl
python eval/run_baseline.py --mode zero-shot    # or --mode few-shot --n-shot 3
python training/train.py                        # QLoRA; best checkpoint -> training/outputs/best_checkpoint
python eval/run_baseline.py --mode finetuned    # add --retries 2 or --vote-samples 8
python scripts/merge_adapter.py --adapter training/outputs/best_checkpoint --out training/outputs/merged
vllm serve training/outputs/merged --dtype bfloat16 --port 8001
uvicorn serving.app:app --port 8000             # POST /query {"question", "schema_text", "execute"}
python eval/benchmark.py --api-url http://localhost:8000
```

Cluster scripts for Hazel (Slurm) are in `hpc/`; the WSL2 pipelines behind the experiments are in `scripts/run_wsl_*.sh`. Details of the 8 GB vLLM settings are in the serving footnote above and in `scripts/run_wsl_benchmark.sh`.

```
configs/config.yaml     hyperparameters          eval/    scorer, hardness tiers, baseline/vLLM runners, benchmark
training/train.py       QLoRA fine-tuning        serving/ FastAPI + execution/retry loop, Dockerfile
scripts/                data prep, adapter merge, WSL pipelines        results/  tracked result summaries
docs/experiment_plan.md pre-registered experiments and addenda         hpc/      Hazel Slurm scripts
```

**Stack:** QLoRA via PEFT + TRL · Qwen2.5-3B-Instruct · vLLM · FastAPI · Docker · custom execution-accuracy harness (SQLite) · NC State Hazel HPC.
