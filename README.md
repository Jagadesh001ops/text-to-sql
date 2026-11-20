# Text-to-SQL: A 3B Model as a Production Service

> *How far can a 3B model go as a production text-to-SQL service when it's fine-tuned, checked against the database it queries, and served on one GPU, and what does each step cost in accuracy, latency, and dollars?*

Each step is treated as a system component that has to earn its place:

1. **Prompting → fine-tuning.** **Qwen2.5-3B-Instruct**, zero-shot and few-shot, then fine-tuned with **QLoRA** on Spider.
2. **Execution feedback.** The generated SQL is run against the real database; failures trigger a retry, or several candidates are generated and voted on by their results. The database acts as a verifier.
3. **Serving.** **vLLM + FastAPI** on a single GPU, measured for latency, throughput under concurrent load, and cost per 1K queries.

Every configuration is scored with an **execution-accuracy harness** against live SQLite databases, and reported with its cost, not just its accuracy.

### Summary

On Spider dev (1,034 questions, execution accuracy):

- **Fine-tuning is the big step:** 55.9% zero-shot → **71.4%** with QLoRA. The best prompt-only setup I found (8 random examples, mean of 3 seeds) reaches 62.1% against 71.1% for the fine-tuned model on the same vLLM engine, so more or better examples do not close the gap.
- **Checking against the database is a small, real step:** retrying failed queries gives +1.1 points (72.5%), voting over 8 candidates by execution result gives +2.9 (**74.3%**), at roughly 1.4× (retry) and several times (voting) the generation cost.
- **One 8 GB GPU can serve it:** 2.0 s p95 and about 17 queries/s at 16 concurrent requests, 59 GPU-seconds per 1,000 queries (about $0.04 at an assumed $2.50/GPU-hour).
- **Data needs:** about 1,000 training examples recover most of the gain (67.8%), 7,000 reach 71.2%. One of two 1,000-example runs collapsed into empty outputs for unknown reasons, so small-data training is unstable in this recipe.
- Several pre-registered hypotheses were **not** supported; they are reported as such (see [Limitations](#limitations-and-how-to-read-these-numbers)).

---

## Results

Spider dev set, all 1,034 examples. **Exec Acc** = the predicted SQL returns the same result set as the gold SQL when both are run against the real SQLite database. Comparison follows the Spider test-suite convention: row counts must match (duplicates count), row order matters only when the gold query has a top-level `ORDER BY`, and column order doesn't.

| Configuration | Exec Acc | Latency p95 | Cost / 1K queries |
|---|---|---|---|
| Zero-shot | 55.9% | — | — |
| 3-shot | 60.8% | — | — |
| **Fine-tuned (QLoRA)** | **71.4%** | 2.0 s | 59 GPU-s ($0.04) |
| Fine-tuned + execution retry (error fed back, ≤2 retries) | 72.5% | 4.2 s | 82 GPU-s ($0.06) |
| Fine-tuned + execution retry (resample, ≤2 retries) | 72.9% | — | — |
| **Fine-tuned + execution voting (8 candidates)** | **74.3%** | — | — |

All rows are Qwen2.5-3B-Instruct. **Latency p95 and cost are measured on the served system** (vLLM + FastAPI on one 8 GB laptop GPU) **at 16 concurrent requests**, the practical operating point there (details and caveats in [Serving](#serving-latency-throughput-and-cost)). Cost is GPU-seconds per 1,000 queries, with dollars at an *assumed* $2.50 per GPU-hour (a placeholder, not what this laptop costs). The retry row was measured through the API, where its accuracy is 72.6%. A dash means that configuration was not served; the voting row's cost is only measured offline (about 5× the extra generation time of a retry, see below).

### By difficulty

| Configuration | Easy (248) | Medium (446) | Hard (172) | Extra (168) | Overall |
|---|---|---|---|---|---|
| Zero-shot | 76.2% | 57.8% | 41.3% | 35.7% | 55.9% |
| 3-shot | 82.3% | 64.1% | 50.6% | 31.0% | 60.8% |
| **Fine-tuned (QLoRA)** | **88.7%** | **75.8%** | **61.0%** | **44.6%** | **71.4%** |
| Fine-tuned + retry (feedback) | 89.1% | 76.9% | 63.4% | 45.8% | 72.5% |
| Fine-tuned + retry (resample) | 88.7% | 77.1% | 63.4% | 48.2% | 72.9% |
| **Fine-tuned + voting (8 candidates)** | 88.3% | **79.4%** | **65.7%** | **48.8%** | **74.3%** |

- **Fine-tuning is the largest single step:** +15.5 points over zero-shot and +10.5 over 3-shot, with gains in every tier, including extra-hard (+8.9 over zero-shot), where 3-shot prompting had not helped. The execution error rate (SQL that fails to run) nearly halves, from 18–19% to 10.5%.
- Accuracy drops steadily with query complexity in every configuration; even after fine-tuning, about a quarter of extra-hard queries fail to execute (nested subqueries, set operations, multi-way joins).
- 3-shot prompting adds +4.9 points overall, concentrated in easy–hard (+6 to +9). Its 4.8-point drop on extra-hard queries is 8 of 168 questions, within sampling noise for a tier that size, and it did **not** reproduce with other example sets (see the next section), so it is not reported as an effect. (An earlier draft of this README explained the drop by the three fixed examples being simple single-table queries; the follow-up experiment does not support that explanation.)

### Why not just prompt better? Number and choice of examples

Pre-registered in [`docs/experiment_plan.md`](docs/experiment_plan.md) before running. Base model, 0 / 1 / 3 / 6 / 8 / 16 examples in the prompt (8 and 16 were added after seeing the first results, with their predictions registered before running), chosen either **randomly** or by a coverage-greedy **diverse** rule (maximise new SQL constructs, difficulty tiers and databases), 3 seeds per cell, all on the same engine (vLLM) and scorer. Mean ± standard deviation over seeds:

| Prompt | Overall | Easy | Medium | Hard | Extra |
|---|---|---|---|---|---|
| 0-shot | 55.2 | 75.8 | 56.7 | 44.2 | 32.1 |
| original fixed 3 | 59.9 | 83.5 | 62.8 | 47.1 | 30.4 |
| random 1 | 57.2 ± 1.9 | 75.4 | 59.6 | 48.5 | 32.9 |
| random 3 | 59.7 ± 2.4 | 77.7 | 63.9 | 48.3 | 33.9 |
| random 6 | 59.9 ± 2.1 | 81.5 | 63.7 | 44.0 | 34.3 |
| random 8 | 62.1 ± 1.5 | 82.0 | 66.2 | 49.6 | 34.5 |
| random 16 | 61.8 ± 2.4 | 81.2 | 65.5 | 49.0 | 36.3 |
| diverse 1 | 55.9 ± 2.0 | 73.5 | 57.6 | 47.1 | 34.7 |
| diverse 3 | 55.2 ± 4.4 | 72.9 | 56.7 | 45.9 | 34.9 |
| diverse 6 | 55.7 ± 4.6 | 74.5 | 57.5 | 42.4 | 37.1 |
| diverse 8 | 61.4 ± 2.2 | 82.0 | 62.9 | 50.2 | 38.5 |
| diverse 16 | 61.0 ± 2.0 | 82.3 | 63.5 | 50.2 | 33.9 |
| **Fine-tuned** (same engine) | **71.1** | **88.3** | **75.8** | **60.5** | **44.0** |

**Pre-registered hypotheses, as written:**

- **H1.1: accuracy rises from 0 to 3 examples, then flattens.** Supported for random examples (+4.5 points from 0 to 3, +0.2 from 3 to 6). *Not* supported for diverse ones, which gave no meaningful gain over 0-shot at any size (+0.7 points at most).
- **H1.2: at matched size, diverse beats random on hard + extra questions by ≥ 2 points.** **Not supported:** the differences are +0.2, −0.7 and +0.6 points at 1, 3 and 6 examples, all well inside the seed-to-seed spread (1–4 points).
- **H1.3: random 3-shot reproduces the extra-hard drop and diverse 3-shot reduces it.** **Not supported:** random 3-shot did not drop on extra-hard (33.9 vs 32.1 for 0-shot), so the original drop was not a general consequence of simple examples.
- **H1.4 (added after the first results, predicted before the 8- and 16-shot runs): the random plateau persists, i.e. 8 and 16 examples stay within 1.5 points of 6.** **Not supported as written:** random 8 is +2.2 and random 16 is +1.9 above random 6. Both differences are only 1.0–1.5 standard errors with 3 seeds, so they are not clearly real either, but the prediction missed its own threshold and the "plateau by 3" reading from the first results was premature.
- **H1.5 (same status): diverse selection converges toward random as examples grow.** **Supported:** the diverse − random gap is −4.5 points at 3 examples, −0.7 at 8 and −0.8 at 16.

**What this does show:**

- **More examples help slowly, then stop at about 8:** random-example accuracy goes 55.2 → 57.2 → 59.7 → 59.9 → 62.1 → 61.8 for 0 / 1 / 3 / 6 / 8 / 16 examples, and 16 adds nothing over 8 (−0.3). The best single prompt in the whole sweep scored 64.5%, still about 7 points below fine-tuning (71.1%); the best mean, 62.1%, is 9 points below.
- **Prompting helps the middle tiers most; fine-tuning helps everywhere.** Over 0-shot, the best random-example cells (8 and 16 examples) gain about +9 points on medium, +5 on hard and +4 on extra-hard, against +19, +16 and +12 for fine-tuning: roughly 2–3× more on every tier.
- **The "diversity" idea was not supported.** At 1–6 examples, diverse sets were no better than random (descriptively worse overall, 55–56% vs 57–60%, but with 3 seeds that is not a significant difference) and more variable (seed sd 4.4–4.6 vs 2.1–2.4). At 8 and 16 examples the two strategies are indistinguishable (gap under 1 point).

<sub>Limits: 3 seeds per cell can only detect differences of roughly 4+ points; the 8- and 16-shot runs were added after seeing the 1–6-shot results (predictions registered first, but the decision to look further was data-dependent); the vLLM engine reads the base model about 0.7–1.0 points below the Hugging Face runs in the main table (addendum in the plan), so compare cells within this table, not across it. Raw per-run results: `results/fewshot_sweep.json`.</sub>

### Execution feedback: retry when the SQL fails to run

The database is used as a verifier: if the generated query errors, the model is asked again, up to twice. Of the 1,034 first attempts, **109 (10.5%) failed to run**.

| Retry style | Failed queries now running | Newly correct | Exec Acc | Error rate |
|---|---|---|---|---|
| None (fine-tuned baseline) | — | — | 71.4% | 10.5% |
| Feedback: re-prompt with the failed SQL and the database error | 40 of 109 | +12 | 72.5% (+1.2) | 6.7% |
| Resample: draw a fresh answer to the original prompt (T=0.7) | 33 of 109 | +16 | 72.9% (+1.5) | 7.4% |

- Retrying adds **+1.2 to +1.5 points** and cuts the share of queries that fail to run by about a third. Only queries that failed are retried, so the cost is one or two extra generations for about 1 in 10 queries.
- The two styles are not distinguishable here: they differ by 4 questions, and resampling is random (one run, no repeats). Feedback gets more queries running, but only about 30% of those are also correct (12 of 40), against about half for resampling (16 of 33). A plausible reason is that the model never saw the `[ERROR]` re-prompt format during fine-tuning, so it tends to produce a query that runs but answers the wrong question.
- About three quarters of the remaining wrong answers are queries that run and return the wrong result (about 20% of all queries, against 7% that fail to run), which retrying cannot fix: nothing signals that they are wrong.

### Execution voting: sample several answers, keep the one the database agrees on

Retrying only helps when a query errors. Voting also catches queries that run but are wrong: generate 8 candidates per question (the greedy answer plus 7 samples at T=0.7), execute them all, and return the query whose *result* most candidates agree on. Candidates that fail to run get no vote, and ties go to the greedy answer.

| | Exec Acc | Error rate | Extra generation time per query* |
|---|---|---|---|
| No feedback (greedy) | 71.4% | 10.5% | — |
| Retry, error fed back | 72.5% (+1.2) | 6.7% | ~1.0 s |
| Retry, resample | 72.9% (+1.5) | 7.4% | ~1.0 s |
| **Vote over 8 candidates** | **74.3% (+2.9)** | **5.5%** | ~4.9 s |

<sub>*Average per query over the whole dev set, on a laptop RTX 5070 with Hugging Face `generate`, one query at a time. For reference, one greedy answer took about 3 s on the same machine (measured on 20 questions). These are not served-system latencies; the vLLM benchmark is what fills the latency and cost columns above.</sub>

- Voting changed the greedy answer for 78 of 1,034 questions (7.5%) and came out **30 questions ahead** overall. In 863 questions (83%) the winning result had at least 5 of the 8 votes, and in 654 (63%) all 8 candidates agreed. For 57 questions no candidate ran at all.
- The gain is in the harder tiers: medium +3.6, hard +4.7, extra +4.2 points. Easy is flat (−0.4, one question), where the greedy answer was already right.
- It costs about **5× the extra generation time of a retry for roughly twice the gain**, because all 8 candidates are generated for every question, not just the 1 in 10 that fail. Whether that trade is worth it depends on the latency and cost budget, which is what the serving benchmark is meant to answer.

### Serving: latency, throughput and cost

The fine-tuned model (adapter merged into the bf16 base) is served by **vLLM 0.34** behind a **FastAPI** layer on a single **RTX 5070 Laptop GPU (8 GB)** inside WSL2. Each pass sends all 1,034 dev questions at a fixed number of requests in flight, after a warm-up; none of the 1,034 × 6 requests failed.

**Generate only** (the API returns the SQL, nothing is executed):

| Requests in flight | Throughput | p50 | p95 | p99 | GPU-seconds / 1K queries | $ / 1K at $2.50/h |
|---|---|---|---|---|---|---|
| 1 | 1.3 QPS | 0.56 s | 1.7 s | 2.4 s | 757 | $0.53 |
| 4 | 5.6 QPS | 0.53 s | 1.6 s | 2.3 s | 179 | $0.12 |
| **16** | **16.9 QPS** | 0.70 s | 2.0 s | 3.0 s | **59** | **$0.04** |
| 64 | 17.5 QPS | 3.4 s | 4.9 s | 5.7 s | 57 | $0.04 |

**With the database as verifier** (the API executes the SQL read-only and retries up to twice with the error fed back):

| Requests in flight | Throughput | p50 | p95 | p99 | Retried | GPU-seconds / 1K queries | $ / 1K |
|---|---|---|---|---|---|---|---|
| 1 | 1.0 QPS | 0.59 s | 3.2 s | 5.2 s | 11.1% | 998 | $0.69 |
| 16 | 12.2 QPS | 0.74 s | 4.2 s | 6.6 s | 11.0% | 82 | $0.06 |

- **Served accuracy matches the offline numbers:** 71.1% generate-only (offline 71.4%) and 72.6% with retries (offline 72.5%), i.e. serving with vLLM did not change the model's behaviour. With retries, easy/medium/hard/extra are 88.7% / 77.4% / 64.0% / 45.2%.
- **Batching is the biggest cost lever:** at 16 requests in flight a query costs about 13× less GPU time than one at a time (59 vs 757 GPU-seconds per 1,000), while p95 latency barely moves (1.7 s → 2.0 s).
- **The verifier costs about 40% more GPU time and doubles p95 latency** (59 → 82 GPU-seconds per 1,000 at 16 in flight; p95 2.0 s → 4.2 s) for **+1.6 accuracy points** (71.1% → 72.6%). Only the ~11% of queries that fail are generated twice or three times, so the median is almost unchanged (0.70 → 0.74 s).
- **Throughput saturates at about 16 requests in flight:** at 64 it barely moves (17.5 QPS) and latency jumps to 3.4 s because requests queue. On this card the model (5.8 GiB) leaves only a 0.5 GiB KV cache (about 14.5k tokens, roughly 16 typical requests), so a larger GPU would saturate later; the shape of the curve, not the absolute numbers, is the transferable result.

<sub>Setup: `scripts/run_wsl_benchmark.sh` (vLLM with `--enforce-eager`, `--max-num-seqs 16`, an explicit 512 MiB KV cache, 2,048-token context; the API's retry mode is error-fed-back, ≤2 retries, matching the offline experiment). One run per level, so differences of a few percent are within noise. The laptop GPU is shared with the Windows desktop, and the dollar figures use a placeholder GPU rate; GPU-seconds per 1,000 queries is the price-independent number. Voting and resampled retries were evaluated offline only.</sub>

### Fine-tuning: which checkpoint ships

| Checkpoint | Training | Easy | Medium | Hard | Extra | Overall | Error rate |
|---|---|---|---|---|---|---|---|
| **Step 400 (lowest validation loss)** | ~0.9 epochs | 88.7% | 75.8% | 61.0% | 44.6% | **71.4%** | 10.5% |
| Step 1299 (end of training) | 3 epochs | 87.5% | 72.2% | 63.4% | 41.1% | 69.3% | 10.8% |

Validation loss bottomed out at step 400 and rose afterwards; the extra two epochs cost 2.0 points of execution accuracy, mostly on medium and extra-hard queries (−3.6 each; hard queries improved by 2.3), which suggests the model starts to overfit Spider's training databases. The shipped model is the step-400 checkpoint.

<sub>Setup: greedy decoding, bf16, max 256 new tokens; 3-shot uses the same 3 fixed training examples in every prompt. Fine-tuning: QLoRA (4-bit NF4 base, LoRA r=16 on q/k/v/o, 7.4M trainable parameters) on Spider's 7,000 training examples, whose databases do not overlap with dev; for evaluation the adapter is merged into the bf16 base and prompted zero-shot. Caveat: Spider's test set is not used here, so the dev set served both to pick the checkpoint (by validation loss, among the saved checkpoints) and to report results; the selection used loss rather than accuracy, so the bias is small but non-zero. Spider dev ships without difficulty labels, so tiers are computed from the gold SQL with Spider's official hardness rules (`eval/hardness.py`), which reproduce the published dev split counts (easy and medium exact, hard/extra within 2 queries). The retry and voting runs reuse the greedy first answer from the plain fine-tuned run (greedy decoding is deterministic; 20 of 20 answers were identical between the two GPUs) and generate only the extra attempts; they ran on a laptop RTX 5070, the other rows on Hazel's RTX 6000 Pro. Voting and resampled retries are random (one run each, no repeats), so differences of a few questions between them are within noise.</sub>

### Fine-tuning: how much training data is needed?

QLoRA on random subsets of the 7,000 training examples, same recipe otherwise; each model is merged into the bf16 base and evaluated zero-shot through vLLM, the same path for every row (including the 7,000 model, re-evaluated). Pre-registered in `docs/experiment_plan.md` (Addenda 3-5).

| Training examples | Easy | Medium | Hard | Extra | Overall | Empty outputs |
|---|---|---|---|---|---|---|
| 0 (base model, zero-shot) | 75.8% | 56.7% | 44.2% | 32.1% | 55.2% | |
| 1,000 (subset seed 0) | 65.3% | 53.1% | 47.1% | 31.5% | **51.6%** | 22% |
| 1,000 (subset seed 1) | 85.9% | 72.7% | 55.8% | 40.5% | 67.8% | 0% |
| 3,000 | 86.7% | 70.6% | 55.8% | 40.5% | 67.1% | 0% |
| 7,000 | 88.7% | 75.8% | 60.5% | 44.1% | 71.2% | 0% |

- **The first 1,000-example run broke, the second did not.** The seed-0 model returns an empty answer for 225 of 1,034 questions (mostly runs of whitespace); on the 809 non-empty answers it scores 65.8%. Its best validation loss (0.266) is the same as the 3,000 model's (0.265), so loss-based checkpoint selection did not catch it. Running that adapter unmerged on the 4-bit base did not fix it (66% empty on the first 200 questions, under a different generation path, so the rate itself is not comparable), so it is not a bf16-merge artifact. I did not find the cause. Neither is it simple under-training: the run's later checkpoints (steps 155 and 186 of 187) still return 215 and 207 empty answers (21% and 20%; accuracy 52.4% and 53.4%), against a pre-registered prediction of under 10%. The two 1,000-example subsets are statistically alike, so the difference is in how that run trained, which I could not isolate. Retraining on a different subset gave 67.8% with no empty outputs, so with two runs the best reading is that small-data training is unstable here, not that 1,000 examples are worse than none.
- **Hypotheses, scored as registered.** H2.1 (gain(1,000) ≥ 0.6 and gain(3,000) ≥ 0.85 of the 0→7,000 improvement): **not supported**. With the seed-0 run gain(1,000) is -0.23; with the seed-1 run it is 0.79, but gain(3,000) is 0.75 either way, below 0.85. H2.2 (hard and extra tiers gain more than easy from data beyond 1,000, each by > 1.5 points): **not supported**. With seed 0 the easy tier gains most (+23 points vs +13 hard, +12.5 extra); with seed 1 hard exceeds easy by 1.9 points but extra does not (0.8), and all of these are small.
- **What I would claim.** About 1,000 well-behaved examples already get most of the gain over prompting (67.8% vs 55.2%), and 7,000 add about 3-4 points on top. The 1,000-example and 3,000-example runs differ by less than the run-to-run spread (the two 1,000 runs differ by 16 points, one of them broken), so the shape between 1,000 and 3,000 is not resolved.

<sub>Setup: `scripts/run_wsl_learning_curve.sh` and `scripts/run_wsl_followup.sh`, QLoRA on the laptop GPU. The new models use micro-batch 2 x 8 gradient accumulation (the 7,000 model used 4 x 4), the same effective batch of 16, learning rate, schedule and 3 epochs; the best checkpoint is chosen by dev loss. One training run per size except 1,000, so differences under about 1.5 points are noise. Same dev-set caveat as above. Raw numbers: `results/learning_curve.json`.</sub>

---

## Limitations and how to read these numbers

- **No held-out test set.** Spider's test set is not used. The dev set both selects the checkpoint (by validation loss, not accuracy) and reports the result, so the headline numbers are slightly optimistic.
- **Mostly single runs.** Retry, voting and the training-size runs are one run each (the n-shot experiment uses 3 seeds per cell). Differences under about 1-1.5 points are noise, and tiers are small (extra-hard is 168 questions, so one question is 0.6 points).
- **Two engines.** The main table was produced with Hugging Face generation (fine-tuned model on Hazel's RTX 6000 Pro; retry and voting on the laptop RTX 5070). The n-shot and training-size experiments, and the served numbers, use vLLM, which differs from Hugging Face by about 0.7-1.0 points on the base model. Comparisons are made only within one engine; the 7,000 model scores 71.4% (Hugging Face) and 71.1-71.2% (vLLM).
- **Serving numbers are for this laptop only.** An 8 GB GPU shared with the Windows desktop, a 512 MiB KV cache, 16 concurrent requests as the operating point. Dollar costs use an assumed GPU price; GPU-seconds per 1,000 queries is the price-independent figure. Voting and resampled retries were evaluated offline and not served.
- **One benchmark.** Spider is small SQLite databases with clean schemas; nothing here says how the system behaves on a messy production schema. Difficulty tiers are computed from the gold SQL (Spider's rules), not shipped labels.
- **Post-hoc work is labelled.** The 8- and 16-shot runs and the 1,000-example reseed were added after seeing earlier results; `docs/experiment_plan.md` records the predictions made before each run (Addenda 2-5). Of the pre-registered hypotheses, H1.1 (partly), H1.5 and nothing else were supported; H1.2-H1.4, H2.1 and H2.2 were not, and the earlier explanation for the 3-shot extra-hard drop was withdrawn.
- **Unexplained failure.** One 1,000-example training run produced empty outputs for 22% of questions and I did not find why; validation loss did not reveal it. The shipped 7,000-example model has no empty outputs, but checkpoint selection by loss alone should not be trusted without an output check.
- **Training protocol difference.** The 1,000- and 3,000-example models used micro-batch 2 × 8 accumulation; the 7,000 model used 4 × 4 (same effective batch of 16).

---

## Architecture

```
Spider Dataset
     │
     ▼
prepare_data.py ──► data/processed/{train,dev}.jsonl
     │
     ├──► eval/run_baseline.py  (zero-shot / few-shot baseline)
     │
     ├──► training/train.py     (QLoRA fine-tuning)
     │         │
     │         └──► training/sweep.py  (hyperparameter sweep)
     │
     ├──► eval/run_baseline.py --mode finetuned
     │
     ├──► eval/scorer.py        (execution accuracy on all runs)
     │
     ├──► serving/app.py        (FastAPI + vLLM serving layer)
     │         │
     │         └──► self-correction loop
     │
     └──► eval/benchmark.py     (latency / throughput / cost report)
```

---

## Quickstart

### 1. Environment

```bash
# Rent a GPU with 16 GB VRAM (RTX 4090 / A10) on RunPod / Lambda / Vast.ai
conda create -n text2sql python=3.11 -y && conda activate text2sql
pip install -r requirements.txt
```

Set environment variables:

```bash
cp .env.example .env
# Edit .env: WANDB_API_KEY (optional), HF_TOKEN (optional)
export $(cat .env | xargs)
```

### 2. Data preparation (Phase 1)

```bash
python scripts/prepare_data.py
# Downloads Spider, parses schemas, writes data/processed/{train,dev}.jsonl
```

### 3. Baseline evaluation (Phase 3)

```bash
# Zero-shot
python eval/run_baseline.py --mode zero-shot

# Few-shot (3 examples in prompt)
python eval/run_baseline.py --mode few-shot --n-shot 3
```

Results are written to `eval/results/`.

### 4. Fine-tuning (Phase 4)

```bash
python training/train.py
# Trains with QLoRA (4-bit), logs to W&B, saves best checkpoint
# to training/outputs/best_checkpoint/
```

### 5. Hyperparameter sweep (Phase 5)

```bash
python training/sweep.py
# Runs 4 configurations; writes eval/results/sweep_results.json
```

### 6. Evaluate fine-tuned model

```bash
python eval/run_baseline.py --mode finetuned
```

### 7. Serve (Phase 6)

```bash
# Option A: local (two terminals)
vllm serve training/outputs/best_checkpoint --dtype bfloat16 --port 8001
uvicorn serving.app:app --host 0.0.0.0 --port 8000

# Option B: Docker
docker build -t text2sql-serving serving/
docker run --gpus all \
  -p 8000:8000 \
  -v $(pwd)/training/outputs/best_checkpoint:/model \
  -v $(pwd)/data/raw/spider/database:/spider_db \
  text2sql-serving
```

Example API call:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How many singers are there?",
    "schema_text": "Table: singer\n  singer_id INTEGER [PK]\n  name TEXT\n  age INTEGER",
    "execute": false
  }'
```

### 8. Production benchmarking (Phase 7)

```bash
python eval/benchmark.py --api-url http://localhost:8000 --mode finetuned
# Prints accuracy / QPS / latency / cost comparison table
```

---

## Project Structure

```
text-to-sql/
├── configs/
│   └── config.yaml          # All hyperparameters in one place
├── data/
│   ├── raw/spider/          # Spider dataset (auto-downloaded)
│   └── processed/           # Preprocessed JSONL files
├── eval/
│   ├── scorer.py            # Execution-accuracy harness
│   ├── run_baseline.py      # Inference runner (all modes)
│   └── benchmark.py        # Load test + cost report
├── training/
│   ├── train.py             # QLoRA fine-tuning
│   └── sweep.py             # Hyperparameter sweep
├── serving/
│   ├── app.py               # FastAPI + self-correction loop
│   ├── Dockerfile           # GPU container
│   └── entrypoint.sh        # vLLM + FastAPI startup
├── scripts/
│   ├── prepare_data.py      # Spider → JSONL preprocessing
│   └── utils.py             # Prompt templates, config loader
├── notebooks/               # Exploration only — not production
└── requirements.txt
```

---

## Key Design Decisions

**Execution accuracy over exact match.** Two syntactically different SQL queries can return the same correct result; two nearly identical queries can return wrong ones. The scorer executes both the predicted and gold queries against real SQLite databases and compares result sets (order-independent, lowercase-normalised).

**Schema-as-text prompt format.** Each example includes the full table/column/PK/FK structure as plain text in the prompt. This format meaningfully affects accuracy; it's documented and consistent across all evaluation modes.

**Self-correction loop.** When a generated SQL fails to execute, the error message is fed back to the model in a single retry. This demonstrates a verify-and-retry pattern applicable beyond SQL generation.

**vLLM for serving.** Continuous batching and PagedAttention make vLLM the industry standard for production LLM serving — the same tooling referenced in AI engineering job descriptions.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Fine-tuning | QLoRA (4-bit) via HuggingFace PEFT + TRL |
| Base model | Qwen2.5-3B-Instruct |
| Experiment tracking | Weights & Biases |
| Inference server | vLLM |
| API layer | FastAPI |
| Containerisation | Docker |
| Evaluation | Custom execution-accuracy harness (SQLite) |
| Compute | NC State Hazel HPC (Slurm), `hpc/` job scripts |
