# Text-to-SQL: A 3B Model as a Production Service

> *How far can a 3B model go as a production text-to-SQL service when it's fine-tuned, checked against the database it queries, and served on one GPU, and what does each step cost in accuracy, latency, and dollars?*

Each step is treated as a system component that has to earn its place:

1. **Prompting → fine-tuning.** **Qwen2.5-3B-Instruct**, zero-shot and few-shot, then fine-tuned with **QLoRA** on Spider.
2. **Execution feedback.** The generated SQL is run against the real database; failures trigger a retry, or several candidates are generated and voted on by their results. The database acts as a verifier.
3. **Serving.** **vLLM + FastAPI** on a single GPU, measured for latency, throughput under concurrent load, and cost per 1K queries.

Every configuration is scored with an **execution-accuracy harness** against live SQLite databases, and reported with its cost, not just its accuracy.

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

- **Fine-tuning is the largest single step:** +15.5 points over zero-shot and +10.5 over 3-shot, with gains in every tier, including extra-hard (+8.9 over zero-shot), where few-shot prompting had *lost* ground. The execution error rate (SQL that fails to run) nearly halves, from 18–19% to 10.5%.
- Accuracy drops steadily with query complexity in every configuration; even after fine-tuning, about a quarter of extra-hard queries fail to execute (nested subqueries, set operations, multi-way joins).
- 3-shot prompting adds +4.9 points overall, concentrated in easy–hard (+6 to +9), but *loses* 4.8 points on extra-hard queries. The 3 fixed examples are single-table queries (one easy, two medium) with no joins, nesting or set operations, which likely biases the model toward flat SQL where nesting is needed.

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
