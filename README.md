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
| **Fine-tuned (QLoRA)** | **71.4%** | — | — |
| Fine-tuned + execution retry | *planned* | — | — |
| Fine-tuned + execution voting | *planned* | — | — |

All rows are Qwen2.5-3B-Instruct. Latency and cost come from the served system (vLLM on one GPU) and are filled in once serving is benchmarked.

### By difficulty

| Configuration | Easy (248) | Medium (446) | Hard (172) | Extra (168) | Overall |
|---|---|---|---|---|---|
| Zero-shot | 76.2% | 57.8% | 41.3% | 35.7% | 55.9% |
| 3-shot | 82.3% | 64.1% | 50.6% | 31.0% | 60.8% |
| **Fine-tuned (QLoRA)** | **88.7%** | **75.8%** | **61.0%** | **44.6%** | **71.4%** |

- **Fine-tuning is the largest single step:** +15.5 points over zero-shot and +10.5 over 3-shot, with gains in every tier, including extra-hard (+8.9 over zero-shot), where few-shot prompting had *lost* ground. The execution error rate (SQL that fails to run) nearly halves, from 18–19% to 10.5%.
- Accuracy drops steadily with query complexity in every configuration; even after fine-tuning, about a quarter of extra-hard queries fail to execute (nested subqueries, set operations, multi-way joins).
- 3-shot prompting adds +4.9 points overall, concentrated in easy–hard (+6 to +9), but *loses* 4.8 points on extra-hard queries. The 3 fixed examples are single-table queries (one easy, two medium) with no joins, nesting or set operations, which likely biases the model toward flat SQL where nesting is needed.

### Fine-tuning: which checkpoint ships

| Checkpoint | Training | Easy | Medium | Hard | Extra | Overall | Error rate |
|---|---|---|---|---|---|---|---|
| **Step 400 (lowest validation loss)** | ~0.9 epochs | 88.7% | 75.8% | 61.0% | 44.6% | **71.4%** | 10.5% |
| Step 1299 (end of training) | 3 epochs | 87.5% | 72.2% | 63.4% | 41.1% | 69.3% | 10.8% |

Validation loss bottomed out at step 400 and rose afterwards; the extra two epochs cost 2.0 points of execution accuracy, mostly on medium and extra-hard queries (−3.6 each; hard queries improved by 2.3), which suggests the model starts to overfit Spider's training databases. The shipped model is the step-400 checkpoint.

<sub>Setup: greedy decoding, bf16, max 256 new tokens; 3-shot uses the same 3 fixed training examples in every prompt. Fine-tuning: QLoRA (4-bit NF4 base, LoRA r=16 on q/k/v/o, 7.4M trainable parameters) on Spider's 7,000 training examples, whose databases do not overlap with dev; for evaluation the adapter is merged into the bf16 base and prompted zero-shot. Caveat: Spider's test set is not used here, so the dev set served both to pick the checkpoint (by validation loss, among the saved checkpoints) and to report results; the selection used loss rather than accuracy, so the bias is small but non-zero. Spider dev ships without difficulty labels, so tiers are computed from the gold SQL with Spider's official hardness rules (`eval/hardness.py`), which reproduce the published dev split counts (easy and medium exact, hard/extra within 2 queries). Remaining rows are filled in as runs complete.</sub>

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
