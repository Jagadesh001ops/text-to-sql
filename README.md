# Text-to-SQL Fine-Tuning & Serving Pipeline

> *Can a fine-tuned 3B model match a frontier API's accuracy at a fraction of the cost?*
> This project builds the full pipeline to answer that question with real data.

Fine-tunes **Qwen2.5-3B-Instruct** via **QLoRA** on the Spider benchmark, evaluates it with an **execution-accuracy harness** against live SQLite databases, serves it through **vLLM + FastAPI**, and benchmarks it head-to-head against **GPT-4o-mini** on accuracy, latency, throughput, and cost.

---

## Results

| Model | Exec Acc | Latency p95 | Cost / 1K queries |
|---|---|---|---|
| Qwen2.5-3B (zero-shot) | — | — | — |
| Qwen2.5-3B (few-shot) | — | — | — |
| **Qwen2.5-3B (fine-tuned)** | **—** | **—** | **$—** |
| GPT-4o-mini (reference) | — | — | $— |

*Results will be filled in after training runs complete.*

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
# Edit .env: WANDB_API_KEY, OPENAI_API_KEY, HF_TOKEN (if needed)
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

# GPT-4o-mini ceiling (requires OPENAI_API_KEY)
python eval/run_baseline.py --mode openai
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
| Reference model | GPT-4o-mini (OpenAI) |
