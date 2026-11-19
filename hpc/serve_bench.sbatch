#!/bin/bash
# Serve the fine-tuned model on ONE GPU (vLLM + FastAPI) and benchmark it. From the repo root:
#   sbatch hpc/serve_bench.sbatch
# Needs: hpc/setup_env.sh and hpc/setup_serve_env.sh done, training/outputs/best_checkpoint present.
# Two passes over the full dev set at several concurrency levels:
#   ft-serve        generate only            (latency / throughput / cost / accuracy)
#   ft-serve-exec   execute + self-correct   (what the database verifier costs)
# Output: eval/results/benchmark_<label>_{report.json,scores.json,predictions.jsonl}
#SBATCH --job-name=t2sql_serve
#SBATCH --partition=gpu_partners
#SBATCH --qos=short_gpu
#SBATCH --gres=gpu:rtx_6000_pro:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/share/csc591f26/jmuthuk/logs/slurm_%x_%j.out
set -eo pipefail

source hpc/env.sh
cd "$PROJECT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
GPU_NAME="rtx_6000_pro"
VLLM_PORT=8001
API_PORT=8000

echo "=== $(date)  job $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv

# 1. vLLM serves a full model directory, so merge the LoRA adapter into the bf16 base (training env).
if [[ ! -f training/outputs/merged/config.json ]]; then
    activate_env "$ENV_PREFIX"
    python scripts/merge_adapter.py --out training/outputs/merged
fi

# 2. Switch to the serving env and start vLLM.
activate_env "$SERVE_ENV_PREFIX"
LOG_DIR="${LOG_DIR:-/share/csc591f26/jmuthuk/logs}"
VLLM_LOG="$LOG_DIR/vllm_${SLURM_JOB_ID}.log"
API_LOG="$LOG_DIR/api_${SLURM_JOB_ID}.log"

cleanup() { kill "${API_PID:-}" "${VLLM_PID:-}" 2>/dev/null || true; }
trap cleanup EXIT

vllm serve training/outputs/merged \
    --served-model-name text2sql --dtype bfloat16 \
    --max-model-len 2048 --gpu-memory-utilization 0.85 \
    --host 127.0.0.1 --port "$VLLM_PORT" > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

wait_for() {   # wait_for <url> <pid> <name> <max_seconds>
    local waited=0
    until curl -sf "$1" > /dev/null 2>&1; do
        if ! kill -0 "$2" 2>/dev/null; then
            echo "ERROR: $3 exited during startup. Last log lines:"; tail -30 "${4:-/dev/null}"; exit 1
        fi
        if (( waited >= $5 )); then echo "ERROR: $3 not ready after ${5}s"; tail -30 "$4"; exit 1; fi
        sleep 5; waited=$((waited + 5))
    done
    echo "$3 ready after ${waited}s"
}
wait_for "http://127.0.0.1:$VLLM_PORT/health" "$VLLM_PID" "vLLM" "$VLLM_LOG" 900

# 3. FastAPI in front of it.
VLLM_URL="http://127.0.0.1:$VLLM_PORT/v1" SERVED_MODEL_NAME=text2sql \
SPIDER_DB_DIR="$PROJECT_DIR/data/raw/spider/database" RETRY_MODE="${RETRY_MODE:-feedback}" \
    uvicorn serving.app:app --host 127.0.0.1 --port "$API_PORT" > "$API_LOG" 2>&1 &
API_PID=$!
wait_for "http://127.0.0.1:$API_PORT/health" "$API_PID" "API" "$API_LOG" 120

# 4. Benchmark.
python eval/benchmark.py --api-url "http://127.0.0.1:$API_PORT" --label ft-serve \
    --gpu-name "$GPU_NAME" --concurrency 1,4,16,64
python eval/benchmark.py --api-url "http://127.0.0.1:$API_PORT" --label ft-serve-exec \
    --gpu-name "$GPU_NAME" --concurrency 1,16 --execute

echo "=== Done $(date). Reports: eval/results/benchmark_ft-serve*_report.json"
