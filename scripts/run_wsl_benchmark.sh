#!/bin/bash
# Serve the fine-tuned model with vLLM on this machine's GPU (inside WSL2) and benchmark it.
# Same steps as hpc/serve_bench.sbatch, without Slurm. Needs the repo, the adapter in
# training/outputs/best_checkpoint, and the Spider data/ copied into $PROJECT.
#
#   WIN_OUT=/mnt/c/Users/<you>/Text-to-SQL/eval/results bash scripts/run_wsl_benchmark.sh
#
# Steps: install vLLM env (first run) -> merge adapter -> vLLM -> FastAPI -> benchmark x2.
# Progress goes to $WIN_OUT/wsl_benchmark.status (if WIN_OUT is set); results are copied there.
set -eo pipefail

PROJECT="${PROJECT:-$HOME/text-to-sql}"
ENVDIR="${VLLM_ENV:-$HOME/vllm-env}"
WIN_OUT="${WIN_OUT:-}"
GPU_NAME="${GPU_NAME:-rtx_5070_laptop}"
VLLM_PORT=8001
API_PORT=8000

status() {
    echo "$(date +%T) $*"
    if [ -n "$WIN_OUT" ]; then echo "$(date +%T) $*" >> "$WIN_OUT/wsl_benchmark.status" || true; fi
}
VLLM_PID=""; API_PID=""
cleanup() { kill $API_PID $VLLM_PID 2>/dev/null || true; }
fail() {
    status "FAILED (exit $?) near line $1"
    if [ -n "$WIN_OUT" ]; then
        cp -f "$PROJECT"/logs/*.log "$WIN_OUT/" 2>/dev/null || true
    fi
}
trap 'fail $LINENO' ERR
trap cleanup EXIT

cd "$PROJECT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p eval/results logs
: > "${WIN_OUT:-/dev/null}/wsl_benchmark.status" 2>/dev/null || true
status "start. project=$PROJECT env=$ENVDIR"

# 1. vLLM env (separate from training: vLLM pins its own torch)
if ! "$ENVDIR/bin/python" -c "import vllm" 2>/dev/null; then
    status "installing vLLM env (several minutes)"
    rm -rf "$ENVDIR"
    python3 -m venv --without-pip "$ENVDIR"
    [ -s "$HOME/get-pip.py" ] || curl -sSL https://bootstrap.pypa.io/get-pip.py -o "$HOME/get-pip.py"
    "$ENVDIR/bin/python" "$HOME/get-pip.py" -q
    "$ENVDIR/bin/python" -m pip install vllm==0.34.0 > logs/pip_vllm.log 2>&1
    "$ENVDIR/bin/python" -m pip install fastapi "uvicorn[standard]" httpx pydantic pyyaml tqdm rich sqlglot \
        peft==0.20.0 pytest > logs/pip_libs.log 2>&1
fi
"$ENVDIR/bin/python" - <<'PY' 2>&1 | grep -v Warn | tee -a logs/env.log
import torch, vllm
print("torch", torch.__version__, "| cuda build", torch.version.cuda, "| gpu", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", "| vllm", vllm.__version__)
PY
status "env ok: $(tail -1 logs/env.log)"

# 2. merge the adapter into the bf16 base (vLLM serves a full model directory)
if [ ! -f training/outputs/merged/config.json ]; then
    status "merging adapter"
    "$ENVDIR/bin/python" scripts/merge_adapter.py --out training/outputs/merged > logs/merge.log 2>&1
fi
status "merged model ready"

# 3. vLLM (two attempts: normal, then a leaner config for an 8 GB card)
wait_for() {   # wait_for <url> <pid> <name> <log> <max_seconds>
    local waited=0
    until curl -sf "$1" > /dev/null 2>&1; do
        if ! kill -0 "$2" 2>/dev/null; then status "$3 exited during startup"; tail -25 "$4"; return 1; fi
        if (( waited >= $5 )); then status "$3 not ready after ${5}s"; tail -25 "$4"; return 1; fi
        sleep 5; waited=$((waited + 5))
    done
    status "$3 ready after ${waited}s"
}
start_vllm() {  # start_vllm <gpu-memory-utilization> [extra args]
    local util="$1"; shift
    "$ENVDIR/bin/vllm" serve training/outputs/merged --served-model-name text2sql --dtype bfloat16 \
        --max-model-len 2048 --gpu-memory-utilization "$util" "$@" \
        --host 127.0.0.1 --port "$VLLM_PORT" > logs/vllm.log 2>&1 &
    VLLM_PID=$!
    wait_for "http://127.0.0.1:$VLLM_PORT/health" "$VLLM_PID" "vLLM" logs/vllm.log 900
}
status "starting vLLM (util 0.85)"
if ! start_vllm 0.85; then
    kill $VLLM_PID 2>/dev/null || true; sleep 5
    status "retrying vLLM leaner: util 0.92 + --enforce-eager"
    start_vllm 0.92 --enforce-eager
fi
status "vLLM: $(grep -E 'GPU KV cache size|Maximum concurrency' logs/vllm.log | tr '\n' ' ' | cut -c1-200)"

# 4. FastAPI in front of it
VLLM_URL="http://127.0.0.1:$VLLM_PORT/v1" SERVED_MODEL_NAME=text2sql RETRY_MODE=feedback \
SPIDER_DB_DIR="$PROJECT/data/raw/spider/database" \
    "$ENVDIR/bin/uvicorn" serving.app:app --host 127.0.0.1 --port "$API_PORT" > logs/api.log 2>&1 &
API_PID=$!
wait_for "http://127.0.0.1:$API_PORT/health" "$API_PID" "API" logs/api.log 120

# 5. smoke test, then the benchmark (full dev set per pass)
curl -sf -X POST "http://127.0.0.1:$API_PORT/query" -H 'Content-Type: application/json' \
    -d '{"question":"How many singers do we have?","schema_text":"Table: singer\n  Singer_ID number [PK]\n  Name text","execute":false}' \
    | tee logs/smoke.json; echo
status "smoke test ok: $(cat logs/smoke.json | cut -c1-120)"

status "benchmark 1/2: generate only"
"$ENVDIR/bin/python" eval/benchmark.py --api-url "http://127.0.0.1:$API_PORT" --label ft-serve \
    --gpu-name "$GPU_NAME" --concurrency 1,4,16,64 > logs/benchmark_ft-serve.log 2>&1
status "benchmark 2/2: execute + self-correct"
"$ENVDIR/bin/python" eval/benchmark.py --api-url "http://127.0.0.1:$API_PORT" --label ft-serve-exec \
    --gpu-name "$GPU_NAME" --concurrency 1,16 --execute > logs/benchmark_ft-serve-exec.log 2>&1

if [ -n "$WIN_OUT" ]; then
    cp -f eval/results/benchmark_ft-serve*_report.json eval/results/benchmark_ft-serve*_scores.json "$WIN_OUT/" 2>/dev/null || true
    cp -f logs/*.log logs/smoke.json "$WIN_OUT/" 2>/dev/null || true
fi
status "ALL DONE"
