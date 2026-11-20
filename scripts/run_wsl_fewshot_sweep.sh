#!/bin/bash
# Experiment 1 (docs/experiment_plan.md): how many examples in the prompt, and which ones?
# Serves the BASE model with vLLM (inside WSL2, 8 GB GPU) and runs the sweep:
#   0-shot, the original first-3, and {1,3,6} shots x {random, diverse} x 3 seeds = 20 runs.
#
#   WIN_OUT=/mnt/c/Users/<you>/Text-to-SQL/eval/results bash scripts/run_wsl_fewshot_sweep.sh
#
# Needs the vLLM env from scripts/run_wsl_benchmark.sh (~/vllm-env, userland toolchain).
# To give the GPU back early: create the file  $WIN_OUT/STOP  — the sweep stops before the next run.
# Progress: $WIN_OUT/fewshot_sweep.status ; results: $WIN_OUT/sweep/summary.jsonl (resumable).
set -eo pipefail

PROJECT="${PROJECT:-$HOME/text-to-sql}"
ENVDIR="${VLLM_ENV:-$HOME/vllm-env}"
WIN_OUT="${WIN_OUT:-}"
PORT=8001

status() {
    echo "$(date +%T) $*"
    if [ -n "$WIN_OUT" ]; then echo "$(date +%T) $*" >> "$WIN_OUT/fewshot_sweep.status" || true; fi
}
VLLM_PID=""
cleanup() { kill $VLLM_PID 2>/dev/null || true; }
trap 'status "FAILED (exit $?) near line $LINENO"; [ -n "$WIN_OUT" ] && cp -f "$PROJECT"/logs/vllm_sweep.log "$WIN_OUT/" 2>/dev/null || true' ERR
trap cleanup EXIT

cd "$PROJECT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PATH="$HOME/bin:$PATH" CC="$HOME/bin/cc" TRITON_LIBCUDA_PATH=/usr/lib/wsl/lib
export CPATH="$HOME/pyinc/usr/include:$HOME/pyinc/usr/include/python3.12:$HOME/pyinc/usr/include/x86_64-linux-gnu/python3.12"
mkdir -p logs eval/results/sweep
[ -n "$WIN_OUT" ] && : > "$WIN_OUT/fewshot_sweep.status"
status "start. commit $(git log --oneline -1)"

# vLLM wants a complete snapshot; the cached one lacks README/LICENSE, so point at the directory.
MODEL_DIR=$(ls -d "$HOME"/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/*/ | head -1)
status "base model: $MODEL_DIR"

# Settings verified on this 8 GB card (see scripts/run_wsl_benchmark.sh); 6144 context for 6-shot prompts.
"$ENVDIR/bin/vllm" serve "$MODEL_DIR" --served-model-name base --dtype bfloat16 \
    --max-model-len 6144 --gpu-memory-utilization 0.84 --kv-cache-memory 536870912 \
    --enforce-eager --max-num-seqs 16 --host 127.0.0.1 --port "$PORT" > logs/vllm_sweep.log 2>&1 &
VLLM_PID=$!
waited=0
until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; do
    kill -0 $VLLM_PID 2>/dev/null || { status "vLLM exited during startup"; tail -25 logs/vllm_sweep.log; exit 1; }
    (( waited >= 600 )) && { status "vLLM not ready after 600s"; exit 1; }
    sleep 4; waited=$((waited + 4))
done
status "vLLM ready after ${waited}s: $(grep -E 'GPU KV cache size' logs/vllm_sweep.log | cut -c1-160)"

run() {   # run <tag> <strategy> <shots> <seed>
    if [ -n "$WIN_OUT" ] && [ -f "$WIN_OUT/STOP" ]; then status "STOP file found: stopping before $1"; return 1; fi
    status "run $1 ($2, n=$3, seed=$4)"
    "$ENVDIR/bin/python" eval/vllm_eval.py --tag "$1" --strategy "$2" --shots "$3" --seed "$4" \
        --model base --url "http://127.0.0.1:$PORT/v1" > "logs/sweep_$1.log" 2>&1
    status "done $1: $(tail -1 eval/results/sweep/summary.jsonl | "$ENVDIR/bin/python" -c 'import json,sys; d=json.loads(sys.stdin.read()); print("exec_acc=%.4f errors=%d wall=%ss" % (d["exec_acc"], d["request_errors"], d["wall_s"]))')"
    if [ -n "$WIN_OUT" ]; then
        mkdir -p "$WIN_OUT/sweep"; cp -f eval/results/sweep/summary.jsonl "$WIN_OUT/sweep/summary.jsonl"
    fi
}

# Most informative first, so partial results are useful if the sweep is stopped early.
run zs none 0 0
run first3 first 3 0
for seed in 0 1 2; do run "rand3_s$seed" random 3 "$seed"; run "div3_s$seed" diverse 3 "$seed"; done
for seed in 0 1 2; do run "rand6_s$seed" random 6 "$seed"; run "div6_s$seed" diverse 6 "$seed"; done
for seed in 0 1 2; do run "rand1_s$seed" random 1 "$seed"; run "div1_s$seed" diverse 1 "$seed"; done

if [ -n "$WIN_OUT" ]; then cp -f eval/results/sweep/*_scores.json "$WIN_OUT/sweep/" 2>/dev/null || true; fi
status "ALL DONE"
