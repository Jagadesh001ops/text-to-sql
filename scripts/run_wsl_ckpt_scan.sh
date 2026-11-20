#!/bin/bash
# Checkpoint scan for the collapsed n=1,000 (subset seed 0) run (docs/experiment_plan.md, Addendum 6):
# evaluate the other saved checkpoints (steps 155 and 186; step 124 is the shipped best_checkpoint,
# already evaluated as ft1000) and count empty outputs.
#   WIN_OUT=/mnt/c/Users/<you>/Text-to-SQL/eval/results bash scripts/run_wsl_ckpt_scan.sh
set -eo pipefail
PROJECT="${PROJECT:-$HOME/text-to-sql}"; TRAIN_ENV="$HOME/train-env"; VLLM_ENV="$HOME/vllm-env"; WIN_OUT="${WIN_OUT:-}"; PORT=8001
status() { echo "$(date +%T) $*"; [ -n "$WIN_OUT" ] && echo "$(date +%T) $*" >> "$WIN_OUT/ckpt_scan.status" || true; }
VLLM_PID=""; stop_vllm() { kill $VLLM_PID 2>/dev/null || true; VLLM_PID=""; }
trap 'status "FAILED (exit $?) near line $LINENO"; stop_vllm' ERR
trap stop_vllm EXIT
cd "$PROJECT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PATH="$HOME/bin:$PATH" CC="$HOME/bin/cc" TRITON_LIBCUDA_PATH=/usr/lib/wsl/lib
export CPATH="$HOME/pyinc/usr/include:$HOME/pyinc/usr/include/python3.12:$HOME/pyinc/usr/include/x86_64-linux-gnu/python3.12"
mkdir -p logs eval/results/curve_scan
[ -n "$WIN_OUT" ] && : > "$WIN_OUT/ckpt_scan.status"
status "start. commit $(git log --oneline -1)"
for step in 155 186; do
    tag="ft1000_step$step"; src="training/outputs_lc/n1000/checkpoint-$step"; m="training/outputs_lc/n1000/merged_$step"
    [ -f "$m/config.json" ] || "$TRAIN_ENV/bin/python" scripts/merge_adapter.py --adapter "$src" --out "$m" > "logs/merge_$step.log" 2>&1
    status "$tag: serving"
    "$VLLM_ENV/bin/vllm" serve "$m" --served-model-name text2sql --dtype bfloat16 --max-model-len 2048 \
        --gpu-memory-utilization 0.84 --kv-cache-memory 536870912 --enforce-eager --max-num-seqs 16 \
        --host 127.0.0.1 --port "$PORT" > "logs/vllm_scan_$step.log" 2>&1 &
    VLLM_PID=$!; waited=0
    until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; do
        kill -0 $VLLM_PID 2>/dev/null || { status "$tag: vLLM exited"; exit 1; }
        (( waited >= 600 )) && { status "$tag: vLLM not ready"; exit 1; }
        sleep 4; waited=$((waited + 4))
    done
    "$VLLM_ENV/bin/python" eval/vllm_eval.py --tag "$tag" --strategy none --model text2sql \
        --url "http://127.0.0.1:$PORT/v1" --out-dir eval/results/curve_scan > "logs/eval_$tag.log" 2>&1
    stop_vllm; sleep 10
    status "$tag: $("$VLLM_ENV/bin/python" - "$tag" <<'PY'
import json, sys
t = sys.argv[1]
d = [json.loads(l) for l in open("eval/results/curve_scan/summary.jsonl") if json.loads(l)["tag"] == t][-1]
rows = [json.loads(l) for l in open(f"eval/results/curve_scan/{t}_predictions.jsonl")]
print("exec_acc=%.4f empty=%d" % (d["exec_acc"], sum(not r["sql_pred"].strip() for r in rows)))
PY
)"
    rm -rf "$m"
done
[ -n "$WIN_OUT" ] && { mkdir -p "$WIN_OUT/curve_scan"; cp -f eval/results/curve_scan/* "$WIN_OUT/curve_scan/" 2>/dev/null || true; }
status "ALL DONE"
