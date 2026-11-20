#!/bin/bash
# Follow-up to the anomalous n=1,000 learning-curve point (docs/experiment_plan.md, Addendum 5).
#   A. run the n=1,000 adapter UNMERGED on the 4-bit base it was trained on (first 200 dev questions)
#      and compare its empty-prediction rate with the merged-bf16 / vLLM predictions.
#   B. retrain n=1,000 on a different random subset (seed 1), merge, evaluate through vLLM.
#
#   WIN_OUT=/mnt/c/Users/<you>/Text-to-SQL/eval/results bash scripts/run_wsl_followup.sh
# Sequential on purpose (one 8 GB GPU). Create $WIN_OUT/STOP to stop before the next stage.
set -eo pipefail
PROJECT="${PROJECT:-$HOME/text-to-sql}"
TRAIN_ENV="${TRAIN_ENV:-$HOME/train-env}"
VLLM_ENV="${VLLM_ENV:-$HOME/vllm-env}"
WIN_OUT="${WIN_OUT:-}"
PORT=8001
status() { echo "$(date +%T) $*"; [ -n "$WIN_OUT" ] && echo "$(date +%T) $*" >> "$WIN_OUT/followup.status" || true; }
VLLM_PID=""
stop_vllm() { kill $VLLM_PID 2>/dev/null || true; VLLM_PID=""; }
trap 'status "FAILED (exit $?) near line $LINENO"; stop_vllm' ERR
trap stop_vllm EXIT
should_stop() { [ -n "$WIN_OUT" ] && [ -f "$WIN_OUT/STOP" ]; }

cd "$PROJECT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PATH="$HOME/bin:$PATH" CC="$HOME/bin/cc" TRITON_LIBCUDA_PATH=/usr/lib/wsl/lib
export CPATH="$HOME/pyinc/usr/include:$HOME/pyinc/usr/include/python3.12:$HOME/pyinc/usr/include/x86_64-linux-gnu/python3.12"
mkdir -p logs eval/results/diag eval/results/curve_s1
[ -n "$WIN_OUT" ] && : > "$WIN_OUT/followup.status"
status "start. commit $(git log --oneline -1)"

# ---- A: unmerged adapter on the 4-bit base -------------------------------------------------
if [ ! -f eval/results/diag/n1000_nf4.done ]; then
    "$TRAIN_ENV/bin/python" scripts/diag_adapter_nf4.py --compare eval/results/curve/ft1000_predictions.jsonl --limit 200 | tee -a logs/diag_nf4.log | while read -r l; do status "A: $l"; done
    "$TRAIN_ENV/bin/python" scripts/diag_adapter_nf4.py --adapter training/outputs_lc/n1000/best_checkpoint \
        --limit 200 --tag n1000_nf4 > logs/diag_nf4_run.log 2>&1
    status "A: $(tail -1 logs/diag_nf4_run.log)"
    touch eval/results/diag/n1000_nf4.done
fi
[ -n "$WIN_OUT" ] && { mkdir -p "$WIN_OUT/diag"; cp -f eval/results/diag/n1000_nf4_* "$WIN_OUT/diag/" 2>/dev/null || true; }

# ---- B: retrain n=1,000 on subset seed 1 ---------------------------------------------------
should_stop && { status "STOP file found: stopping before B"; exit 0; }
out=training/outputs_lc/n1000_s1
if [ ! -f "$out/best_checkpoint/adapter_model.safetensors" ]; then
    for bs in "--batch-size 2" "--batch-size 1"; do
        rm -rf "$out"
        status "B: training n=1000 subset-seed 1 (187 steps, eval every 31) $bs"
        if PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$TRAIN_ENV/bin/python" training/train.py \
               --train-size 1000 --subset-seed 1 --output-dir "$out" --eval-steps 31 --run-name lc1000s1 $bs \
               > logs/train_n1000_s1.log 2>&1; then
            status "B: trained: $(grep -E "'train_runtime'" logs/train_n1000_s1.log | tail -1 | cut -c1-120)"; break
        fi
        grep -qiE "out of memory|OutOfMemory" logs/train_n1000_s1.log || { status "B: training failed (not OOM)"; exit 1; }
    done
fi
[ -f "$out/merged/config.json" ] || "$TRAIN_ENV/bin/python" scripts/merge_adapter.py --adapter "$out/best_checkpoint" --out "$out/merged" > logs/merge_n1000_s1.log 2>&1
should_stop && { status "STOP file found: stopping before evaluating B"; exit 0; }

status "B: serving merged model"
"$VLLM_ENV/bin/vllm" serve "$out/merged" --served-model-name text2sql --dtype bfloat16 \
    --max-model-len 2048 --gpu-memory-utilization 0.84 --kv-cache-memory 536870912 \
    --enforce-eager --max-num-seqs 16 --host 127.0.0.1 --port "$PORT" > logs/vllm_lc_ft1000s1.log 2>&1 &
VLLM_PID=$!
waited=0
until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; do
    kill -0 $VLLM_PID 2>/dev/null || { status "B: vLLM exited during startup"; exit 1; }
    (( waited >= 600 )) && { status "B: vLLM not ready"; exit 1; }
    sleep 4; waited=$((waited + 4))
done
"$VLLM_ENV/bin/python" eval/vllm_eval.py --tag ft1000s1 --strategy none --model text2sql \
    --url "http://127.0.0.1:$PORT/v1" --out-dir eval/results/curve_s1 > logs/eval_ft1000s1.log 2>&1
stop_vllm
"$VLLM_ENV/bin/python" - <<'PY' > logs/ft1000s1_result.txt
import json
d = [json.loads(l) for l in open("eval/results/curve_s1/summary.jsonl")][-1]
rows = [json.loads(l) for l in open("eval/results/curve_s1/ft1000s1_predictions.jsonl")]
empty = sum(not r["sql_pred"].strip() for r in rows)
print("exec_acc=%.4f empty=%d tiers=%s" % (d["exec_acc"], empty, {k: round(v, 3) for k, v in d["by_difficulty"].items()}))
PY
status "B: ft1000s1: $(cat logs/ft1000s1_result.txt)"
if [ -n "$WIN_OUT" ]; then
    mkdir -p "$WIN_OUT/curve_s1"
    cp -f eval/results/curve_s1/* "$WIN_OUT/curve_s1/" 2>/dev/null || true
    cp -f "$out/best_checkpoint/trainer_state.json" "$WIN_OUT/curve_s1/trainer_state_n1000_s1.json" 2>/dev/null || true
    cp -f logs/train_n1000_s1.log "$WIN_OUT/curve_s1/" 2>/dev/null || true
fi
status "ALL DONE"
