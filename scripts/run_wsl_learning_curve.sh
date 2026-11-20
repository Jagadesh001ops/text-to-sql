#!/bin/bash
# Experiment 2 (docs/experiment_plan.md): how much training data does fine-tuning need?
# For each size N: QLoRA-train on a nested random subset (best checkpoint by dev loss), merge the
# adapter into the bf16 base, serve it with vLLM, evaluate zero-shot on Spider dev. The existing
# 7,000-example model is re-evaluated through the same path.
#
#   LC_SIZES=1000,3000 WIN_OUT=/mnt/c/Users/<you>/Text-to-SQL/eval/results bash scripts/run_wsl_learning_curve.sh
#
# Resumable (finished stages are skipped). To give the GPU back: create  $WIN_OUT/STOP  — the
# pipeline stops before its next stage. Progress: $WIN_OUT/learning_curve.status
set -eo pipefail

PROJECT="${PROJECT:-$HOME/text-to-sql}"
TRAIN_ENV="${TRAIN_ENV:-$HOME/train-env}"
VLLM_ENV="${VLLM_ENV:-$HOME/vllm-env}"
WIN_OUT="${WIN_OUT:-}"
PORT=8001
IFS=, read -ra SIZES <<< "${LC_SIZES:-1000,3000}"

status() {
    echo "$(date +%T) $*"
    if [ -n "$WIN_OUT" ]; then echo "$(date +%T) $*" >> "$WIN_OUT/learning_curve.status" || true; fi
}
VLLM_PID=""
stop_vllm() { kill $VLLM_PID 2>/dev/null || true; VLLM_PID=""; }
trap 'status "FAILED (exit $?) near line $LINENO"; stop_vllm; [ -n "$WIN_OUT" ] && cp -f "$PROJECT"/logs/*.log "$WIN_OUT/" 2>/dev/null || true' ERR
trap stop_vllm EXIT
should_stop() { [ -n "$WIN_OUT" ] && [ -f "$WIN_OUT/STOP" ]; }

cd "$PROJECT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PATH="$HOME/bin:$PATH" CC="$HOME/bin/cc" TRITON_LIBCUDA_PATH=/usr/lib/wsl/lib
export CPATH="$HOME/pyinc/usr/include:$HOME/pyinc/usr/include/python3.12:$HOME/pyinc/usr/include/x86_64-linux-gnu/python3.12"
mkdir -p logs eval/results/curve
[ -n "$WIN_OUT" ] && : > "$WIN_OUT/learning_curve.status"
status "start. commit $(git log --oneline -1)"

status "training env: $(bash scripts/wsl_train_env.sh 2>&1 | tail -1 | cut -c1-200)"

copy_out() {
    [ -n "$WIN_OUT" ] || return 0
    mkdir -p "$WIN_OUT/curve"
    cp -f eval/results/curve/summary.jsonl "$WIN_OUT/curve/" 2>/dev/null || true
    for d in training/outputs_lc/n*; do
        [ -f "$d/best_checkpoint/trainer_state.json" ] && cp -f "$d/best_checkpoint/trainer_state.json" "$WIN_OUT/curve/trainer_state_$(basename "$d").json" 2>/dev/null || true
    done
    cp -f logs/train_n*.log logs/vllm_lc_*.log "$WIN_OUT/curve/" 2>/dev/null || true
}

train() {   # train <N>: 3 epochs, ~6 evaluations; micro-batch 2 x 8 accumulation (effective 16), 1 x 16 if OOM
    # (batch 4 peaks at ~5.9 GB on typical examples and the 152k-token vocab makes long batches OOM;
    #  see docs/experiment_plan.md, Addendum 4)
    local n="$1" out="training/outputs_lc/n$1"
    [ -f "$out/best_checkpoint/adapter_model.safetensors" ] && { status "n=$n: already trained"; return 0; }
    local steps=$(( n * 3 / 16 )); local every=$(( steps / 6 )); [ "$every" -lt 10 ] && every=10
    for bs in "--batch-size 2" "--batch-size 1"; do
        rm -rf "$out"
        status "n=$n: training ($steps steps, eval every $every) $bs"
        if PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$TRAIN_ENV/bin/python" training/train.py \
               --train-size "$n" --output-dir "$out" --eval-steps "$every" \
               --run-name "lc$n" $bs > "logs/train_n$n.log" 2>&1; then
            status "n=$n: trained: $(grep -E "'train_runtime'" "logs/train_n$n.log" | tail -1 | cut -c1-120)"
            return 0
        fi
        grep -qiE "out of memory|OutOfMemory" "logs/train_n$n.log" || { status "n=$n: training failed (not OOM)"; tail -15 "logs/train_n$n.log"; return 1; }
        status "n=$n: out of GPU memory${bs:+ even with $bs}, trying a smaller batch"
    done
    status "n=$n: training failed at every batch size"; return 1
}

serve_and_eval() {   # serve_and_eval <tag> <model_dir>
    local tag="$1" model="$2"
    if grep -q "\"tag\": \"$tag\"" eval/results/curve/summary.jsonl 2>/dev/null; then status "$tag: already evaluated"; return 0; fi
    status "$tag: serving $model"
    "$VLLM_ENV/bin/vllm" serve "$model" --served-model-name text2sql --dtype bfloat16 \
        --max-model-len 2048 --gpu-memory-utilization 0.84 --kv-cache-memory 536870912 \
        --enforce-eager --max-num-seqs 16 --host 127.0.0.1 --port "$PORT" > "logs/vllm_lc_$tag.log" 2>&1 &
    VLLM_PID=$!
    local waited=0
    until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; do
        kill -0 $VLLM_PID 2>/dev/null || { status "$tag: vLLM exited during startup"; tail -20 "logs/vllm_lc_$tag.log"; return 1; }
        (( waited >= 600 )) && { status "$tag: vLLM not ready"; return 1; }
        sleep 4; waited=$((waited + 4))
    done
    "$VLLM_ENV/bin/python" eval/vllm_eval.py --tag "$tag" --strategy none --model text2sql \
        --url "http://127.0.0.1:$PORT/v1" --out-dir eval/results/curve > "logs/eval_$tag.log" 2>&1
    stop_vllm; sleep 10     # let the GPU memory drain before the next stage
    status "$tag: $("$VLLM_ENV/bin/python" -c '
import json, sys
d = [json.loads(l) for l in open("eval/results/curve/summary.jsonl") if json.loads(l)["tag"] == sys.argv[1]][-1]
print("exec_acc=%.4f errors=%d tiers=%s" % (d["exec_acc"], d["request_errors"], {k: round(v, 3) for k, v in d["by_difficulty"].items()}))
' "$tag")"
    copy_out
}

for n in "${SIZES[@]}"; do
    should_stop && { status "STOP file found: stopping before n=$n"; exit 0; }
    train "$n"
    out="training/outputs_lc/n$n"
    if [ ! -f "$out/merged/config.json" ]; then
        status "n=$n: merging adapter"
        "$TRAIN_ENV/bin/python" scripts/merge_adapter.py --adapter "$out/best_checkpoint" --out "$out/merged" > "logs/merge_n$n.log" 2>&1
    fi
    copy_out
    should_stop && { status "STOP file found: stopping before evaluating n=$n"; exit 0; }
    serve_and_eval "ft$n" "$out/merged"
done

# the existing 7,000-example model, through the identical evaluation path
should_stop && { status "STOP file found: stopping before ft7000"; exit 0; }
if [ -f training/outputs/merged/config.json ]; then serve_and_eval ft7000 training/outputs/merged
else status "no training/outputs/merged (7,000 model) found: skipping ft7000"; fi

copy_out
status "ALL DONE"
