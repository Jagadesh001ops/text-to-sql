#!/bin/bash
# QLoRA fine-tune Qwen2.5-3B on Spider. From the repo root on Hazel:
#   sbatch hpc/train.sbatch
# short_gpu caps jobs at 2h. Slurm does NOT requeue on a wall-time kill, so
# Slurm sends USR1 5 min before the limit and the trap below requeues the job;
# the next run resumes from the newest complete checkpoint (every save_steps).
#SBATCH --job-name=t2sql_train
#SBATCH --partition=gpu_partners
#SBATCH --qos=short_gpu
#SBATCH --gres=gpu:rtx_6000_pro:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --output=/share/csc591f26/jmuthuk/logs/slurm_%x_%j.out
set -eo pipefail

source hpc/env.sh
activate_env
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # compute nodes hang on network
export WANDB_API_KEY=$(cat ~/.wandb_key 2>/dev/null || echo "")
# Log W&B to local files ($WANDB_DIR); upload later from the login node with:
#   wandb sync /share/csc591f26/jmuthuk/wandb/wandb/offline-run-*
export WANDB_MODE=offline
cd "$PROJECT_DIR"

echo "=== $(date)  job $SLURM_JOB_ID  restart #${SLURM_RESTART_COUNT:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; assert torch.cuda.is_available(), 'No GPU visible'; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

if [[ ! -f data/processed/train.jsonl ]]; then
    echo "ERROR: data/processed/train.jsonl missing. Run bash hpc/setup_env.sh first."
    exit 1
fi

# Newest checkpoint that finished saving (trainer_state.json is written last).
RESUME_ARG=()
LATEST_CKPT=""
for d in $(ls -td training/outputs/checkpoint-* 2>/dev/null); do
    if [[ -f "$d/trainer_state.json" ]]; then LATEST_CKPT="$d"; break; fi
done
if [[ -n "$LATEST_CKPT" ]]; then
    echo "Resuming from $LATEST_CKPT"
    RESUME_ARG=(--resume-from "$LATEST_CKPT")
else
    echo "No checkpoint found, starting fresh."
fi

REQUEUED=0
on_usr1() {
    echo "=== $(date) near wall time, requeueing job $SLURM_JOB_ID"
    REQUEUED=1
    scontrol requeue "$SLURM_JOB_ID"
}
trap on_usr1 USR1

# Run in the background and wait, so bash can handle USR1 while Python runs.
python training/train.py --run-name "t2sql-qlora-${SLURM_JOB_ID}" "${RESUME_ARG[@]}" &
PY_PID=$!
wait "$PY_PID" || STATUS=$?
if [[ "$REQUEUED" == 1 ]]; then
    exit 0
fi
if [[ -n "${STATUS:-}" ]]; then
    echo "train.py failed with exit code $STATUS"
    exit "$STATUS"
fi

ls -la training/outputs/best_checkpoint
echo "=== Training complete $(date)"
