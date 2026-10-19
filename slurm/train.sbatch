#!/bin/bash
#SBATCH --job-name=text2sql_qlora
#SBATCH --partition=gpu_partners
#SBATCH --qos=short_gpu
#SBATCH --gres=gpu:rtx_6000_pro:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/share/csc591f26/jmuthuk/logs/slurm_%x_%j.out

# ── Environment ───────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate /share/csc591f26/jmuthuk/conda/envs/bert-compress

export HF_HOME=/share/csc591f26/jmuthuk/hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=$(cat ~/.wandb_key 2>/dev/null || echo "")

CODE_DIR=/share/csc591f26/jmuthuk/code/text-to-sql
# vllm is serving-only — not needed for training
LOG_DIR=/share/csc591f26/jmuthuk/logs
mkdir -p "$LOG_DIR"
cd "$CODE_DIR"

# ── Sanity checks ─────────────────────────────────────────────────────────────
nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| available', torch.cuda.is_available())"

# ── Data guard ────────────────────────────────────────────────────────────────
if [[ ! -f data/processed/train.jsonl ]]; then
    echo "ERROR: data/processed/train.jsonl not found. Run prepare_data.py from a login node first."
    exit 1
fi

# ── Resume from checkpoint if one exists ─────────────────────────────────────
# With 2-hour wall + --requeue the job restarts automatically after preemption
# or wall-time kill. SFTTrainer checkpoints every save_steps; we pick up the
# latest one so each requeue wastes at most save_steps of compute.
RESUME_ARG=""
LATEST_CKPT=$(ls -td training/outputs/checkpoint-* 2>/dev/null | head -1)
if [[ -n "$LATEST_CKPT" ]]; then
    echo "Resuming from checkpoint: $LATEST_CKPT"
    RESUME_ARG="--resume-from $LATEST_CKPT"
else
    echo "No checkpoint found — starting fresh."
fi

# ── Train ─────────────────────────────────────────────────────────────────────
python training/train.py \
    --config configs/config.yaml \
    --run-name "text2sql-qlora-$(date +%Y%m%d)" \
    $RESUME_ARG

echo "Training job finished (or hit wall time — will requeue if not complete)."
