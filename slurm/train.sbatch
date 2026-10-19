#!/bin/bash
#SBATCH --job-name=text2sql_qlora
#SBATCH --partition=gpu_partners
#SBATCH --qos=scavenger_gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/share/csc591f26/jmuthuk/logs/slurm_%x_%j.out

# ── Environment ───────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate /share/csc591f26/jmuthuk/conda/envs/text2sql

# Point HF at the shared model cache (avoids re-downloading Qwen weights each run)
export HF_HOME=/share/csc591f26/jmuthuk/hf_home
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=$(cat ~/.wandb_key 2>/dev/null || echo "")

CODE_DIR=/share/csc591f26/jmuthuk/code/text-to-sql
cd "$CODE_DIR"

# ── Sanity checks ─────────────────────────────────────────────────────────────
nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| available', torch.cuda.is_available())"

# ── Data guard ────────────────────────────────────────────────────────────────
# Compute nodes have no internet — data must be ready before the job starts.
# Run scripts/prefetch_model.py and scripts/prepare_data.py from a login node first.
if [[ ! -f data/processed/train.jsonl ]]; then
    echo "ERROR: data/processed/train.jsonl not found."
    echo "Run this from a login node first:"
    echo "  python scripts/prepare_data.py"
    exit 1
fi

if [[ ! -d data/raw/spider/database ]]; then
    echo "ERROR: Spider SQLite databases not found at data/raw/spider/database/"
    echo "Run this from a login node first:"
    echo "  python scripts/prepare_data.py"
    exit 1
fi

# ── Resume from checkpoint if one exists ──────────────────────────────────────
# With scavenger QOS the job can be preempted. SFTTrainer writes checkpoints
# every eval_steps; on requeue we resume from the latest one automatically.
RESUME_ARG=""
LATEST_CKPT=$(ls -td training/outputs/checkpoint-* 2>/dev/null | head -1)
if [[ -n "$LATEST_CKPT" ]]; then
    echo "Resuming from checkpoint: $LATEST_CKPT"
    RESUME_ARG="--resume-from $LATEST_CKPT"
fi

# ── Train ─────────────────────────────────────────────────────────────────────
python training/train.py \
    --config configs/config.yaml \
    --run-name "text2sql-qlora-$(date +%Y%m%d)" \
    $RESUME_ARG

echo "Training complete."
