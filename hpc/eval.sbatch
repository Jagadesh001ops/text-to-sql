#!/bin/bash
# Generate + score on the full Spider dev set (1,034 examples). From the repo root:
#   sbatch hpc/eval.sbatch zero-shot
#   sbatch hpc/eval.sbatch few-shot
#   sbatch --dependency=afterok:<TRAIN_JOB_ID> hpc/eval.sbatch finetuned
# Extra args go straight to run_baseline.py, e.g. score a specific checkpoint:
#   sbatch hpc/eval.sbatch finetuned --checkpoint training/outputs/checkpoint-1299 --tag finetuned-final
#SBATCH --job-name=t2sql_eval
#SBATCH --partition=gpu_partners
#SBATCH --qos=short_gpu
#SBATCH --gres=gpu:rtx_6000_pro:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/share/csc591f26/jmuthuk/logs/slurm_%x_%j.out
set -eo pipefail

MODE="${1:?usage: sbatch hpc/eval.sbatch <zero-shot|few-shot|finetuned> [run_baseline.py args...]}"
shift

source hpc/env.sh
activate_env
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$PROJECT_DIR"

echo "=== $(date)  job $SLURM_JOB_ID  mode=$MODE"
nvidia-smi --query-gpu=name,memory.total --format=csv
python eval/run_baseline.py --mode "$MODE" --n-shot 3 "$@"

echo "Results written to eval/results/ (named by --tag if given, else ${MODE})"
