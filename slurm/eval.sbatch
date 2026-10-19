#!/bin/bash
#SBATCH --job-name=text2sql_eval
#SBATCH --partition=gpu_partners
#SBATCH --qos=short_gpu
#SBATCH --gres=gpu:rtx_6000_pro:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/share/csc591f26/jmuthuk/logs/slurm_%x_%j.out

# Usage: sbatch slurm/eval.sbatch <mode> [checkpoint_dir] [tag] [extra run_baseline.py args...]
#   sbatch slurm/eval.sbatch finetuned training/outputs/best_checkpoint ft-best
#   sbatch slurm/eval.sbatch finetuned training/outputs/checkpoint-1314 ft-final
#   sbatch slurm/eval.sbatch zero-shot "" zero-shot --limit 300

MODE=${1:?mode required}
CKPT=${2:-}
TAG=${3:-$MODE}
shift $(( $# < 3 ? $# : 3 ))

source ~/.bashrc
conda activate /share/csc591f26/jmuthuk/conda/envs/bert-compress

export HF_HOME=/share/csc591f26/jmuthuk/hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd /share/csc591f26/jmuthuk/code/text-to-sql
nvidia-smi --query-gpu=name,memory.total --format=csv

CKPT_ARG=()
[[ -n "$CKPT" ]] && CKPT_ARG=(--checkpoint "$CKPT")

python eval/run_baseline.py --mode "$MODE" --tag "$TAG" "${CKPT_ARG[@]}" "$@"
