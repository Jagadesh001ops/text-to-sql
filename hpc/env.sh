# Shared settings for NC State Hazel. Sourced by setup_env.sh and the sbatch
# scripts. Override any of these by exporting them before you source / sbatch.
#
# /share is scratch: NOT backed up, files untouched for 30 days are deleted.
# Copy eval/results/ and training/outputs/best_checkpoint somewhere safe.

export WORK="${WORK:-/share/csc591f26/jmuthuk}"

# Separate clone so this rerun never touches the earlier run in code/text-to-sql.
export PROJECT_DIR="${PROJECT_DIR:-$WORK/code/text-to-sql-v2}"

# Dedicated env with the versions train.py was tested against. To reuse the
# old env instead: export ENV_PREFIX=$WORK/conda/envs/bert-compress
export ENV_PREFIX="${ENV_PREFIX:-$WORK/conda/envs/t2sql}"

# Same HF cache as before, so Qwen2.5-3B is already downloaded.
export HF_HOME="$WORK/hf_home"
export PIP_CACHE_DIR="$WORK/pip_cache"
export WANDB_DIR="$WORK/wandb"
export LOG_DIR="$WORK/logs"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$WANDB_DIR" "$LOG_DIR"

# Activate the env in a batch job (same pattern as your other Hazel jobs).
activate_env() {
    source ~/.bashrc
    module load cuda     # Hazel is on CUDA 13.2; older CUDA hangs on the new GPUs
    conda activate "$ENV_PREFIX"
}
