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
export PYTHONNOUSERSITE=1   # don't let ~/.local packages shadow the env's
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$WANDB_DIR" "$LOG_DIR"

# Make `conda activate` usable in a non-interactive script. ~/.bashrc often
# returns non-zero when the shell isn't interactive, which kills a `set -e`
# script silently, so source it with errexit off and then load conda's hook.
load_conda() {
    if ! command -v conda >/dev/null 2>&1; then
        set +e
        source ~/.bashrc
        set -e
    fi
    if ! command -v conda >/dev/null 2>&1; then
        echo "ERROR: conda not found (tried ~/.bashrc). Check 'which conda' on the login node."
        return 1
    fi
    eval "$(conda shell.bash hook)"
}

# Serving gets its own env: vLLM pins its own torch version, and installing it
# into the training env would replace the torch that training/eval run on.
export SERVE_ENV_PREFIX="${SERVE_ENV_PREFIX:-$WORK/conda/envs/t2sql-serve}"

# Activate an env in a batch job (default: the training/eval env).
activate_env() {
    local prefix="${1:-$ENV_PREFIX}"
    load_conda
    module load cuda     # Hazel is on CUDA 13.2; older CUDA hangs on the new GPUs
    conda activate "$prefix"
    echo "Activated $prefix (python: $(which python))"
}
