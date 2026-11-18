#!/bin/bash
# One-time setup on a Hazel LOGIN node (needs internet). From the repo root:
#   bash hpc/setup_env.sh
# Creates the conda env, installs pinned deps, makes sure the base model is in
# the HF cache, and copies the Spider data from the earlier clone if present.
set -eo pipefail   # no -u: ~/.bashrc and conda's hooks reference unset vars
source "$(dirname "$0")/env.sh"

echo "PROJECT_DIR=$PROJECT_DIR"
echo "ENV_PREFIX=$ENV_PREFIX"
echo "HF_HOME=$HF_HOME"

source ~/.bashrc
if [ ! -d "$ENV_PREFIX" ]; then
    conda create -y -p "$ENV_PREFIX" python=3.11
fi
conda activate "$ENV_PREFIX"

# CUDA 13.x wheels to match Hazel's 13.2 driver.
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130

# Pinned to the set train.py was smoke-tested against.
pip install \
    transformers==5.16.1 trl==1.12.0 peft==0.20.0 bitsandbytes==0.50.2 \
    accelerate datasets sqlglot pandas numpy tqdm rich pyyaml \
    python-dotenv openai wandb

python -c "import torch, trl, transformers, peft, bitsandbytes; print('torch', torch.__version__, '| cuda build', torch.version.cuda)"

# No-op if the model is already cached from the earlier run.
hf download Qwen/Qwen2.5-3B-Instruct

# Data is gitignored; reuse the copy from the earlier clone instead of re-downloading.
OLD="$WORK/code/text-to-sql/data"
if [ ! -f "$PROJECT_DIR/data/processed/train.jsonl" ] && [ -d "$OLD" ]; then
    echo "Copying data from $OLD"
    cp -r "$OLD" "$PROJECT_DIR/"
fi
ls "$PROJECT_DIR/data/processed"
echo "Spider DBs: $(ls "$PROJECT_DIR/data/raw/spider/database" | wc -l)"

echo "Setup complete."
