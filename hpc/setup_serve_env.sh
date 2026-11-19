#!/bin/bash
# One-time setup of the SERVING env on a Hazel LOGIN node (needs internet). From the repo root:
#   bash hpc/setup_serve_env.sh
# Separate from the training env because vLLM pins its own torch (0.34.0 -> torch==2.14.0,
# while training runs on 2.13.0).
set -eo pipefail   # no -u: ~/.bashrc and conda's hooks reference unset vars
source "$(dirname "$0")/env.sh"

echo "SERVE_ENV_PREFIX=$SERVE_ENV_PREFIX"

load_conda
if [ ! -d "$SERVE_ENV_PREFIX" ]; then
    echo ">>> Creating conda env…"
    conda create -y -p "$SERVE_ENV_PREFIX" python=3.11 pip
fi
conda activate "$SERVE_ENV_PREFIX"
export PYTHONNOUSERSITE=1
if ! python -m pip --version >/dev/null 2>&1; then
    python -m ensurepip --upgrade || conda install -y -p "$SERVE_ENV_PREFIX" pip
fi
echo ">>> Using python: $(which python)"

# vLLM brings its own torch/transformers (large download; several minutes).
python -m pip install vllm==0.34.0

# What serving/app.py and eval/benchmark.py + scorer.py import.
python -m pip install fastapi "uvicorn[standard]" httpx pydantic pyyaml tqdm rich sqlglot

python - <<'EOF'
import torch, vllm, fastapi, httpx
print("torch", torch.__version__, "| cuda build", torch.version.cuda, "| vllm", vllm.__version__)
EOF
echo "Serving env ready."
