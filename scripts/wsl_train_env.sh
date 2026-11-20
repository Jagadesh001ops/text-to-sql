#!/bin/bash
# Create the QLoRA TRAINING environment inside WSL2 (separate from the vLLM one, which pins its
# own torch). Pinned to the versions train.py was tested against (same as hpc/setup_env.sh).
#   bash scripts/wsl_train_env.sh          # idempotent: does nothing if the env already imports
set -eo pipefail
ENVDIR="${TRAIN_ENV:-$HOME/train-env}"
mkdir -p "$HOME/logs_env"

pip_retry() {   # WSL's DNS occasionally hiccups
    local log="$1"; shift
    for attempt in 1 2 3 4; do
        if "$ENVDIR/bin/python" -m pip install "$@" > "$log" 2>&1; then return 0; fi
        echo "pip failed (attempt $attempt): $(tail -1 "$log" | cut -c1-100); retrying in 20s"; sleep 20
    done
    return 1
}

if "$ENVDIR/bin/python" -c "import torch, trl, peft, bitsandbytes, transformers, datasets, yaml" 2>/dev/null; then
    echo "train env already complete: $ENVDIR"
else
    echo "creating train env at $ENVDIR"
    rm -rf "$ENVDIR"
    python3 -m venv --without-pip "$ENVDIR"
    [ -s "$HOME/get-pip.py" ] || curl -sSL https://bootstrap.pypa.io/get-pip.py -o "$HOME/get-pip.py"
    "$ENVDIR/bin/python" "$HOME/get-pip.py" -q
    pip_retry "$HOME/logs_env/pip_torch.log" torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
    pip_retry "$HOME/logs_env/pip_libs.log" transformers==5.16.1 trl==1.12.0 peft==0.20.0 bitsandbytes==0.50.2 \
        accelerate datasets pyyaml sqlglot tqdm rich pytest
fi
"$ENVDIR/bin/python" - <<'PY' 2>&1 | grep -v Warn
import torch, transformers, trl, peft, bitsandbytes
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| gpu", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
      "| transformers", transformers.__version__, "| trl", trl.__version__, "| peft", peft.__version__,
      "| bitsandbytes", bitsandbytes.__version__)
PY
