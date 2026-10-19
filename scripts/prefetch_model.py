"""Pre-download model weights to the HF cache on a login node.

Compute nodes on the HPC cluster have no internet access. Run this script
once from a login node before submitting the SLURM job.

Usage:
    python scripts/prefetch_model.py [--config configs/config.yaml]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger, load_config

log = get_logger("prefetch_model")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-download model weights")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_id = cfg["model"]["base_id"]

    hf_home = os.getenv("HF_HOME", "/share/hpc-support/rsenthi/bertbase/hf_home")
    os.environ["HF_HOME"] = hf_home
    log.info("HF_HOME = %s", hf_home)

    log.info("Downloading tokenizer and model weights for: %s", model_id)
    log.info("This may take several minutes (~6 GB for Qwen2.5-3B-Instruct) …")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.getenv("HF_TOKEN")

    log.info("  [1/2] Tokenizer …")
    AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=token)

    log.info("  [2/2] Model weights (downloading only, not loading into VRAM) …")
    # Use snapshot_download so we get all shards without allocating GPU memory
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=model_id,
        repo_type="model",
        token=token,
        ignore_patterns=["*.gguf", "*.ggml"],  # skip quantised formats we don't use
    )

    log.info("✓ Model cached. You can now submit the SLURM job:")
    log.info("  sbatch slurm/train.sbatch")


if __name__ == "__main__":
    main()
