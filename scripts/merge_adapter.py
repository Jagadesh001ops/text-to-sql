"""Merge a trained LoRA adapter into the bf16 base model and save a standalone model.

vLLM serves a full model directory, not a bare adapter. Merging into the bf16
base is exactly what eval/run_baseline.py does in memory, so the served model
matches the evaluated one.

Usage:
    python scripts/merge_adapter.py                      # best_checkpoint -> training/outputs/merged
    python scripts/merge_adapter.py --adapter training/outputs/checkpoint-1299 --out /tmp/merged
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger, load_config  # noqa: E402

log = get_logger("merge_adapter")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--adapter", default=None, help="Adapter dir (default: model.finetuned_dir)")
    parser.add_argument("--out", default="training/outputs/merged")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_config(args.config)
    adapter = args.adapter or cfg["model"]["finetuned_dir"]
    out = Path(args.out)

    log.info("Merging %s into %s (bf16)", adapter, cfg["model"]["base_id"])
    tokenizer = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["base_id"], dtype=torch.bfloat16, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter).merge_and_unload()

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)   # also writes generation_config.json
    tokenizer.save_pretrained(out)
    log.info("Merged model saved -> %s", out)


if __name__ == "__main__":
    main()
