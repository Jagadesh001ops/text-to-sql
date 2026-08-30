"""Phase 5 — Hyperparameter sweep runner.

Runs 3-4 QLoRA configurations sequentially, evaluates each with the scorer,
and writes a comparison table to eval/results/sweep_results.json.

Usage:
    python training/sweep.py [--config configs/config.yaml]
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from utils import get_logger, load_config  # noqa

log = get_logger("sweep")

# ── Sweep grid ────────────────────────────────────────────────────────────────
# Vary one axis at a time to keep results interpretable.
SWEEP_CONFIGS = [
    {"lora_r": 16, "lr": 2e-4, "epochs": 3, "name": "r16_lr2e4_ep3"},   # baseline
    {"lora_r": 8,  "lr": 2e-4, "epochs": 3, "name": "r8_lr2e4_ep3"},    # smaller rank
    {"lora_r": 16, "lr": 1e-4, "epochs": 3, "name": "r16_lr1e4_ep3"},   # lower LR
    {"lora_r": 16, "lr": 2e-4, "epochs": 5, "name": "r16_lr2e4_ep5"},   # more epochs
]


def run_sweep(config_path: str = "configs/config.yaml") -> None:
    cfg = load_config(config_path)
    spider_dir = Path(cfg["data"]["raw_dir"])
    results: list[dict] = []

    for run in SWEEP_CONFIGS:
        log.info("\n" + "=" * 60)
        log.info("Starting sweep run: %s", run["name"])
        log.info("=" * 60)

        # 1. Fine-tune
        cmd = [
            sys.executable, "training/train.py",
            "--config", config_path,
            "--lora-r", str(run["lora_r"]),
            "--lr", str(run["lr"]),
            "--epochs", str(run["epochs"]),
            "--run-name", f"sweep_{run['name']}",
        ]
        ret = subprocess.run(cmd, check=False)
        if ret.returncode != 0:
            log.error("Training failed for run %s — skipping eval.", run["name"])
            results.append({"name": run["name"], "config": run, "error": "training_failed"})
            continue

        # 2. Generate predictions with fine-tuned model
        pred_cmd = [
            sys.executable, "eval/run_baseline.py",
            "--mode", "finetuned",
            "--config", config_path,
            "--skip-scoring",
        ]
        ret = subprocess.run(pred_cmd, check=False)
        if ret.returncode != 0:
            log.error("Prediction generation failed for run %s.", run["name"])
            results.append({"name": run["name"], "config": run, "error": "eval_failed"})
            continue

        # 3. Score
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
        from scorer import score_predictions  # type: ignore  # noqa: E402
        pred_file = Path("eval/results/finetuned_predictions.jsonl")
        score_path = Path(f"eval/results/sweep_{run['name']}_scores.json")
        agg = score_predictions(pred_file, spider_dir, out_path=score_path)

        results.append({
            "name": run["name"],
            "config": run,
            "exec_acc": agg["overall"]["exec_acc"],
            "exact_acc": agg["overall"]["exact_acc"],
            "error_rate": agg["overall"]["error_rate"],
            "scores_file": str(score_path),
        })

    # ── Summary table ─────────────────────────────────────────────────────────
    out = Path("eval/results/sweep_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    log.info("\n=== SWEEP SUMMARY ===")
    log.info("%-30s  %10s  %10s", "Run", "Exec Acc", "Exact Acc")
    log.info("-" * 55)
    for r in results:
        if "exec_acc" in r:
            log.info("%-30s  %10.2f%%  %10.2f%%", r["name"],
                     r["exec_acc"] * 100, r["exact_acc"] * 100)
        else:
            log.info("%-30s  ERROR: %s", r["name"], r.get("error", "?"))

    log.info("\nFull results → %s", out)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    run_sweep(args.config)
