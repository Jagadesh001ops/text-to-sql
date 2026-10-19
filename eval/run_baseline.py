"""Phase 3 — Baseline benchmarking (before fine-tuning).

Runs the un-fine-tuned base model (and optionally GPT-4o-mini) over the
Spider dev set and scores with the execution-accuracy harness.

Usage:
    # Zero-shot baseline
    python eval/run_baseline.py --mode zero-shot

    # Few-shot baseline (uses --n-shot examples from train split)
    python eval/run_baseline.py --mode few-shot --n-shot 3

    # GPT-4o-mini reference ceiling
    python eval/run_baseline.py --mode openai

    # Fine-tuned model (post-training evaluation)
    python eval/run_baseline.py --mode finetuned
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from utils import (  # noqa
    GENERATION_STOP_STRINGS, build_prompt, extract_sql, get_logger, load_config, schema_to_text,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scorer import score_predictions  # noqa

log = get_logger("run_baseline")

RESULTS_DIR = Path("eval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Model inference helpers ───────────────────────────────────────────────────

def load_hf_model(model_id: str, load_in_4bit: bool = False):
    """Load a HuggingFace model + tokenizer for local inference."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    log.info("Loading model: %s (4-bit=%s)", model_id, load_in_4bit)

    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def hf_generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> tuple[str, float]:
    """Generate SQL from a HuggingFace model. Returns (raw_output, latency_ms)."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy — deterministic SQL
            pad_token_id=tokenizer.eos_token_id,
            # stop when the model starts a new prompt section instead of
            # running on to max_new_tokens
            stop_strings=GENERATION_STOP_STRINGS,
            tokenizer=tokenizer,
        )
    latency_ms = (time.perf_counter() - t0) * 1000
    # Decode only newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True), latency_ms


def load_finetuned_model(checkpoint_dir: str):
    """Load a fine-tuned model with merged LoRA adapters."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Loading fine-tuned model from %s", checkpoint_dir)
    base_model_id = load_config()["model"]["base_id"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, checkpoint_dir)
    model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def openai_generate(client, model_id: str, prompt: str, max_tokens: int = 256) -> tuple[str, float]:
    """Generate SQL via OpenAI API. Returns (raw_output, latency_ms)."""
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "You are an expert SQL assistant. Output ONLY the SQL query."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    return response.choices[0].message.content or "", latency_ms


# ── Main runner ───────────────────────────────────────────────────────────────

def run_evaluation(
    mode: str,
    cfg: dict,
    n_shot: int = 0,
    limit: int | None = None,
    checkpoint: str | None = None,
    tag: str | None = None,
) -> Path:
    """Run inference on Spider dev set and write predictions JSONL.

    Returns path to the predictions file.
    """
    dev_path = Path(cfg["data"]["dev_file"])
    if not dev_path.exists():
        raise FileNotFoundError(
            f"{dev_path} not found — run `python scripts/prepare_data.py` first."
        )

    records: list[dict] = []
    with open(dev_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if limit:
        records = records[:limit]
        log.info("Limiting to %d examples (--limit)", limit)

    log.info("Loaded %d dev examples, mode=%s", len(records), mode)

    # ── Instantiate the appropriate model ─────────────────────────────────────
    if mode in ("zero-shot", "few-shot"):
        model, tokenizer = load_hf_model(cfg["model"]["base_id"], load_in_4bit=True)
        generate_fn = lambda prompt: hf_generate(model, tokenizer, prompt, cfg["serving"]["max_new_tokens"])

    elif mode == "finetuned":
        model, tokenizer = load_finetuned_model(checkpoint or cfg["model"]["finetuned_dir"])
        generate_fn = lambda prompt: hf_generate(model, tokenizer, prompt, cfg["serving"]["max_new_tokens"])

    elif mode == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        ref_cfg = cfg["reference_model"]
        generate_fn = lambda prompt: openai_generate(client, ref_cfg["model_id"], prompt, ref_cfg["max_tokens"])

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # ── Few-shot pool (reuse the prompts already baked into dev JSONL) ────────
    # For zero-shot we rebuild without examples; for few-shot we use what's there.
    # (prepare_data.py bakes few-shot examples into dev prompts when --few-shot > 0)

    pred_file = RESULTS_DIR / f"{tag or mode}_predictions.jsonl"
    latencies: list[float] = []

    with open(pred_file, "w") as fout:
        for rec in tqdm(records, desc=f"Generating ({mode})"):
            prompt = rec["prompt"]

            try:
                raw_output, lat_ms = generate_fn(prompt)
                sql_pred = extract_sql(raw_output)
                error = None
            except Exception as e:
                raw_output = ""
                sql_pred = ""
                lat_ms = 0.0
                error = str(e)
                log.warning("Generation error for '%s': %s", rec["question"][:60], e)

            latencies.append(lat_ms)

            out_rec = {
                "db_id": rec["db_id"],
                "question": rec["question"],
                "sql_pred": sql_pred,
                "raw_output": raw_output,
                "sql_gold": rec["sql"],
                "difficulty": rec.get("difficulty", "unknown"),
                "latency_ms": round(lat_ms, 2),
                "generation_error": error,
            }
            fout.write(json.dumps(out_rec) + "\n")

    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    p95_lat = sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0
    log.info("Generation complete. avg_latency=%.1f ms  p95=%.1f ms", avg_lat, p95_lat)
    log.info("Predictions → %s", pred_file)

    return pred_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["zero-shot", "few-shot", "finetuned", "openai"],
        default="zero-shot",
    )
    parser.add_argument("--n-shot", type=int, default=3,
                        help="Number of few-shot examples (only used with --mode few-shot)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N examples (for quick smoke tests)")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Adapter dir for --mode finetuned (default: config finetuned_dir)")
    parser.add_argument("--tag", default=None,
                        help="Name for result files (default: the mode)")
    parser.add_argument("--skip-scoring", action="store_true",
                        help="Only generate predictions; don't run the scorer")
    args = parser.parse_args()

    cfg = load_config(args.config)

    pred_file = run_evaluation(
        args.mode, cfg, n_shot=args.n_shot, limit=args.limit,
        checkpoint=args.checkpoint, tag=args.tag,
    )

    if not args.skip_scoring:
        score_path = RESULTS_DIR / f"{args.tag or args.mode}_scores.json"
        score_predictions(
            pred_file,
            Path(cfg["data"]["raw_dir"]),
            timeout=cfg["eval"]["timeout_seconds"],
            out_path=score_path,
        )


if __name__ == "__main__":
    main()
