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

    # Re-prompt with the database error when a predicted query fails to run
    python eval/run_baseline.py --mode finetuned --retries 2
    # …or resample from the original prompt instead
    python eval/run_baseline.py --mode finetuned --retries 2 --retry-mode sample
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
    GENERATION_STOP_STRINGS, build_correction_prompt, build_prompt, extract_sql, get_logger,
    load_config, schema_to_text,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scorer import execute_sql, score_predictions  # noqa

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


def hf_generate(
    model, tokenizer, prompt: str, max_new_tokens: int = 256, temperature: float = 0.0,
) -> tuple[str, float]:
    """Generate SQL from a HuggingFace model. Returns (raw_output, latency_ms).

    temperature=0 is greedy (deterministic SQL); >0 samples (used for retries).
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    sampling = {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}
    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **sampling,
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


def openai_generate(
    client, model_id: str, prompt: str, max_tokens: int = 256, temperature: float = 0.0,
) -> tuple[str, float]:
    """Generate SQL via OpenAI API. Returns (raw_output, latency_ms)."""
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "You are an expert SQL assistant. Output ONLY the SQL query."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
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
    retries: int = 0,
    retry_mode: str = "feedback",
    retry_temperature: float = 0.7,
) -> Path:
    """Run inference on Spider dev set and write predictions JSONL.

    With retries > 0, each predicted query is executed against its database and,
    if it fails, regenerated up to `retries` times:
        feedback — re-prompt with the failed SQL and the database error (greedy)
        sample   — resample from the original prompt at `retry_temperature`

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
        generate_fn = lambda prompt, temperature=0.0: hf_generate(
            model, tokenizer, prompt, cfg["serving"]["max_new_tokens"], temperature)

    elif mode == "finetuned":
        model, tokenizer = load_finetuned_model(checkpoint or cfg["model"]["finetuned_dir"])
        generate_fn = lambda prompt, temperature=0.0: hf_generate(
            model, tokenizer, prompt, cfg["serving"]["max_new_tokens"], temperature)

    elif mode == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        ref_cfg = cfg["reference_model"]
        generate_fn = lambda prompt, temperature=0.0: openai_generate(
            client, ref_cfg["model_id"], prompt, ref_cfg["max_tokens"], temperature)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # ── Few-shot pool (reuse the prompts already baked into dev JSONL) ────────
    # For zero-shot we rebuild without examples; for few-shot we use what's there.
    # (prepare_data.py bakes few-shot examples into dev prompts when --few-shot > 0)

    pred_file = RESULTS_DIR / f"{tag or mode}_predictions.jsonl"
    db_dir = Path(cfg["data"]["raw_dir"]) / "database"
    timeout = cfg["eval"]["timeout_seconds"]
    latencies: list[float] = []
    n_failed_first = n_recovered = 0

    def generate(gen_prompt: str, temperature: float = 0.0) -> tuple[str, str, float, str | None]:
        """Returns (raw_output, sql, latency_ms, generation_error)."""
        try:
            raw, lat = generate_fn(gen_prompt, temperature)
            return raw, extract_sql(raw), lat, None
        except Exception as e:
            log.warning("Generation error: %s", e)
            return "", "", 0.0, str(e)

    def exec_error(db_path: Path, sql: str) -> str | None:
        """Execution error for `sql`, or None if it runs."""
        if not sql:
            return "empty prediction"
        return execute_sql(db_path, sql, timeout)[1]

    with open(pred_file, "w") as fout:
        for rec in tqdm(records, desc=f"Generating ({mode})"):
            prompt = rec["prompt"]
            raw_output, sql_pred, lat_ms, error = generate(prompt)

            # Execution-guided retries: regenerate while the query fails to run
            attempts: list[dict] = []
            db_path = db_dir / rec["db_id"] / f"{rec['db_id']}.sqlite"
            if retries > 0 and error is None and db_path.exists():
                err = exec_error(db_path, sql_pred)
                n_failed_first += err is not None
                while err and len(attempts) < retries:
                    attempts.append({"sql": sql_pred, "error": err})
                    retry_prompt = (
                        build_correction_prompt(prompt, sql_pred, err)
                        if retry_mode == "feedback" else prompt
                    )
                    temperature = retry_temperature if retry_mode == "sample" else 0.0
                    raw_output, sql_pred, extra_ms, error = generate(retry_prompt, temperature)
                    lat_ms += extra_ms
                    if error:
                        break
                    err = exec_error(db_path, sql_pred)
                n_recovered += bool(attempts) and err is None and error is None

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
            if attempts:
                out_rec["failed_attempts"] = attempts
            fout.write(json.dumps(out_rec) + "\n")

    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    p95_lat = sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0
    log.info("Generation complete. avg_latency=%.1f ms  p95=%.1f ms", avg_lat, p95_lat)
    if retries > 0:
        log.info("Retries (%s, max %d): %d/%d first attempts failed to run; %d of those now run",
                 retry_mode, retries, n_failed_first, len(records), n_recovered)
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
    parser.add_argument("--retries", type=int, default=0,
                        help="Regenerate a prediction up to N times when it fails to execute")
    parser.add_argument("--retry-mode", choices=["feedback", "sample"], default="feedback",
                        help="feedback: re-prompt with the DB error; sample: resample the original prompt")
    parser.add_argument("--retry-temperature", type=float, default=0.7,
                        help="Sampling temperature for --retry-mode sample")
    parser.add_argument("--skip-scoring", action="store_true",
                        help="Only generate predictions; don't run the scorer")
    args = parser.parse_args()

    cfg = load_config(args.config)

    pred_file = run_evaluation(
        args.mode, cfg, n_shot=args.n_shot, limit=args.limit,
        checkpoint=args.checkpoint, tag=args.tag,
        retries=args.retries, retry_mode=args.retry_mode,
        retry_temperature=args.retry_temperature,
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
