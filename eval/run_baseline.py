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

    # Majority vote by execution result over greedy + 7 sampled candidates
    python eval/run_baseline.py --mode finetuned --vote-samples 8
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
from scorer import execute_sql, result_signature, score_predictions  # noqa

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


def _hf_generate_n(
    model, tokenizer, prompt: str, max_new_tokens: int, n: int, temperature: float,
) -> tuple[list[str], float]:
    """Generate n outputs for one prompt in a single batched call.

    temperature=0 is greedy (deterministic SQL); >0 samples.
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
            num_return_sequences=n,
            pad_token_id=tokenizer.eos_token_id,
            # stop when the model starts a new prompt section instead of
            # running on to max_new_tokens
            stop_strings=GENERATION_STOP_STRINGS,
            tokenizer=tokenizer,
        )
    latency_ms = (time.perf_counter() - t0) * 1000
    # Decode only newly generated tokens
    start = inputs["input_ids"].shape[1]
    return [tokenizer.decode(seq[start:], skip_special_tokens=True) for seq in output_ids], latency_ms


def hf_generate(
    model, tokenizer, prompt: str, max_new_tokens: int = 256, temperature: float = 0.0,
) -> tuple[str, float]:
    """Generate SQL from a HuggingFace model. Returns (raw_output, latency_ms)."""
    outputs, latency_ms = _hf_generate_n(model, tokenizer, prompt, max_new_tokens, 1, temperature)
    return outputs[0], latency_ms


def hf_sample(
    model, tokenizer, prompt: str, n: int, max_new_tokens: int = 256, temperature: float = 0.7,
) -> tuple[list[str], float]:
    """Sample n SQL candidates for one prompt. Returns (raw_outputs, latency_ms)."""
    return _hf_generate_n(model, tokenizer, prompt, max_new_tokens, n, temperature)


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


def openai_sample(
    client, model_id: str, prompt: str, n: int = 1, max_tokens: int = 256, temperature: float = 0.0,
) -> tuple[list[str], float]:
    """Generate n SQL candidates via OpenAI API. Returns (raw_outputs, latency_ms)."""
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "You are an expert SQL assistant. Output ONLY the SQL query."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        n=n,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    return [c.message.content or "" for c in response.choices], latency_ms


def openai_generate(
    client, model_id: str, prompt: str, max_tokens: int = 256, temperature: float = 0.0,
) -> tuple[str, float]:
    """Generate SQL via OpenAI API. Returns (raw_output, latency_ms)."""
    outputs, latency_ms = openai_sample(client, model_id, prompt, 1, max_tokens, temperature)
    return outputs[0], latency_ms


def majority_vote(
    db_path: Path, candidates: list[str], timeout: int,
) -> tuple[str, list[dict], int]:
    """Pick the candidate whose execution result most candidates agree on.

    Candidates that fail to run get no vote. Ties go to the earliest candidate,
    so the greedy prediction (candidates[0]) wins ties. If none run, the greedy
    prediction is returned with 0 votes.

    Returns (chosen_sql, per-candidate [{sql, error}], winning_vote_count).
    """
    outcomes: dict[str, tuple[tuple | None, str | None]] = {}
    groups: dict[tuple, list[int]] = {}
    details: list[dict] = []
    for i, sql in enumerate(candidates):
        if sql not in outcomes:
            if not sql:
                outcomes[sql] = (None, "empty prediction")
            else:
                rows, err = execute_sql(db_path, sql, timeout)
                outcomes[sql] = (None if err else result_signature(rows), err)
        signature, err = outcomes[sql]
        details.append({"sql": sql, "error": err})
        if err is None:
            groups.setdefault(signature, []).append(i)

    if not groups:
        return candidates[0], details, 0
    winner = max(groups.values(), key=lambda idx: (len(idx), -idx[0]))
    return candidates[winner[0]], details, len(winner)


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
    vote_samples: int = 0,
    vote_temperature: float = 0.7,
) -> Path:
    """Run inference on Spider dev set and write predictions JSONL.

    With vote_samples = K > 1, the greedy prediction plus K-1 samples drawn at
    `vote_temperature` are executed, and the query whose result most candidates
    agree on is kept (see majority_vote).

    With retries > 0, a prediction that still fails to execute is regenerated
    up to `retries` times:
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
        sample_fn = lambda prompt, n, temperature: hf_sample(
            model, tokenizer, prompt, n, cfg["serving"]["max_new_tokens"], temperature)

    elif mode == "finetuned":
        model, tokenizer = load_finetuned_model(checkpoint or cfg["model"]["finetuned_dir"])
        generate_fn = lambda prompt, temperature=0.0: hf_generate(
            model, tokenizer, prompt, cfg["serving"]["max_new_tokens"], temperature)
        sample_fn = lambda prompt, n, temperature: hf_sample(
            model, tokenizer, prompt, n, cfg["serving"]["max_new_tokens"], temperature)

    elif mode == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        ref_cfg = cfg["reference_model"]
        generate_fn = lambda prompt, temperature=0.0: openai_generate(
            client, ref_cfg["model_id"], prompt, ref_cfg["max_tokens"], temperature)
        sample_fn = lambda prompt, n, temperature: openai_sample(
            client, ref_cfg["model_id"], prompt, n, ref_cfg["max_tokens"], temperature)

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
    n_vote_changed = n_vote_none_ran = 0

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
            db_path = db_dir / rec["db_id"] / f"{rec['db_id']}.sqlite"

            # Majority vote over the greedy prediction plus sampled candidates
            vote: dict | None = None
            if vote_samples > 1 and error is None and db_path.exists():
                candidates = [sql_pred]
                try:
                    raws, extra_ms = sample_fn(prompt, vote_samples - 1, vote_temperature)
                    lat_ms += extra_ms
                    candidates += [extract_sql(r) for r in raws]
                except Exception as e:
                    log.warning("Sampling error: %s", e)
                greedy_sql = sql_pred
                sql_pred, details, votes = majority_vote(db_path, candidates, timeout)
                if sql_pred != greedy_sql:
                    raw_output = None  # chosen query came from a sample
                    n_vote_changed += 1
                n_vote_none_ran += votes == 0
                vote = {"greedy_sql": greedy_sql, "winner_votes": votes, "candidates": details}

            # Execution-guided retries: regenerate while the query fails to run
            attempts: list[dict] = []
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
            if vote:
                out_rec["vote"] = vote
            if attempts:
                out_rec["failed_attempts"] = attempts
            fout.write(json.dumps(out_rec) + "\n")

    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    p95_lat = sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0
    log.info("Generation complete. avg_latency=%.1f ms  p95=%.1f ms", avg_lat, p95_lat)
    if vote_samples > 1:
        log.info("Voting (%d candidates, T=%.2f): picked a sampled query over greedy for %d/%d; "
                 "no candidate ran for %d",
                 vote_samples, vote_temperature, n_vote_changed, len(records), n_vote_none_ran)
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
    parser.add_argument("--vote-samples", type=int, default=0,
                        help="Majority-vote over K candidates (greedy + K-1 samples) by execution result")
    parser.add_argument("--vote-temperature", type=float, default=0.7,
                        help="Sampling temperature for --vote-samples candidates")
    parser.add_argument("--skip-scoring", action="store_true",
                        help="Only generate predictions; don't run the scorer")
    args = parser.parse_args()

    cfg = load_config(args.config)

    pred_file = run_evaluation(
        args.mode, cfg, n_shot=args.n_shot, limit=args.limit,
        checkpoint=args.checkpoint, tag=args.tag,
        retries=args.retries, retry_mode=args.retry_mode,
        retry_temperature=args.retry_temperature,
        vote_samples=args.vote_samples, vote_temperature=args.vote_temperature,
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
