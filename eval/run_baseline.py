"""Generate SQL for the Spider dev set and score it with the execution harness.

Modes:
    zero-shot / few-shot — base Qwen2.5-3B-Instruct, prompting only
    finetuned            — base model + trained LoRA adapter (merged, bf16)

Execution feedback (any mode; uses the dev SQLite DB as a verifier):
    --retries N          if the query fails to run, regenerate up to N times
      --retry-mode feedback   re-prompt with the failed SQL + DB error
      --retry-mode sample     resample the original prompt at --retry-temperature
    --vote-samples K     generate greedy + K-1 samples, execute all, keep the
                         query whose result most candidates agree on

Usage:
    python eval/run_baseline.py --mode zero-shot
    python eval/run_baseline.py --mode few-shot --n-shot 3
    python eval/run_baseline.py --mode finetuned
    python eval/run_baseline.py --mode finetuned --retries 2 --retry-mode feedback --tag ft-retry
    python eval/run_baseline.py --mode finetuned --vote-samples 8 --tag ft-vote8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from utils import (  # noqa
    GENERATION_STOP_STRINGS, build_correction_prompt, build_prompt, extract_sql, get_logger, load_config,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scorer import execute_sql, result_signature, score_predictions  # noqa

log = get_logger("run_baseline")

RESULTS_DIR = Path("eval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Model loading ─────────────────────────────────────────────────────────────

def load_hf_model(model_id: str):
    """Load the base model in bf16 (same precision as the merged fine-tuned model)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Loading model: %s (bf16)", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", trust_remote_code=True, dtype=torch.bfloat16,
    )
    model.eval()
    return model, tokenizer


def load_finetuned_model(base_model_id: str, checkpoint_dir: str):
    """Base model in bf16 with the LoRA adapter merged in."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Loading fine-tuned model from %s", checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id, device_map="auto", trust_remote_code=True, dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, checkpoint_dir).merge_and_unload()
    model.eval()
    return model, tokenizer


# ── Generation ────────────────────────────────────────────────────────────────

def hf_generate_n(
    model, tokenizer, prompt: str, n: int = 1, temperature: float = 0.0, max_new_tokens: int = 256,
) -> tuple[list[str], float]:
    """Generate n outputs for one prompt in a single batched call.

    temperature=0 is greedy (deterministic); >0 samples. Returns (raw_outputs, latency_ms).
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    sampling = {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}
    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=n,
            pad_token_id=tokenizer.eos_token_id,
            stop_strings=GENERATION_STOP_STRINGS,
            tokenizer=tokenizer,
            **sampling,
        )
    latency_ms = (time.perf_counter() - t0) * 1000
    start = inputs["input_ids"].shape[1]
    return [tokenizer.decode(seq[start:], skip_special_tokens=True) for seq in output_ids], latency_ms


# ── Execution feedback ────────────────────────────────────────────────────────

def majority_vote(db_path: Path, candidates: list[str], timeout: int) -> tuple[str, list[dict], int]:
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
    """Run inference on the Spider dev set and write a predictions JSONL.

    Returns the path to the predictions file.
    """
    dev_path = Path(cfg["data"]["dev_file"])
    if not dev_path.exists():
        raise FileNotFoundError(f"{dev_path} not found — run `python scripts/prepare_data.py` first.")

    with open(dev_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    if limit:
        records = records[:limit]
        log.info("Limiting to %d examples (--limit)", limit)
    log.info("Loaded %d dev examples, mode=%s", len(records), mode)

    # ── Model ─────────────────────────────────────────────────────────────────
    base_id = cfg["model"]["base_id"]
    if mode in ("zero-shot", "few-shot"):
        model, tokenizer = load_hf_model(base_id)
    elif mode == "finetuned":
        model, tokenizer = load_finetuned_model(base_id, checkpoint or cfg["model"]["finetuned_dir"])
    else:
        raise ValueError(f"Unknown mode: {mode}")
    max_new = cfg["serving"]["max_new_tokens"]

    def generate_n(prompt: str, n: int = 1, temperature: float = 0.0) -> tuple[list[str], float]:
        return hf_generate_n(model, tokenizer, prompt, n, temperature, max_new)

    # ── Few-shot pool ─────────────────────────────────────────────────────────
    # dev.jsonl prompts are zero-shot. For few-shot, rebuild each prompt with
    # the first n_shot train examples (fixed, so every run sees the same shots).
    shots: list[dict] | None = None
    if mode == "few-shot":
        with open(cfg["data"]["train_file"]) as f:
            shots = [
                {"schema_text": ex["schema_text"], "question": ex["question"], "sql": ex["sql"]}
                for ex in (json.loads(line) for _, line in zip(range(n_shot), f))
            ]
        log.info("Few-shot: using %d train examples per prompt", len(shots))

    pred_file = RESULTS_DIR / f"{tag or mode}_predictions.jsonl"
    db_dir = Path(cfg["data"]["raw_dir"]) / "database"
    timeout = cfg["eval"]["timeout_seconds"]
    latencies: list[float] = []
    n_failed_first = n_recovered = 0
    n_vote_changed = n_vote_none_ran = 0

    def exec_error(db_path: Path, sql: str) -> str | None:
        """Execution error for `sql`, or None if it runs."""
        if not sql:
            return "empty prediction"
        return execute_sql(db_path, sql, timeout)[1]

    with open(pred_file, "w") as fout:
        for rec in tqdm(records, desc=f"Generating ({tag or mode})"):
            prompt = build_prompt(rec["schema_text"], rec["question"], examples=shots) if shots else rec["prompt"]
            db_path = db_dir / rec["db_id"] / f"{rec['db_id']}.sqlite"

            try:
                raws, lat_ms = generate_n(prompt)
                raw_output, error = raws[0], None
            except Exception as e:
                raw_output, lat_ms, error = "", 0.0, str(e)
                log.warning("Generation error for '%s': %s", rec["question"][:60], e)
            sql_pred = extract_sql(raw_output)

            # Majority vote over the greedy prediction plus sampled candidates
            vote: dict | None = None
            if vote_samples > 1 and error is None and db_path.exists():
                candidates = [sql_pred]
                try:
                    sampled, extra_ms = generate_n(prompt, vote_samples - 1, vote_temperature)
                    lat_ms += extra_ms
                    candidates += [extract_sql(r) for r in sampled]
                except Exception as e:
                    log.warning("Sampling error: %s", e)
                greedy_sql = sql_pred
                sql_pred, details, votes = majority_vote(db_path, candidates, timeout)
                n_vote_changed += sql_pred != greedy_sql
                n_vote_none_ran += votes == 0
                vote = {"greedy_sql": greedy_sql, "winner_votes": votes, "candidates": details}

            # Execution-guided retries: regenerate while the query fails to run
            attempts: list[dict] = []
            if retries > 0 and error is None and db_path.exists():
                err = exec_error(db_path, sql_pred)
                n_failed_first += err is not None
                while err and len(attempts) < retries:
                    attempts.append({"sql": sql_pred, "error": err})
                    if retry_mode == "feedback":
                        retry_prompt, temperature = build_correction_prompt(prompt, sql_pred, err), 0.0
                    else:
                        retry_prompt, temperature = prompt, retry_temperature
                    try:
                        raws, extra_ms = generate_n(retry_prompt, 1, temperature)
                    except Exception as e:
                        log.warning("Retry generation error: %s", e)
                        break
                    lat_ms += extra_ms
                    sql_pred = extract_sql(raws[0])
                    err = exec_error(db_path, sql_pred)
                n_recovered += bool(attempts) and err is None

            latencies.append(lat_ms)
            out_rec = {
                "db_id": rec["db_id"],
                "question": rec["question"],
                "sql_pred": sql_pred,
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
    parser.add_argument("--mode", choices=["zero-shot", "few-shot", "finetuned"], default="zero-shot")
    parser.add_argument("--n-shot", type=int, default=3,
                        help="Number of few-shot examples (only used with --mode few-shot)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N examples (for quick smoke tests)")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="LoRA checkpoint dir for --mode finetuned (default: model.finetuned_dir in config)")
    parser.add_argument("--tag", default=None,
                        help="Name for the output files (default: the mode), e.g. ft-vote8 -> ft-vote8_scores.json")
    parser.add_argument("--retries", type=int, default=0,
                        help="Regenerate up to N times while the query fails to execute")
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
        args.mode, cfg, n_shot=args.n_shot, limit=args.limit, checkpoint=args.checkpoint, tag=args.tag,
        retries=args.retries, retry_mode=args.retry_mode, retry_temperature=args.retry_temperature,
        vote_samples=args.vote_samples, vote_temperature=args.vote_temperature,
    )

    if not args.skip_scoring:
        score_predictions(
            pred_file,
            Path(cfg["data"]["raw_dir"]),
            timeout=cfg["eval"]["timeout_seconds"],
            out_path=RESULTS_DIR / f"{args.tag or args.mode}_scores.json",
        )


if __name__ == "__main__":
    main()
