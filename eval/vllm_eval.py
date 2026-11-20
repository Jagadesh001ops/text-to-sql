"""Evaluate a model served by vLLM on the Spider dev set, with any prompt style.

Used by the follow-up experiments (docs/experiment_plan.md): few-shot prompts with different
example sets against the base model, and zero-shot prompts against fine-tuned models.
Greedy decoding, same scorer and prompt format as the main results.

    python eval/vllm_eval.py --tag zs --strategy none
    python eval/vllm_eval.py --tag rand3_s0 --strategy random --shots 3 --seed 0
    python eval/vllm_eval.py --tag ft1k --strategy none --model text2sql

Each run appends one JSON line to <out-dir>/summary.jsonl and is skipped if its tag is already
there, so a sweep can be restarted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
from tqdm.asyncio import tqdm_asyncio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from utils import GENERATION_STOP_STRINGS, build_prompt, extract_sql, get_logger, load_config  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shots as shotlib  # noqa: E402
from scorer import score_predictions  # noqa: E402

log = get_logger("vllm_eval")

# vLLM's OpenAI-compatible API accepts at most 4 stop sequences; extract_sql() cuts at all markers.
STOP = GENERATION_STOP_STRINGS[:4]


async def generate(
    client: httpx.AsyncClient, url: str, model: str, prompts: list[str],
    concurrency: int = 16, max_tokens: int = 256,
) -> list[tuple[str, str | None]]:
    """Greedy completions for every prompt -> [(text, error)], in order."""
    sem = asyncio.Semaphore(concurrency)

    async def one(prompt: str) -> tuple[str, str | None]:
        async with sem:
            try:
                resp = await client.post(
                    f"{url}/completions",
                    json={"model": model, "prompt": prompt, "max_tokens": max_tokens,
                          "temperature": 0.0, "stop": STOP},
                    timeout=600.0,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["text"], None
            except Exception as e:  # noqa: BLE001 — counted and reported, not hidden
                return "", f"{type(e).__name__}: {e}"[:200]

    return list(await tqdm_asyncio.gather(*(one(p) for p in prompts), desc="requests"))


def choose_shots(strategy: str, n: int, seed: int, train_rows: list[dict]) -> tuple[list[dict], list[int]]:
    """The few-shot examples for a run, and their indices (into the compact pool, or train rows)."""
    if strategy == "none" or n == 0:
        return [], []
    if strategy == "first":
        return shotlib.select_first(train_rows, n), list(range(n))
    pool = shotlib.build_pool(train_rows)
    pick = {"random": shotlib.select_random, "diverse": shotlib.select_diverse}[strategy]
    idx = pick(pool, n, seed)
    return [pool[i] for i in idx], idx


async def run(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.jsonl"
    if summary_path.exists():
        for line in summary_path.read_text().splitlines():
            if line.strip() and json.loads(line)["tag"] == args.tag:
                log.info("tag %s already in %s, skipping", args.tag, summary_path)
                return json.loads(line)

    with open(cfg["data"]["train_file"]) as f:
        train_rows = [json.loads(line) for line in f if line.strip()]
    with open(cfg["data"]["dev_file"]) as f:
        dev = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        dev = dev[: args.limit]

    examples, idx = choose_shots(args.strategy, args.shots, args.seed, train_rows)
    prompts = [build_prompt(r["schema_text"], r["question"], examples=examples or None) for r in dev]
    log.info("%s: %d questions, %d shots (%s, seed %d)", args.tag, len(dev), len(examples), args.strategy, args.seed)

    t0 = time.perf_counter()
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=args.concurrency + 8)) as client:
        outputs = await generate(client, args.url, args.model, prompts, args.concurrency)
    wall = time.perf_counter() - t0

    pred_path = out_dir / f"{args.tag}_predictions.jsonl"
    n_errors = 0
    with open(pred_path, "w") as f:
        for rec, (text, err) in zip(dev, outputs):
            n_errors += err is not None
            f.write(json.dumps({
                "db_id": rec["db_id"], "question": rec["question"], "sql_pred": extract_sql(text),
                "sql_gold": rec["sql"], "difficulty": "unknown", "request_error": err,
            }) + "\n")
    agg = score_predictions(pred_path, Path(cfg["data"]["raw_dir"]),
                            timeout=cfg["eval"]["timeout_seconds"], out_path=out_dir / f"{args.tag}_scores.json")

    summary = {
        "tag": args.tag, "strategy": args.strategy, "n_shots": len(examples), "seed": args.seed,
        "model": args.model, "n_questions": len(dev), "request_errors": n_errors, "wall_s": round(wall, 1),
        "exec_acc": agg["overall"]["exec_acc"], "exact_acc": agg["overall"]["exact_acc"],
        "error_rate": agg["overall"]["error_rate"],
        "by_difficulty": {k: v["exec_acc"] for k, v in agg["by_difficulty"].items()},
        "shots": [{"db_id": e["db_id"], "question": e["question"], "sql": e["sql"]} for e in examples],
        "shot_pool_indices": idx,
    }
    with open(summary_path, "a") as f:
        f.write(json.dumps(summary) + "\n")
    log.info("%s: exec_acc=%.4f (%d request errors, %.0fs)", args.tag, summary["exec_acc"], n_errors, wall)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True, help="Unique name for this run")
    p.add_argument("--strategy", choices=["none", "first", "random", "diverse"], default="none")
    p.add_argument("--shots", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--url", default="http://127.0.0.1:8001/v1")
    p.add_argument("--model", default="text2sql", help="vLLM --served-model-name")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out-dir", default="eval/results/sweep")
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    if args.strategy in ("first", "random", "diverse") and args.shots < 1:
        p.error("--shots must be >= 1 for this strategy")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
