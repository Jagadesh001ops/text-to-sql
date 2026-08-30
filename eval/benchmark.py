"""Phase 7 — Production benchmarking.

Measures the served API on:
  - Execution accuracy  (reuses scorer.py against live API calls)
  - Latency            (p50 / p95 across N requests)
  - Throughput         (queries per second under concurrent load)
  - Cost               (self-hosted GPU vs OpenAI per-token pricing)

Usage:
    # Make sure the serving stack is running (see serving/Dockerfile)
    python eval/benchmark.py \
        --api-url http://localhost:8000 \
        --mode finetuned       # or: openai

    # Quick smoke test (50 requests)
    python eval/benchmark.py --api-url http://localhost:8000 --n-requests 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import httpx
from tqdm.asyncio import tqdm_asyncio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from utils import get_logger, load_config  # noqa

log = get_logger("benchmark")
RESULTS_DIR = Path("eval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Single-request timing ─────────────────────────────────────────────────────

async def timed_query(
    client: httpx.AsyncClient,
    api_url: str,
    record: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Fire one /query request and record latency + result."""
    payload = {
        "question": record["question"],
        "schema_text": record["schema_text"],
        "execute": False,
    }
    async with semaphore:
        t0 = time.perf_counter()
        try:
            resp = await client.post(f"{api_url}/query", json=payload, timeout=60.0)
            resp.raise_for_status()
            data = resp.json()
            latency_ms = (time.perf_counter() - t0) * 1000
            return {
                "db_id": record["db_id"],
                "question": record["question"],
                "sql_pred": data.get("sql", ""),
                "sql_gold": record["sql"],
                "difficulty": record.get("difficulty", "unknown"),
                "latency_ms": latency_ms,
                "error": None,
            }
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            return {
                "db_id": record["db_id"],
                "question": record["question"],
                "sql_pred": "",
                "sql_gold": record["sql"],
                "difficulty": record.get("difficulty", "unknown"),
                "latency_ms": latency_ms,
                "error": str(e),
            }


# ── Throughput test ───────────────────────────────────────────────────────────

async def run_load_test(
    api_url: str,
    records: list[dict],
    n_requests: int,
    n_concurrent: int,
) -> dict:
    """Fire `n_requests` requests with `n_concurrent` concurrency and return metrics."""
    # Use records cyclically if there are fewer than n_requests
    test_records = [records[i % len(records)] for i in range(n_requests)]

    semaphore = asyncio.Semaphore(n_concurrent)
    results: list[dict] = []

    wall_start = time.perf_counter()
    async with httpx.AsyncClient() as client:
        tasks = [
            timed_query(client, api_url, rec, semaphore)
            for rec in test_records
        ]
        results = await tqdm_asyncio.gather(*tasks, desc="Load test")
    wall_time = time.perf_counter() - wall_start

    latencies = [r["latency_ms"] for r in results if r["error"] is None]
    errors = [r for r in results if r["error"]]

    metrics = {
        "n_requests": n_requests,
        "n_concurrent": n_concurrent,
        "wall_time_s": round(wall_time, 2),
        "throughput_qps": round(n_requests / wall_time, 2),
        "error_rate": round(len(errors) / n_requests, 4),
    }

    if latencies:
        sorted_lat = sorted(latencies)
        metrics.update({
            "latency_p50_ms": round(statistics.median(sorted_lat), 1),
            "latency_p95_ms": round(sorted_lat[int(0.95 * len(sorted_lat))], 1),
            "latency_p99_ms": round(sorted_lat[int(0.99 * len(sorted_lat))], 1),
            "latency_mean_ms": round(statistics.mean(sorted_lat), 1),
        })

    return metrics, results


# ── Cost estimation ───────────────────────────────────────────────────────────

def estimate_cost(metrics: dict, cfg: dict) -> dict:
    """Estimate cost per 1,000 queries for self-hosted vs OpenAI."""
    qps = metrics.get("throughput_qps", 1)
    gpu_hourly = cfg["benchmark"]["gpu_hourly_cost_usd"]

    # Self-hosted: GPU cost / throughput
    queries_per_hour = qps * 3600
    self_hosted_per_1k = (gpu_hourly / queries_per_hour) * 1000

    # OpenAI gpt-4o-mini: ~$0.15/1M input + $0.60/1M output tokens
    # Approximate: 400 input tokens (schema+question) + 50 output tokens (SQL)
    openai_input_per_1k = (400 / 1_000_000) * 0.15 * 1000
    openai_output_per_1k = (50 / 1_000_000) * 0.60 * 1000
    openai_per_1k = openai_input_per_1k + openai_output_per_1k

    return {
        "self_hosted_per_1k_usd": round(self_hosted_per_1k, 4),
        "openai_gpt4o_mini_per_1k_usd": round(openai_per_1k, 4),
        "cost_ratio_self_vs_openai": round(self_hosted_per_1k / openai_per_1k, 2),
        "note": (
            "Self-hosted cost assumes GPU is fully utilized. "
            "OpenAI pricing as of model training cutoff — verify current rates."
        ),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

async def main_async(args: argparse.Namespace, cfg: dict) -> None:
    dev_path = Path(cfg["data"]["dev_file"])
    if not dev_path.exists():
        raise FileNotFoundError(f"{dev_path} not found — run prepare_data.py first.")

    records: list[dict] = []
    with open(dev_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    log.info("Loaded %d dev records.", len(records))

    # ── Check API is up ───────────────────────────────────────────────────────
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{args.api_url}/health", timeout=5.0)
            r.raise_for_status()
        log.info("API is healthy at %s", args.api_url)
    except Exception as e:
        log.error("Cannot reach API at %s: %s", args.api_url, e)
        sys.exit(1)

    # ── Load test ─────────────────────────────────────────────────────────────
    log.info(
        "Running load test: %d requests, %d concurrent …",
        args.n_requests, args.n_concurrent,
    )
    perf_metrics, results = await run_load_test(
        args.api_url, records, args.n_requests, args.n_concurrent
    )

    # ── Execution accuracy (score the predictions) ────────────────────────────
    pred_file = RESULTS_DIR / f"benchmark_{args.mode}_predictions.jsonl"
    with open(pred_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    log.info("Scoring predictions …")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from scorer import score_predictions  # type: ignore
    spider_dir = Path(cfg["data"]["raw_dir"])
    agg = score_predictions(
        pred_file, spider_dir,
        timeout=cfg["eval"]["timeout_seconds"],
        out_path=RESULTS_DIR / f"benchmark_{args.mode}_scores.json",
    )

    # ── Cost estimation ───────────────────────────────────────────────────────
    cost = estimate_cost(perf_metrics, cfg)

    # ── Final report ──────────────────────────────────────────────────────────
    report = {
        "mode": args.mode,
        "performance": perf_metrics,
        "accuracy": {
            "exec_acc": agg["overall"]["exec_acc"],
            "exact_acc": agg["overall"]["exact_acc"],
            "by_difficulty": agg.get("by_difficulty", {}),
        },
        "cost_estimate": cost,
    }

    report_path = RESULTS_DIR / f"benchmark_{args.mode}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  BENCHMARK REPORT — {args.mode.upper()}")
    print("=" * 60)
    print(f"  Execution accuracy  : {agg['overall']['exec_acc']:.2%}")
    print(f"  Throughput          : {perf_metrics['throughput_qps']:.1f} QPS")
    print(f"  Latency p50 / p95   : {perf_metrics.get('latency_p50_ms', '?')} / "
          f"{perf_metrics.get('latency_p95_ms', '?')} ms")
    print(f"  Cost per 1K queries : ${cost['self_hosted_per_1k_usd']:.4f} (self-hosted) "
          f"vs ${cost['openai_gpt4o_mini_per_1k_usd']:.4f} (OpenAI)")
    print(f"  Full report         : {report_path}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Production benchmarking")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--mode", default="finetuned", help="Label for this run")
    parser.add_argument("--n-requests", type=int, default=None)
    parser.add_argument("--n-concurrent", type=int, default=None)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.n_requests is None:
        args.n_requests = cfg["benchmark"]["n_requests"]
    if args.n_concurrent is None:
        args.n_concurrent = cfg["benchmark"]["n_concurrent"]

    asyncio.run(main_async(args, cfg))


if __name__ == "__main__":
    main()
