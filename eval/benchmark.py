"""Benchmark the served system: accuracy, latency, throughput and cost.

Fires the full Spider dev set at the API at several concurrency levels and
reports, per level: p50/p95/p99 latency, queries/second, error rate, and cost
per 1,000 queries. Accuracy is scored once with the execution harness.

Cost is reported two ways: GPU-seconds per 1K queries (no price assumption)
and dollars per 1K queries at --gpu-hourly-usd (an explicit assumption: this
project's GPU time on a university cluster is not billed).

Usage (stack running, see hpc/serve_bench.sbatch):
    python eval/benchmark.py --api-url http://localhost:8000 --label ft
    python eval/benchmark.py --api-url http://localhost:8000 --label ft-exec --execute
    python eval/benchmark.py --concurrency 1,8 --n-requests 100      # quick smoke test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from pathlib import Path

import httpx
from tqdm.asyncio import tqdm_asyncio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from utils import get_logger, load_config  # noqa

sys.path.insert(0, str(Path(__file__).resolve().parent))

log = get_logger("benchmark")
RESULTS_DIR = Path("eval/results")


# ── Pure helpers (unit-tested) ────────────────────────────────────────────────

def percentile(sorted_values: list[float], p: float) -> float:
    """Nearest-rank percentile of an ascending list; p in (0, 1]."""
    if not sorted_values:
        raise ValueError("empty list")
    rank = max(1, math.ceil(p * len(sorted_values)))
    return sorted_values[rank - 1]


def summarize(latencies_ms: list[float], wall_s: float, n_requests: int, n_errors: int) -> dict:
    """Latency/throughput summary for one load-test pass."""
    out = {
        "n_requests": n_requests,
        "wall_time_s": round(wall_s, 2),
        "throughput_qps": round(n_requests / wall_s, 3) if wall_s > 0 else 0.0,
        "error_rate": round(n_errors / n_requests, 4) if n_requests else 0.0,
    }
    if latencies_ms:
        s = sorted(latencies_ms)
        out.update({
            "latency_p50_ms": round(percentile(s, 0.50), 1),
            "latency_p95_ms": round(percentile(s, 0.95), 1),
            "latency_p99_ms": round(percentile(s, 0.99), 1),
            "latency_mean_ms": round(statistics.mean(s), 1),
        })
    return out


def cost_per_1k(qps: float, gpu_hourly_usd: float) -> dict:
    """Cost of serving 1,000 queries on one GPU running at `qps` queries/second."""
    if qps <= 0:
        return {"gpu_seconds_per_1k": None, "usd_per_1k": None}
    gpu_seconds = 1000 / qps
    return {
        "gpu_seconds_per_1k": round(gpu_seconds, 2),
        "usd_per_1k": round(gpu_seconds / 3600 * gpu_hourly_usd, 4),
    }


# ── Load generation ───────────────────────────────────────────────────────────

async def timed_query(
    client: httpx.AsyncClient, api_url: str, record: dict, semaphore: asyncio.Semaphore, execute: bool,
) -> dict:
    """Fire one /query request; record latency and the prediction."""
    payload = {"question": record["question"], "schema_text": record["schema_text"], "execute": execute}
    if execute:
        payload["db_id"] = record["db_id"]
    base = {
        "db_id": record["db_id"], "question": record["question"], "sql_gold": record["sql"],
        "difficulty": record.get("difficulty", "unknown"),
    }
    async with semaphore:
        t0 = time.perf_counter()
        try:
            resp = await client.post(f"{api_url}/query", json=payload, timeout=120.0)
            resp.raise_for_status()
            data = resp.json()
            return {**base, "sql_pred": data.get("sql", ""), "latency_ms": (time.perf_counter() - t0) * 1000,
                    "attempts": data.get("attempts", 1), "error": None}
        except Exception as e:  # noqa: BLE001 — any failure counts against the error rate
            return {**base, "sql_pred": "", "latency_ms": (time.perf_counter() - t0) * 1000,
                    "attempts": 0, "error": str(e) or type(e).__name__}


async def run_load_test(
    client: httpx.AsyncClient, api_url: str, records: list[dict], n_requests: int,
    concurrency: int, execute: bool = False,
) -> tuple[dict, list[dict]]:
    """Send n_requests (cycling through records) with `concurrency` in flight."""
    work = [records[i % len(records)] for i in range(n_requests)]
    semaphore = asyncio.Semaphore(concurrency)
    t0 = time.perf_counter()
    results = await tqdm_asyncio.gather(
        *(timed_query(client, api_url, r, semaphore, execute) for r in work),
        desc=f"concurrency={concurrency}",
    )
    wall = time.perf_counter() - t0
    ok = [r["latency_ms"] for r in results if r["error"] is None]
    metrics = summarize(ok, wall, n_requests, n_requests - len(ok))
    metrics["concurrency"] = concurrency
    if execute:
        retried = sum(1 for r in results if r["attempts"] > 1)
        metrics["retry_rate"] = round(retried / n_requests, 4)
    return metrics, list(results)


# ── Main ──────────────────────────────────────────────────────────────────────

async def main_async(args: argparse.Namespace, cfg: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(cfg["data"]["dev_file"]) as f:
        records = [json.loads(line) for line in f if line.strip()]
    n_requests = args.n_requests or len(records)
    levels = [int(c) for c in args.concurrency.split(",")]
    log.info("%d dev records; %d requests per pass; concurrency levels %s", len(records), n_requests, levels)

    limits = httpx.Limits(max_connections=max(levels) + 8, max_keepalive_connections=max(levels) + 8)
    async with httpx.AsyncClient(limits=limits) as client:
        try:
            health = (await client.get(f"{args.api_url}/health", timeout=10.0)).json()
        except Exception as e:  # noqa: BLE001
            log.error("Cannot reach API at %s: %s", args.api_url, e)
            sys.exit(1)
        log.info("API healthy: %s", health)

        # Warm-up: first requests pay one-off costs (CUDA graphs, caches, connections).
        if args.warmup:
            warm = await run_load_test(client, args.api_url, records, args.warmup, 4, args.execute)
            log.info("Warm-up done (%d requests, p50 %.0f ms)", args.warmup, warm[0].get("latency_p50_ms", 0))

        passes, first_results = [], None
        for level in levels:
            metrics, results = await run_load_test(client, args.api_url, records, n_requests, level, args.execute)
            metrics.update(cost_per_1k(metrics["throughput_qps"], args.gpu_hourly_usd))
            passes.append(metrics)
            first_results = first_results or results
            log.info("c=%d: %.2f QPS, p50 %s ms, p95 %s ms, errors %.1f%%", level, metrics["throughput_qps"],
                     metrics.get("latency_p50_ms"), metrics.get("latency_p95_ms"), 100 * metrics["error_rate"])

    # ── Accuracy: score one pass (greedy decoding, so passes agree) ───────────
    pred_file = RESULTS_DIR / f"benchmark_{args.label}_predictions.jsonl"
    with open(pred_file, "w") as f:
        for r in first_results:
            f.write(json.dumps(r) + "\n")
    from scorer import score_predictions  # noqa
    agg = score_predictions(
        pred_file, Path(cfg["data"]["raw_dir"]), timeout=cfg["eval"]["timeout_seconds"],
        out_path=RESULTS_DIR / f"benchmark_{args.label}_scores.json",
    )

    report = {
        "label": args.label, "execute": args.execute, "gpu": args.gpu_name,
        "gpu_hourly_usd_assumed": args.gpu_hourly_usd, "api_health": health,
        "accuracy": {"exec_acc": agg["overall"]["exec_acc"], "by_difficulty": agg.get("by_difficulty", {})},
        "passes": passes,
    }
    report_path = RESULTS_DIR / f"benchmark_{args.label}_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 78)
    print(f"  {args.label.upper()}  |  GPU: {args.gpu_name}  |  execute={args.execute}  |  "
          f"exec acc {agg['overall']['exec_acc']:.2%}")
    print("=" * 78)
    print(f"  {'conc':>4} {'QPS':>8} {'p50 ms':>8} {'p95 ms':>8} {'p99 ms':>8} {'err':>6} "
          f"{'GPU-s/1K':>9} {'$/1K':>8}")
    for m in passes:
        print(f"  {m['concurrency']:>4} {m['throughput_qps']:>8.2f} {m.get('latency_p50_ms', 0):>8.0f} "
              f"{m.get('latency_p95_ms', 0):>8.0f} {m.get('latency_p99_ms', 0):>8.0f} "
              f"{m['error_rate']:>6.1%} {m['gpu_seconds_per_1k'] or 0:>9.1f} {m['usd_per_1k'] or 0:>8.4f}")
    print(f"  $/1K assumes ${args.gpu_hourly_usd}/GPU-hour.  Report: {report_path}")
    print("=" * 78 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the served text-to-SQL system")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--label", default="finetuned", help="Name for this run's output files")
    parser.add_argument("--concurrency", default="1,4,16,64", help="Comma-separated concurrency levels")
    parser.add_argument("--n-requests", type=int, default=None, help="Requests per pass (default: all dev)")
    parser.add_argument("--warmup", type=int, default=20, help="Untimed warm-up requests")
    parser.add_argument("--execute", action="store_true",
                        help="Ask the API to execute the SQL and self-correct (adds db round-trips + retries)")
    parser.add_argument("--gpu-name", default="unknown", help="Label only, e.g. rtx_6000_pro")
    parser.add_argument("--gpu-hourly-usd", type=float, default=None,
                        help="Assumed $/GPU-hour (default: benchmark.gpu_hourly_cost_usd in config)")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.gpu_hourly_usd is None:
        args.gpu_hourly_usd = cfg["benchmark"]["gpu_hourly_cost_usd"]
    asyncio.run(main_async(args, cfg))


if __name__ == "__main__":
    main()
