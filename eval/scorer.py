"""Phase 2 — Execution-accuracy evaluation harness.

Scores a model's generated SQL against Spider dev databases by actually
executing both the generated and ground-truth queries and comparing result sets.

Usage (standalone):
    python eval/scorer.py \
        --predictions eval/results/baseline_predictions.jsonl \
        --spider-dir data/raw/spider \
        --config configs/config.yaml

Predictions JSONL format (one record per line):
    {"db_id": "...", "question": "...", "sql_pred": "...", "sql_gold": "..."}

Output:
    • Console summary table (accuracy by difficulty tier)
    • eval/results/<run_name>_scores.json  — per-example scores + aggregates
"""

from __future__ import annotations

import argparse
import json
import signal
import sqlite3
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from utils import get_logger, load_config  # noqa: E402

log = get_logger("scorer")


# ── Timeout context manager ────────────────────────────────────────────────────

class TimeoutError(Exception):
    pass


@contextmanager
def time_limit(seconds: int):
    """Cross-platform timeout: uses SIGALRM on Unix, threading.Timer on Windows."""
    if hasattr(signal, "SIGALRM"):
        def handler(signum, frame):
            raise TimeoutError("SQL execution timed out")

        old = signal.signal(signal.SIGALRM, handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
    else:
        import threading

        timer_fired = threading.Event()

        def _fire():
            timer_fired.set()

        timer = threading.Timer(seconds, _fire)
        timer.start()
        try:
            yield
            if timer_fired.is_set():
                raise TimeoutError("SQL execution timed out")
        finally:
            timer.cancel()


# ── SQL execution ─────────────────────────────────────────────────────────────

def execute_sql(db_path: Path, sql: str, timeout: int = 10) -> tuple[list[tuple], str | None]:
    """Execute SQL against a SQLite database.

    Returns:
        (rows, error_message)  — rows is empty list on error.
    """
    try:
        with time_limit(timeout):
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.execute(sql)
            rows = cursor.fetchall()
            conn.close()
            return rows, None
    except TimeoutError:
        return [], "TIMEOUT"
    except sqlite3.Error as e:
        return [], str(e)
    except Exception as e:
        return [], f"UNEXPECTED: {e}"


def normalise_result(rows: list[tuple]) -> frozenset[tuple]:
    """Order-independent result comparison — convert to a frozenset of tuples.

    Each value is lowercased (if string) and stripped to reduce surface-level
    formatting differences.
    """
    normalised = set()
    for row in rows:
        norm_row = tuple(
            str(v).strip().lower() if v is not None else None for v in row
        )
        normalised.add(norm_row)
    return frozenset(normalised)


# ── Per-example scoring ───────────────────────────────────────────────────────

def score_example(
    db_path: Path,
    sql_pred: str,
    sql_gold: str,
    timeout: int = 10,
) -> dict[str, Any]:
    """Score a single prediction.

    Returns a dict with:
        exec_match  — bool: result sets match
        exact_match — bool: SQL strings match (after normalisation)
        pred_error  — str | None: execution error for prediction
        gold_error  — str | None: execution error for gold (should always be None)
        latency_ms  — float: time to execute predicted SQL
    """
    t0 = time.perf_counter()
    pred_rows, pred_error = execute_sql(db_path, sql_pred, timeout)
    latency_ms = (time.perf_counter() - t0) * 1000

    gold_rows, gold_error = execute_sql(db_path, sql_gold, timeout)

    exec_match = (
        pred_error is None
        and gold_error is None
        and normalise_result(pred_rows) == normalise_result(gold_rows)
    )

    # Exact match: normalise whitespace and case
    def _norm_sql(s: str) -> str:
        import re
        return re.sub(r"\s+", " ", s.strip().lower())

    exact_match = _norm_sql(sql_pred) == _norm_sql(sql_gold)

    return {
        "exec_match": exec_match,
        "exact_match": exact_match,
        "pred_error": pred_error,
        "gold_error": gold_error,
        "latency_ms": round(latency_ms, 2),
        "n_pred_rows": len(pred_rows),
        "n_gold_rows": len(gold_rows),
    }


# ── Batch scoring ─────────────────────────────────────────────────────────────

def score_predictions(
    predictions_path: Path,
    spider_dir: Path,
    timeout: int = 10,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Score all predictions in a JSONL file.

    Returns aggregate metrics dict.
    """
    from tqdm import tqdm

    records: list[dict] = []
    with open(predictions_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    log.info("Scoring %d predictions …", len(records))

    results: list[dict] = []
    for rec in tqdm(records, desc="Scoring"):
        db_id = rec["db_id"]
        db_path = spider_dir / "database" / db_id / f"{db_id}.sqlite"

        if not db_path.exists():
            log.warning("DB not found: %s — skipping", db_path)
            continue

        scores = score_example(db_path, rec["sql_pred"], rec["sql_gold"], timeout)
        results.append({**rec, **scores})

    # ── Aggregate ─────────────────────────────────────────────────────────────
    tiers = ["easy", "medium", "hard", "extra", "unknown"]
    agg: dict[str, Any] = {"overall": {}, "by_difficulty": {}}

    def _agg_tier(subset: list[dict]) -> dict:
        if not subset:
            return {"n": 0, "exec_acc": 0.0, "exact_acc": 0.0, "error_rate": 0.0}
        n = len(subset)
        return {
            "n": n,
            "exec_acc": round(sum(r["exec_match"] for r in subset) / n, 4),
            "exact_acc": round(sum(r["exact_match"] for r in subset) / n, 4),
            "error_rate": round(
                sum(r["pred_error"] is not None for r in subset) / n, 4
            ),
        }

    agg["overall"] = _agg_tier(results)

    for tier in tiers:
        subset = [r for r in results if r.get("difficulty", "unknown") == tier]
        if subset:
            agg["by_difficulty"][tier] = _agg_tier(subset)

    # Error analysis
    error_types: dict[str, int] = {}
    for r in results:
        err = r.get("pred_error")
        if err:
            bucket = err.split(":")[0][:40]
            error_types[bucket] = error_types.get(bucket, 0) + 1
    agg["top_errors"] = dict(sorted(error_types.items(), key=lambda x: -x[1])[:10])

    output = {"aggregates": agg, "examples": results}

    # ── Pretty print ──────────────────────────────────────────────────────────
    _print_summary(agg)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        log.info("Scores saved → %s", out_path)

    return agg


def _print_summary(agg: dict) -> None:
    """Print a readable summary table to the console."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Evaluation Results", show_lines=True)
        table.add_column("Split", style="cyan")
        table.add_column("N", justify="right")
        table.add_column("Exec Acc", justify="right", style="green")
        table.add_column("Exact Acc", justify="right")
        table.add_column("Error Rate", justify="right", style="red")

        ov = agg["overall"]
        table.add_row("OVERALL", str(ov["n"]), f"{ov['exec_acc']:.2%}",
                      f"{ov['exact_acc']:.2%}", f"{ov['error_rate']:.2%}")

        for tier, stats in agg.get("by_difficulty", {}).items():
            table.add_row(tier.capitalize(), str(stats["n"]),
                          f"{stats['exec_acc']:.2%}", f"{stats['exact_acc']:.2%}",
                          f"{stats['error_rate']:.2%}")

        console.print(table)
    except ImportError:
        # Fallback if rich not installed
        ov = agg["overall"]
        print(f"\n{'='*50}")
        print(f"OVERALL  exec_acc={ov['exec_acc']:.2%}  "
              f"exact_acc={ov['exact_acc']:.2%}  "
              f"error_rate={ov['error_rate']:.2%}")
        for tier, stats in agg.get("by_difficulty", {}).items():
            print(f"  {tier:10}  exec_acc={stats['exec_acc']:.2%}  n={stats['n']}")
        print("="*50)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SQL predictions")
    parser.add_argument("--predictions", required=True,
                        help="JSONL file with sql_pred and sql_gold fields")
    parser.add_argument("--spider-dir", default="data/raw/spider")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--out", default=None,
                        help="Path to write scores JSON (default: auto-named in eval/results/)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    timeout = cfg["eval"]["timeout_seconds"]

    pred_path = Path(args.predictions)
    out_path = (
        Path(args.out)
        if args.out
        else Path("eval/results") / (pred_path.stem + "_scores.json")
    )

    score_predictions(pred_path, Path(args.spider_dir), timeout=timeout, out_path=out_path)


if __name__ == "__main__":
    main()
