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
import re
import signal
import sqlite3
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from utils import extract_sql, get_logger, load_config  # noqa: E402

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
    conn = None
    try:
        with time_limit(timeout):
            conn = sqlite3.connect(str(db_path))
            # Some Spider DBs (e.g. wta_1) hold non-UTF-8 bytes in TEXT columns
            conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
            # Predicted SQL must never modify the benchmark databases
            conn.execute("PRAGMA query_only = ON")
            rows = conn.execute(sql).fetchall()
            return rows, None
    except TimeoutError:
        return [], "TIMEOUT"
    except sqlite3.Error as e:
        return [], str(e)
    except Exception as e:
        return [], f"UNEXPECTED: {e}"
    finally:
        if conn is not None:
            conn.close()


# ── Result comparison ─────────────────────────────────────────────────────────
# Follows the Spider test-suite convention: row counts must match (duplicates
# count), columns may come back in a different order, and row order only
# matters when the gold query has a top-level ORDER BY.

MAX_COLUMN_PERMUTATIONS = 1000


def _norm_value(v: Any) -> Any:
    """Normalise one cell: lowercase/strip strings, unify 2 and 2.0, round floats."""
    if v is None:
        return None
    if isinstance(v, bytes):
        v = v.decode("utf-8", errors="replace")
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else str(round(v, 6))
    return str(v).strip().lower()


def normalise_rows(rows: list[tuple]) -> list[tuple]:
    return [tuple(_norm_value(v) for v in row) for row in rows]


def has_top_level_order_by(sql: str) -> bool:
    """True if ORDER BY appears outside any parenthesised subquery or literal."""
    depth = 0
    quote: str | None = None
    masked = []
    for ch in sql:
        if quote:
            if ch == quote:
                quote = None
            masked.append(" ")
            continue
        if ch in ("'", '"', "`"):
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(depth - 1, 0)
        masked.append(ch if depth == 0 else " ")
    return re.search(r"\border\s+by\b", "".join(masked), re.IGNORECASE) is not None


def _column_permutations(pred: list[tuple], gold: list[tuple]):
    """Yield column orderings of `pred` whose per-column values match `gold`'s.

    Each yielded tuple maps gold column i -> pred column perm[i].
    """
    n = len(gold[0])
    pred_cols = [Counter(r[j] for r in pred) for j in range(n)]
    gold_cols = [Counter(r[i] for r in gold) for i in range(n)]
    options = [[j for j in range(n) if pred_cols[j] == gold_cols[i]] for i in range(n)]

    yielded = 0

    def backtrack(i: int, used: set[int], perm: list[int]):
        nonlocal yielded
        if yielded >= MAX_COLUMN_PERMUTATIONS:
            return
        if i == n:
            yielded += 1
            yield tuple(perm)
            return
        for j in options[i]:
            if j not in used:
                used.add(j)
                perm.append(j)
                yield from backtrack(i + 1, used, perm)
                perm.pop()
                used.discard(j)

    yield from backtrack(0, set(), [])


def results_match(pred_rows: list[tuple], gold_rows: list[tuple], ordered: bool) -> bool:
    """Compare two query results as multisets of rows (lists if `ordered`),
    allowing the predicted columns to be in any order."""
    if len(pred_rows) != len(gold_rows):
        return False
    if not gold_rows:
        return True
    pred, gold = normalise_rows(pred_rows), normalise_rows(gold_rows)
    if len(pred[0]) != len(gold[0]):
        return False

    gold_counts = None if ordered else Counter(gold)
    for perm in _column_permutations(pred, gold):
        permuted = [tuple(row[j] for j in perm) for row in pred]
        if ordered:
            if permuted == gold:
                return True
        elif Counter(permuted) == gold_counts:
            return True
    return False


def result_signature(rows: list[tuple]) -> tuple:
    """Hashable key for a query result, invariant to row order and column order.

    Results that results_match(..., ordered=False) treats as equal get the same
    key; used to group candidate queries by the answer they return.
    """
    norm = normalise_rows(rows)
    if not norm:
        return ()

    def sortable(v: str | None) -> tuple[bool, str]:
        return (v is None, v or "")

    ncols = len(norm[0])
    col_sigs = [tuple(sorted(sortable(r[j]) for r in norm)) for j in range(ncols)]
    col_order = sorted(range(ncols), key=lambda j: col_sigs[j])
    return (ncols, tuple(sorted(tuple(sortable(r[j]) for j in col_order) for r in norm)))


def normalise_sql(sql: str) -> str:
    """Lowercase, collapse whitespace and drop trailing semicolons."""
    return re.sub(r"\s+", " ", sql.strip().rstrip(";").strip().lower())


# ── Per-example scoring ───────────────────────────────────────────────────────

def score_example(
    db_path: Path,
    sql_pred: str,
    sql_gold: str,
    timeout: int = 10,
) -> dict[str, Any]:
    """Score a single prediction.

    Returns a dict with:
        exec_match  — bool: query results match (see results_match)
        exact_match — bool: SQL strings equal after whitespace/case/semicolon
                      normalisation. This is NOT Spider's exact-set-match metric.
        pred_error  — str | None: execution error for prediction
        gold_error  — str | None: execution error for gold (should always be None)
        exec_ms     — float: time to execute predicted SQL (generation latency
                      stays in the prediction record's latency_ms)
    """
    t0 = time.perf_counter()
    pred_rows, pred_error = execute_sql(db_path, sql_pred, timeout)
    exec_ms = (time.perf_counter() - t0) * 1000

    gold_rows, gold_error = execute_sql(db_path, sql_gold, timeout)

    exec_match = (
        pred_error is None
        and gold_error is None
        and results_match(pred_rows, gold_rows, ordered=has_top_level_order_by(sql_gold))
    )

    exact_match = normalise_sql(sql_pred) == normalise_sql(sql_gold)

    return {
        "exec_match": exec_match,
        "exact_match": exact_match,
        "pred_error": pred_error,
        "gold_error": gold_error,
        "exec_ms": round(exec_ms, 2),
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

        # Re-extract so prediction files written before extract_sql handled
        # run-on output are cleaned too (idempotent on already-clean SQL).
        sql_pred = extract_sql(rec["sql_pred"])
        if sql_pred != rec["sql_pred"]:
            rec = {**rec, "sql_pred": sql_pred, "sql_pred_raw": rec["sql_pred"]}

        scores = score_example(db_path, sql_pred, rec["sql_gold"], timeout)
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
    agg["overall"]["gold_errors"] = sum(r["gold_error"] is not None for r in results)
    agg["overall"]["cleaned_preds"] = sum("sql_pred_raw" in r for r in results)

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
