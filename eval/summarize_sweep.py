"""Summarise the few-shot sweep (Experiment 1) and score each hypothesis in docs/experiment_plan.md.

    python eval/summarize_sweep.py [eval/results/sweep/summary.jsonl] [--json results/fewshot_sweep.json]

The hypothesis checks apply the rules exactly as written in the plan; nothing else is concluded.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

TIER_N = {"easy": 248, "medium": 446, "hard": 172, "extra": 168}   # Spider dev tier sizes


def pct(x: float) -> float:
    return round(100 * x, 2)


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def hard_extra(by_tier: dict) -> float:
    """Accuracy over the hard + extra questions together (weighted by tier size)."""
    n = TIER_N["hard"] + TIER_N["extra"]
    return (by_tier["hard"] * TIER_N["hard"] + by_tier["extra"] * TIER_N["extra"]) / n


def cell_stats(runs: list[dict]) -> dict:
    """Mean and sample std over seeds of overall, per-tier and hard+extra accuracy."""
    def agg(values):
        return {"mean": pct(statistics.mean(values)),
                "sd": pct(statistics.stdev(values)) if len(values) > 1 else None,
                "min": pct(min(values)), "max": pct(max(values))}
    out = {"n_runs": len(runs), "overall": agg([r["exec_acc"] for r in runs]),
           "hard_extra": agg([hard_extra(r["by_difficulty"]) for r in runs])}
    for tier in TIER_N:
        out[tier] = agg([r["by_difficulty"][tier] for r in runs])
    return out


def summarize(rows: list[dict]) -> dict:
    cells: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        cells[(r["strategy"], r["n_shots"])].append(r)
    stats = {f"{s}-{n}": cell_stats(v) for (s, n), v in sorted(cells.items())}
    by_tag = {r["tag"]: r for r in rows}
    zs, first3 = by_tag["zs"], by_tag["first3"]

    m = lambda key, field="overall": stats[key][field]["mean"]
    checks: dict[str, dict] = {}

    # H1.1: accuracy rises 0 -> 3 shots, then flattens (6 adds < 1.5 points over 3)
    for strat in ("random", "diverse"):
        rise = m(f"{strat}-3") - pct(zs["exec_acc"])
        gain_3_to_6 = m(f"{strat}-6") - m(f"{strat}-3")
        checks[f"H1.1 {strat}"] = {
            "rises_0_to_3": rise > 0, "gain_0_to_3_points": round(rise, 2),
            "gain_3_to_6_points": round(gain_3_to_6, 2), "supported": rise > 0 and gain_3_to_6 < 1.5}

    # H1.2: diverse beats random on hard+extra by >= 2 points at matched n; no difference on easy
    for n in (1, 3, 6):    # the original, pre-registered sizes
        d_he = m(f"diverse-{n}", "hard_extra") - m(f"random-{n}", "hard_extra")
        d_easy = m(f"diverse-{n}", "easy") - m(f"random-{n}", "easy")
        spread = max(stats[f"diverse-{n}"]["hard_extra"]["sd"], stats[f"random-{n}"]["hard_extra"]["sd"])
        easy_spread = max(stats[f"diverse-{n}"]["easy"]["sd"], stats[f"random-{n}"]["easy"]["sd"])
        checks[f"H1.2 n={n}"] = {
            "hard_extra_diff_points": round(d_he, 2), "seed_sd_points": spread,
            "easy_diff_points": round(d_easy, 2), "easy_seed_sd_points": easy_spread,
            "diff_exceeds_seed_spread": abs(d_he) > spread,
            "supported": d_he >= 2.0 and abs(d_easy) <= easy_spread}

    # H1.3: random-3 reproduces the extra-hard drop (below 0-shot); diverse-3 reduces it
    zs_extra = pct(zs["by_difficulty"]["extra"])
    r3, d3 = m("random-3", "extra"), m("diverse-3", "extra")
    checks["H1.3"] = {
        "zero_shot_extra": zs_extra, "first3_extra": pct(first3["by_difficulty"]["extra"]),
        "random3_extra_mean": r3, "diverse3_extra_mean": d3,
        "random3_reproduces_drop": r3 < zs_extra, "diverse3_reduces_it": d3 > r3,
        "supported": r3 < zs_extra and d3 > r3}
    # Extension (plan, Addendum 2; post-hoc in scope, predictions fixed before the runs)
    if "random-8" in stats and "random-16" in stats:
        d8, d16 = m("random-8") - m("random-6"), m("random-16") - m("random-6")
        checks["H1.4 random plateau persists"] = {
            "random6": m("random-6"), "random8": m("random-8"), "random16": m("random-16"),
            "diff_8_vs_6": round(d8, 2), "diff_16_vs_6": round(d16, 2),
            "supported": abs(d8) <= 1.5 and abs(d16) <= 1.5}
    if "diverse-16" in stats and "random-16" in stats:
        gap16 = m("diverse-16") - m("random-16")
        gap3 = m("diverse-3") - m("random-3")
        checks["H1.5 diverse converges to random"] = {
            "gap_n3": round(gap3, 2), "gap_n16": round(gap16, 2),
            "gap_n8": round(m("diverse-8") - m("random-8"), 2) if "diverse-8" in stats else None,
            "supported": abs(gap16) < 2.0 and abs(gap16) < abs(gap3)}
    return {"cells": stats, "zero_shot": pct(zs["exec_acc"]), "first3": pct(first3["exec_acc"]), "checks": checks}


def print_report(res: dict) -> None:
    print(f"0-shot {res['zero_shot']}%   original first-3 {res['first3']}%\n")
    print("Overall exec acc, mean ± sd over 3 seeds (min–max)")
    print(f"{'cell':<12}{'overall':<24}{'easy':>7}{'medium':>8}{'hard':>7}{'extra':>7}{'hard+extra':>12}")
    for key, c in res["cells"].items():
        o = c["overall"]
        sd = f"±{o['sd']}" if o["sd"] is not None else ""
        print(f"{key:<12}{o['mean']:>6}{sd:<8}({o['min']}–{o['max']})   "
              f"{c['easy']['mean']:>5}{c['medium']['mean']:>8}{c['hard']['mean']:>7}{c['extra']['mean']:>7}"
              f"{c['hard_extra']['mean']:>12}")
    print("\nPre-registered checks")
    for name, c in res["checks"].items():
        print(f"  {name}: {json.dumps(c)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("summary", nargs="?", default="eval/results/sweep/summary.jsonl")
    p.add_argument("--json", default=None, help="Write the aggregated result here")
    args = p.parse_args()
    res = summarize(load(Path(args.summary)))
    print_report(res)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
