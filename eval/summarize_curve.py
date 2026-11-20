"""Summarise the fine-tuning learning curve (Experiment 2) and score H2.1 / H2.2.

    python eval/summarize_curve.py [--curve eval/results/curve/summary.jsonl]
                                   [--sweep eval/results/sweep/summary.jsonl]
                                   [--json results/learning_curve.json]

Rules are the ones in docs/experiment_plan.md (including Addendum 3); nothing else is concluded.
  gain(N)  = (acc(N) - acc(0-shot)) / (acc(7000) - acc(0-shot))
  H2.1     gain(1000) >= 0.6 and gain(3000) >= 0.85
  H2.2     acc(7000) - acc(1000) is larger on hard AND on extra than on easy, each by > 1.5 points
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TIERS = ("easy", "medium", "hard", "extra")
NOISE_POINTS = 1.5


def load(path: Path) -> dict[str, dict]:
    return {r["tag"]: r for r in (json.loads(l) for l in path.read_text().splitlines() if l.strip())}


def analyse(curve: dict[str, dict], zero_shot: dict) -> dict:
    sizes = sorted(int(t[2:]) for t in curve if t.startswith("ft"))
    acc = {0: zero_shot["exec_acc"], **{n: curve[f"ft{n}"]["exec_acc"] for n in sizes}}
    tier = {0: zero_shot["by_difficulty"], **{n: curve[f"ft{n}"]["by_difficulty"] for n in sizes}}
    out: dict = {"sizes": sizes, "accuracy": {n: round(100 * a, 2) for n, a in acc.items()},
                 "tiers": {n: {k: round(100 * v, 2) for k, v in t.items()} for n, t in tier.items()}}
    full = max(sizes)
    if full != 7000 or len(sizes) < 2:
        out["checks"] = {"note": "need ft7000 and at least one smaller size to score the hypotheses"}
        return out

    denominator = acc[full] - acc[0]
    out["gain_fraction"] = {n: round((acc[n] - acc[0]) / denominator, 3) for n in sizes}
    checks: dict = {}
    g1, g3 = out["gain_fraction"].get(1000), out["gain_fraction"].get(3000)
    checks["H2.1"] = {"gain_1000": g1, "gain_3000": g3,
                      "supported": g1 is not None and g3 is not None and g1 >= 0.6 and g3 >= 0.85}
    # falsifier from the plan: gain(1000) < 0.4, or 3000 within one point of 7000
    checks["H2.1"]["against_us"] = {
        "gain_1000_below_0.4": g1 is not None and g1 < 0.4,
        "3000_within_1pt_of_7000": 3000 in acc and abs(acc[full] - acc[3000]) * 100 <= 1.0}

    if 1000 in tier:
        extra_data_gain = {k: round(100 * (tier[full][k] - tier[1000][k]), 2) for k in TIERS}
        checks["H2.2"] = {
            "gain_beyond_1000_points": extra_data_gain,
            "hard_exceeds_easy": extra_data_gain["hard"] - extra_data_gain["easy"] > NOISE_POINTS,
            "extra_exceeds_easy": extra_data_gain["extra"] - extra_data_gain["easy"] > NOISE_POINTS}
        checks["H2.2"]["supported"] = checks["H2.2"]["hard_exceeds_easy"] and checks["H2.2"]["extra_exceeds_easy"]
    out["checks"] = checks
    return out


def print_report(res: dict) -> None:
    print("Learning curve (exec acc %, base 0-shot = size 0)")
    print(f"{'size':>6}{'overall':>9}" + "".join(f"{t:>8}" for t in TIERS))
    for n in [0, *res["sizes"]]:
        print(f"{n:>6}{res['accuracy'][n]:>9}" + "".join(f"{res['tiers'][n][t]:>8}" for t in TIERS))
    if "gain_fraction" in res:
        print("gain fraction of the 0-shot -> 7,000 improvement:", res["gain_fraction"])
    print("checks:", json.dumps(res["checks"], indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--curve", default="eval/results/curve/summary.jsonl")
    p.add_argument("--sweep", default="eval/results/sweep/summary.jsonl")
    p.add_argument("--json", default=None)
    args = p.parse_args()
    res = analyse(load(Path(args.curve)), load(Path(args.sweep))["zs"])
    print_report(res)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
