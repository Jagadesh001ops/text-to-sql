"""Tests for the sweep summarizer's hypothesis checks, on synthetic runs."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

import summarize_sweep as S  # noqa: E402


def run(tag, strategy, n, acc, tier=None):
    tier = tier or {"easy": acc, "medium": acc, "hard": acc, "extra": acc}
    return {"tag": tag, "strategy": strategy, "n_shots": n, "exec_acc": acc, "by_difficulty": tier}


def rows(random_by_n, diverse_by_n, zs=0.5):
    out = [run("zs", "none", 0, zs), run("first3", "first", 3, zs + 0.04)]
    for n, accs in random_by_n.items():
        out += [run(f"rand{n}_s{i}", "random", n, a) for i, a in enumerate(accs)]
    for n, accs in diverse_by_n.items():
        out += [run(f"div{n}_s{i}", "diverse", n, a) for i, a in enumerate(accs)]
    return out


def test_plateau_and_convergence_are_detected():
    sizes = (1, 3, 6, 8, 16)
    rand = {n: [0.60, 0.61, 0.59] for n in sizes}                       # flat from n=1
    div = {n: ([0.55, 0.56, 0.54] if n <= 6 else [0.60, 0.605, 0.595]) for n in sizes}
    res = S.summarize(rows(rand, div))
    assert res["checks"]["H1.4 random plateau persists"]["supported"] is True
    assert res["checks"]["H1.5 diverse converges to random"]["supported"] is True   # gap 0 at n=16 < gap at n=3


def test_continued_gains_falsify_the_plateau():
    sizes = (1, 3, 6, 8, 16)
    rand = {n: [x, x, x] for n, x in zip(sizes, (0.55, 0.58, 0.60, 0.62, 0.66))}   # keeps rising
    div = {n: [0.55, 0.55, 0.55] for n in sizes}
    res = S.summarize(rows(rand, div))
    assert res["checks"]["H1.4 random plateau persists"]["supported"] is False
    assert res["checks"]["H1.5 diverse converges to random"]["supported"] is False  # gap stays large


def test_extension_checks_are_absent_without_the_extra_runs():
    sizes = (1, 3, 6)
    res = S.summarize(rows({n: [0.6, 0.6, 0.6] for n in sizes}, {n: [0.55, 0.55, 0.55] for n in sizes}))
    assert not any(k.startswith(("H1.4", "H1.5")) for k in res["checks"])


def test_hard_extra_is_size_weighted():
    assert abs(S.hard_extra({"hard": 1.0, "extra": 0.0}) - 172 / 340) < 1e-9
