"""Tests for the learning-curve hypothesis checks (synthetic numbers)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

import summarize_curve as C  # noqa: E402


def rec(tag, overall, easy, medium, hard, extra):
    return {"tag": tag, "exec_acc": overall,
            "by_difficulty": {"easy": easy, "medium": medium, "hard": hard, "extra": extra}}


ZS = rec("zs", 0.55, 0.76, 0.57, 0.44, 0.32)


def test_most_gain_early_supports_h21():
    curve = {"ft1000": rec("ft1000", 0.68, 0.87, 0.73, 0.58, 0.40),
             "ft3000": rec("ft3000", 0.705, 0.88, 0.75, 0.60, 0.43),
             "ft7000": rec("ft7000", 0.71, 0.88, 0.76, 0.61, 0.44)}
    res = C.analyse(curve, ZS)
    assert res["gain_fraction"][1000] == round((0.68 - 0.55) / (0.71 - 0.55), 3)
    assert res["checks"]["H2.1"]["supported"] is True


def test_slow_early_gain_falsifies_h21_and_is_flagged():
    curve = {"ft1000": rec("ft1000", 0.60, 0.80, 0.62, 0.48, 0.34),
             "ft3000": rec("ft3000", 0.66, 0.85, 0.70, 0.55, 0.38),
             "ft7000": rec("ft7000", 0.71, 0.88, 0.76, 0.61, 0.44)}
    h = C.analyse(curve, ZS)["checks"]["H2.1"]
    assert h["supported"] is False and h["against_us"]["gain_1000_below_0.4"] is True


def test_h22_needs_both_hard_and_extra_to_beat_easy_beyond_noise():
    base = {"ft3000": rec("ft3000", 0.70, 0.87, 0.74, 0.59, 0.42)}
    gains_in_hard_extra = {**base, "ft1000": rec("ft1000", 0.66, 0.86, 0.70, 0.52, 0.34),
                           "ft7000": rec("ft7000", 0.71, 0.88, 0.76, 0.61, 0.44)}
    assert C.analyse(gains_in_hard_extra, ZS)["checks"]["H2.2"]["supported"] is True
    only_hard = {**base, "ft1000": rec("ft1000", 0.66, 0.86, 0.70, 0.52, 0.43),
                 "ft7000": rec("ft7000", 0.71, 0.88, 0.76, 0.61, 0.44)}
    h = C.analyse(only_hard, ZS)["checks"]["H2.2"]
    assert h["hard_exceeds_easy"] is True and h["extra_exceeds_easy"] is False and h["supported"] is False


def test_differences_inside_the_noise_do_not_count():
    curve = {"ft1000": rec("ft1000", 0.69, 0.865, 0.74, 0.60, 0.43),
             "ft7000": rec("ft7000", 0.71, 0.88, 0.76, 0.61, 0.44)}
    h = C.analyse(curve, ZS)["checks"]["H2.2"]       # hard gains 1.0, easy 1.5: no difference beyond 1.5
    assert h["supported"] is False


def test_missing_full_model_is_reported_not_crashed():
    res = C.analyse({"ft1000": rec("ft1000", 0.66, 0.85, 0.7, 0.5, 0.35)}, ZS)
    assert "note" in res["checks"]
