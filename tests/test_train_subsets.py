"""Tests for the nested training subsets used by the learning-curve experiment (no GPU)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))

from train import subset_indices  # noqa: E402


def test_subsets_are_nested_for_a_fixed_seed():
    small, large = subset_indices(7000, 1000, seed=0), subset_indices(7000, 3000, seed=0)
    assert set(small) < set(large) and len(small) == 1000 and len(large) == 3000


def test_different_seeds_give_different_subsets():
    assert subset_indices(7000, 1000, seed=0) != subset_indices(7000, 1000, seed=1)


def test_indices_are_unique_sorted_and_in_range():
    idx = subset_indices(500, 120, seed=3)
    assert idx == sorted(set(idx)) and idx[0] >= 0 and idx[-1] < 500


def test_full_size_is_everything_and_invalid_sizes_fail():
    assert subset_indices(50, 50) == list(range(50))
    for bad in (0, 51, -1):
        with pytest.raises(ValueError):
            subset_indices(50, bad)


def test_subset_is_reasonably_spread_across_the_data():
    idx = subset_indices(7000, 1000, seed=0)
    halves = sum(i < 3500 for i in idx)
    assert 400 < halves < 600          # not concentrated at the start or the end
