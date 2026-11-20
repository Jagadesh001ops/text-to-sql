"""Tests for few-shot example selection (no GPU needed)."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "scripts"))

import shots  # noqa: E402

POOL = [
    {"db_id": "a", "schema_text": "Table: t", "question": "q1", "sql": "SELECT count(*) FROM t"},
    {"db_id": "a", "schema_text": "Table: t", "question": "q2", "sql": "SELECT name FROM t WHERE age > 3"},
    {"db_id": "b", "schema_text": "Table: u", "question": "q3",
     "sql": "SELECT T1.a, count(*) FROM u AS T1 JOIN v AS T2 ON T1.id = T2.id GROUP BY T1.a ORDER BY count(*) DESC LIMIT 3"},
    {"db_id": "c", "schema_text": "Table: w", "question": "q4",
     "sql": "SELECT name FROM w WHERE id NOT IN (SELECT id FROM x) UNION SELECT name FROM y"},
    {"db_id": "d", "schema_text": "Table: z", "question": "q5", "sql": "SELECT name FROM z"},
]


def test_features_detect_constructs_and_tier():
    f = shots.sql_features(POOL[2]["sql"])
    assert {"join", "group", "order", "limit", "count"} <= f
    assert any(x.startswith("tier:") for x in f)
    f = shots.sql_features(POOL[3]["sql"])
    assert {"setop", "subquery", "in", "not"} <= f
    assert shots.sql_features("this is not sql ((") >= frozenset()   # unparsable: no crash


def test_pool_drops_large_schemas():
    rows = [{"db_id": "a", "schema_text": "x" * 10, "question": "q", "sql": "s", "prompt": "p", "difficulty": "u"},
            {"db_id": "b", "schema_text": "x" * 5000, "question": "q", "sql": "s", "prompt": "p", "difficulty": "u"}]
    pool = shots.build_pool(rows)
    assert [p["db_id"] for p in pool] == ["a"] and set(pool[0]) == {"db_id", "schema_text", "question", "sql"}


@pytest.mark.parametrize("fn", [shots.select_random, shots.select_diverse])
def test_selection_is_seeded_distinct_and_in_range(fn):
    a, b = fn(POOL, 3, seed=1), fn(POOL, 3, seed=1)
    assert a == b and len(set(a)) == 3 and all(0 <= i < len(POOL) for i in a)


def test_diverse_covers_more_features_than_most_random_draws():
    cover = lambda idx: len(set().union(*(shots.sql_features(POOL[i]["sql"]) for i in idx)))
    diverse = cover(shots.select_diverse(POOL, 2, seed=0))
    randoms = [cover(shots.select_random(POOL, 2, seed=s)) for s in range(30)]
    assert diverse >= max(randoms)          # greedy coverage is at least as good as any random pair here


def test_diverse_first_pick_is_the_most_complex_example():
    first = shots.select_diverse(POOL, 1, seed=0)[0]
    best = max(range(len(POOL)), key=lambda i: len(shots.sql_features(POOL[i]["sql"])))
    assert len(shots.sql_features(POOL[first]["sql"])) == len(shots.sql_features(POOL[best]["sql"]))


def test_diverse_prefers_new_databases_when_coverage_ties():
    pool = [{"db_id": d, "schema_text": "s", "question": "q", "sql": "SELECT name FROM t"} for d in "aabbcc"]
    picks = shots.select_diverse(pool, 3, seed=0)
    assert len({pool[i]["db_id"] for i in picks}) == 3


def test_first_n_matches_the_original_fixed_shots():
    rows = [{"db_id": str(i), "schema_text": "s", "question": f"q{i}", "sql": "x", "prompt": "p"} for i in range(5)]
    assert [r["question"] for r in shots.select_first(rows, 3)] == ["q0", "q1", "q2"]
