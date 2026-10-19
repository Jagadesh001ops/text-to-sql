"""Tests for SQL extraction and execution-match scoring.

Cases are taken from real ft-final predictions on Spider dev.
Run: python -m pytest tests/
"""

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "eval"))

from scorer import has_top_level_order_by, results_match, score_example  # noqa: E402
from utils import extract_sql  # noqa: E402


# ── extract_sql ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ("SELECT count(*) FROM singer", "SELECT count(*) FROM singer"),
    ("```sql\nSELECT a FROM t\n```", "SELECT a FROM t"),
    ("```SELECT a FROM t```", "SELECT a FROM t"),
    ("SELECT a FROM t\n```", "SELECT a FROM t"),
    # stray leading backtick (dog_kennels)
    ("`\nSELECT charge_type ,  charge_amount FROM Charges", "SELECT charge_type ,  charge_amount FROM Charges"),
    # model keeps writing the prompt format (world_1)
    ("SELECT Code FROM country\n\n[QUESTION]\nWhat is the name of the city?\n\n[SQL]\nSELECT Name FROM city",
     "SELECT Code FROM country"),
    # model starts a new schema block (dog_kennels)
    ("SELECT first_name FROM owners\n\n[SCHEMA]\nTable: Customers", "SELECT first_name FROM owners"),
    ("SELECT a FROM t;", "SELECT a FROM t"),
    ("SELECT a FROM t; SELECT b FROM u", "SELECT a FROM t"),
    # semicolon inside a literal is kept
    ("SELECT a FROM t WHERE b = 'x;y'", "SELECT a FROM t WHERE b = 'x;y'"),
    # multi-line SQL without blank lines is kept whole
    ("SELECT a\nFROM t\nWHERE b = 1", "SELECT a\nFROM t\nWHERE b = 1"),
    ("", ""),
])
def test_extract_sql(raw, expected):
    assert extract_sql(raw) == expected


def test_extract_sql_is_idempotent():
    sql = "SELECT T1.name FROM a AS T1 JOIN b AS T2 ON T1.id = T2.id WHERE T2.x = 'y'"
    assert extract_sql(extract_sql(sql)) == sql


# ── ORDER BY detection ────────────────────────────────────────────────────────

@pytest.mark.parametrize("sql, expected", [
    ("SELECT name FROM singer ORDER BY age DESC", True),
    ("select count(*) from concert where stadium_id = "
     "(select stadium_id from stadium order by capacity desc limit 1)", False),
    ("SELECT name FROM t WHERE note = 'order by me'", False),
    ("SELECT name FROM t", False),
])
def test_has_top_level_order_by(sql, expected):
    assert has_top_level_order_by(sql) is expected


# ── results_match ─────────────────────────────────────────────────────────────

def test_duplicate_rows_must_match():
    # flight_2: 100 copies of the answer vs 1 row used to score as correct
    assert not results_match([("USA",)] * 100, [("USA",)], ordered=False)


def test_column_order_is_ignored():
    pred = [(3, "france"), (2, "netherlands")]
    gold = [("france", 3), ("netherlands", 2)]
    assert results_match(pred, gold, ordered=False)


def test_column_permutation_with_identical_columns():
    pred = [(1, 1, "a"), (2, 2, "b")]
    gold = [("a", 1, 1), ("b", 2, 2)]
    assert results_match(pred, gold, ordered=False)


def test_row_order_ignored_without_order_by():
    assert results_match([("b",), ("a",)], [("a",), ("b",)], ordered=False)


def test_row_order_enforced_with_order_by():
    assert not results_match([("b",), ("a",)], [("a",), ("b",)], ordered=True)
    assert results_match([("a",), ("b",)], [("a",), ("b",)], ordered=True)


def test_wrong_values_rejected():
    assert not results_match([(1,)], [(2,)], ordered=False)


def test_different_column_count_rejected():
    assert not results_match([(1, 2)], [(1,)], ordered=False)


def test_both_empty_match():
    assert results_match([], [], ordered=False)


def test_value_normalisation():
    assert results_match([(2.0, " France ")], [(2, "france")], ordered=False)
    assert results_match([(1 / 3,)], [(0.33333333,)], ordered=False)


# ── score_example against a real SQLite DB ────────────────────────────────────

@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE singer (name TEXT, country TEXT, age INT)")
    conn.executemany("INSERT INTO singer VALUES (?, ?, ?)", [
        ("Joe", "Netherlands", 52), ("Tribal", "United States", 29), ("Ann", "France", 41),
    ])
    conn.execute("CREATE TABLE players (last_name TEXT)")
    # invalid UTF-8 stored as TEXT, like wta_1's 'Treyes Albarrac\xc3N'
    conn.execute("INSERT INTO players VALUES (CAST(X'416C62617272616343' || X'C3' || X'4E' AS TEXT))")
    conn.commit()
    conn.close()
    return path


def test_score_column_order(db):
    r = score_example(db, "SELECT count(*), country FROM singer GROUP BY country",
                      "SELECT country, count(*) FROM singer GROUP BY country")
    assert r["exec_match"] and r["pred_error"] is None


def test_score_bad_column(db):
    r = score_example(db, "SELECT pet_type FROM singer", "SELECT name FROM singer")
    assert not r["exec_match"]
    assert r["pred_error"].startswith("no such column")


def test_score_non_utf8_text(db):
    r = score_example(db, "SELECT last_name FROM players", "SELECT last_name FROM players")
    assert r["gold_error"] is None and r["exec_match"]


def test_predicted_sql_cannot_modify_db(db):
    r = score_example(db, "DELETE FROM singer", "SELECT name FROM singer")
    assert r["pred_error"] is not None
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT count(*) FROM singer").fetchone()[0] == 3


def test_exact_match_ignores_trailing_semicolon(db):
    r = score_example(db, "SELECT name FROM singer", "select name  from singer;")
    assert r["exact_match"]
