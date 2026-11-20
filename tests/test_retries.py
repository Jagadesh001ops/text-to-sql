"""Tests for execution-guided retries and majority voting in run_baseline.

The model is replaced by a fake generator, so no GPU or weights are needed;
the SQL runs against a real (tiny) SQLite database.
"""

import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

import run_baseline  # noqa: E402

BAD = "SELECT pet_type FROM pets"                        # no such column
GOOD = "SELECT PetType FROM pets"                        # -> dog, cat
ALSO_GOOD = "SELECT PetType FROM pets ORDER BY PetType"  # same result set as GOOD
WRONG = "SELECT PetType FROM pets WHERE PetType = 'dog'"  # runs, different result


@pytest.fixture
def setup(tmp_path, monkeypatch):
    db_dir = tmp_path / "spider" / "database" / "pets_1"
    db_dir.mkdir(parents=True)
    with sqlite3.connect(db_dir / "pets_1.sqlite") as conn:
        conn.execute("CREATE TABLE pets (PetType TEXT)")
        conn.executemany("INSERT INTO pets VALUES (?)", [("dog",), ("cat",)])

    dev = tmp_path / "dev.jsonl"
    dev.write_text(json.dumps({
        "db_id": "pets_1", "question": "Pet types?", "sql": GOOD, "schema_text": "Table: pets",
        "prompt": "[SCHEMA]\nTable: pets\n  PetType text\n\n[QUESTION]\nPet types?\n\n[SQL]",
    }) + "\n")

    cfg = {
        "data": {"dev_file": str(dev), "raw_dir": str(tmp_path / "spider")},
        "eval": {"timeout_seconds": 5},
        "serving": {"max_new_tokens": 64},
        "model": {"base_id": "unused", "finetuned_dir": "unused"},
    }
    monkeypatch.setattr(run_baseline, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run_baseline, "load_finetuned_model", lambda *_: (None, None))

    calls = []
    queue: list[str] = []

    def fake_generate_n(model, tokenizer, prompt, n=1, temperature=0.0, max_new_tokens=256):
        calls.append({"prompt": prompt, "n": n, "temperature": temperature})
        out = [queue.pop(0) for _ in range(n)]
        return out, 1.0

    monkeypatch.setattr(run_baseline, "hf_generate_n", fake_generate_n)
    return cfg, calls, queue


def _run(cfg, **kw):
    path = run_baseline.run_evaluation("finetuned", cfg, tag="t", **kw)
    rec, = [json.loads(line) for line in Path(path).read_text().splitlines()]
    return rec


# ── Retries ───────────────────────────────────────────────────────────────────

def test_no_retries_keeps_failing_query(setup):
    cfg, calls, queue = setup
    queue += [BAD]
    rec = _run(cfg)
    assert rec["sql_pred"] == BAD and "failed_attempts" not in rec
    assert len(calls) == 1


def test_feedback_retry_recovers(setup):
    cfg, calls, queue = setup
    queue += [BAD, GOOD]
    rec = _run(cfg, retries=2)
    assert rec["sql_pred"] == GOOD
    assert rec["failed_attempts"] == [{"sql": BAD, "error": "no such column: pet_type"}]
    retry = calls[1]
    assert "[ERROR] The above SQL failed with: no such column: pet_type" in retry["prompt"]
    assert BAD in retry["prompt"] and retry["temperature"] == 0.0
    assert rec["latency_ms"] == 2.0  # latency includes the retry


def test_sample_retry_reuses_original_prompt(setup):
    cfg, calls, queue = setup
    queue += [BAD, GOOD]
    rec = _run(cfg, retries=1, retry_mode="sample", retry_temperature=0.9)
    assert rec["sql_pred"] == GOOD
    assert calls[1]["prompt"] == calls[0]["prompt"] and calls[1]["temperature"] == 0.9


def test_retries_stop_at_limit(setup):
    cfg, calls, queue = setup
    queue += [BAD, BAD, BAD]
    rec = _run(cfg, retries=2)
    assert rec["sql_pred"] == BAD
    assert len(rec["failed_attempts"]) == 2 and len(calls) == 3


def test_query_that_runs_is_not_retried(setup):
    cfg, calls, queue = setup
    queue += [WRONG]
    rec = _run(cfg, retries=2)
    assert rec["sql_pred"] == WRONG and len(calls) == 1  # retries only fix errors


# ── Reusing earlier first attempts ────────────────────────────────────────────

def _first_attempts(tmp_path, sql, question="Pet types?"):
    path = tmp_path / "first.jsonl"
    path.write_text(json.dumps({"db_id": "pets_1", "question": question, "sql_pred": sql,
                                "generation_error": None}) + "\n")
    return path


def test_first_attempt_is_not_regenerated(setup, tmp_path):
    cfg, calls, queue = setup
    queue += ["SHOULD NOT BE USED"]
    rec = _run(cfg, first_attempts=_first_attempts(tmp_path, WRONG))
    assert rec["sql_pred"] == WRONG and calls == []          # model never called


def test_failed_first_attempt_is_retried_with_feedback(setup, tmp_path):
    cfg, calls, queue = setup
    queue += [GOOD]
    rec = _run(cfg, retries=2, first_attempts=_first_attempts(tmp_path, BAD))
    assert rec["sql_pred"] == GOOD and len(calls) == 1       # only the retry was generated
    assert BAD in calls[0]["prompt"] and "[ERROR]" in calls[0]["prompt"]
    assert rec["failed_attempts"] == [{"sql": BAD, "error": "no such column: pet_type"}]


def test_first_attempts_for_a_different_question_are_rejected(setup, tmp_path):
    cfg, calls, queue = setup
    with pytest.raises(ValueError, match="different question"):
        _run(cfg, first_attempts=_first_attempts(tmp_path, GOOD, question="Something else?"))


# ── Voting ────────────────────────────────────────────────────────────────────

def test_vote_picks_majority_result(setup):
    cfg, calls, queue = setup
    # greedy is wrong; two samples agree on the right result (different SQL text)
    queue += [WRONG, GOOD, ALSO_GOOD, BAD]
    rec = _run(cfg, vote_samples=4)
    assert rec["sql_pred"] == GOOD
    assert rec["vote"]["greedy_sql"] == WRONG and rec["vote"]["winner_votes"] == 2
    assert calls[1]["n"] == 3 and calls[1]["temperature"] == 0.7


def test_vote_tie_goes_to_greedy(setup):
    cfg, calls, queue = setup
    queue += [WRONG, GOOD]
    rec = _run(cfg, vote_samples=2)
    assert rec["sql_pred"] == WRONG and rec["vote"]["winner_votes"] == 1


def test_vote_when_nothing_runs_keeps_greedy(setup):
    cfg, calls, queue = setup
    queue += [BAD, BAD, "SELECT nope FROM nowhere"]
    rec = _run(cfg, vote_samples=3)
    assert rec["sql_pred"] == BAD and rec["vote"]["winner_votes"] == 0


def test_predicted_sql_cannot_modify_database(setup, tmp_path):
    cfg, calls, queue = setup
    queue += ["DELETE FROM pets"]
    _run(cfg, retries=1, retry_mode="sample")
    with sqlite3.connect(tmp_path / "spider" / "database" / "pets_1" / "pets_1.sqlite") as conn:
        assert conn.execute("SELECT count(*) FROM pets").fetchone()[0] == 2
