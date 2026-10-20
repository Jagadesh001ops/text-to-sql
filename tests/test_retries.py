"""Tests for execution-guided retries in run_baseline.run_evaluation.

The model is replaced by a fake generator, so no GPU or weights are needed.
"""

import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

import run_baseline  # noqa: E402

BAD = "SELECT pet_type FROM pets"
GOOD = "SELECT PetType FROM pets"


@pytest.fixture
def setup(tmp_path, monkeypatch):
    db_dir = tmp_path / "spider" / "database" / "pets_1"
    db_dir.mkdir(parents=True)
    with sqlite3.connect(db_dir / "pets_1.sqlite") as conn:
        conn.execute("CREATE TABLE pets (PetType TEXT)")
        conn.execute("INSERT INTO pets VALUES ('dog')")

    dev = tmp_path / "dev.jsonl"
    dev.write_text(json.dumps({
        "db_id": "pets_1", "question": "Pet types?", "sql": GOOD,
        "prompt": "[SCHEMA]\nTable: pets\n  PetType text\n\n[QUESTION]\nPet types?\n\n[SQL]",
    }) + "\n")

    cfg = {
        "data": {"dev_file": str(dev), "raw_dir": str(tmp_path / "spider")},
        "eval": {"timeout_seconds": 5},
        "serving": {"max_new_tokens": 64},
        "model": {"finetuned_dir": "unused"},
    }
    monkeypatch.setattr(run_baseline, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run_baseline, "load_finetuned_model", lambda _: (None, None))

    calls = []

    def fake_generate(model, tokenizer, prompt, max_new_tokens, temperature=0.0):
        calls.append((prompt, temperature))
        return fake_generate.outputs[min(len(calls), len(fake_generate.outputs)) - 1], 1.0

    monkeypatch.setattr(run_baseline, "hf_generate", fake_generate)
    return cfg, calls, fake_generate


def _read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_no_retries_keeps_failing_query(setup):
    cfg, calls, gen = setup
    gen.outputs = [BAD]
    rec, = _read(run_baseline.run_evaluation("finetuned", cfg, tag="t"))
    assert rec["sql_pred"] == BAD and "failed_attempts" not in rec
    assert len(calls) == 1


def test_feedback_retry_recovers(setup):
    cfg, calls, gen = setup
    gen.outputs = [BAD, GOOD]
    rec, = _read(run_baseline.run_evaluation("finetuned", cfg, tag="t", retries=2))
    assert rec["sql_pred"] == GOOD
    assert rec["failed_attempts"] == [{"sql": BAD, "error": "no such column: pet_type"}]
    retry_prompt, temperature = calls[1]
    assert "[ERROR] The above SQL failed with: no such column: pet_type" in retry_prompt
    assert BAD in retry_prompt and temperature == 0.0
    assert rec["latency_ms"] == 2.0


def test_sample_retry_reuses_original_prompt(setup):
    cfg, calls, gen = setup
    gen.outputs = [BAD, GOOD]
    run_baseline.run_evaluation("finetuned", cfg, tag="t", retries=1,
                                retry_mode="sample", retry_temperature=0.7)
    assert calls[1] == (calls[0][0], 0.7)


def test_retries_stop_at_limit(setup):
    cfg, calls, gen = setup
    gen.outputs = [BAD]
    rec, = _read(run_baseline.run_evaluation("finetuned", cfg, tag="t", retries=2))
    assert len(calls) == 3
    assert len(rec["failed_attempts"]) == 2 and rec["sql_pred"] == BAD


def test_working_query_is_not_retried(setup):
    cfg, calls, gen = setup
    gen.outputs = [GOOD]
    rec, = _read(run_baseline.run_evaluation("finetuned", cfg, tag="t", retries=2))
    assert len(calls) == 1 and "failed_attempts" not in rec


# ── Majority voting ───────────────────────────────────────────────────────────

GOOD_ALT = "SELECT DISTINCT PetType FROM pets"   # same result as GOOD
WRONG = "SELECT 'cat' FROM pets"


@pytest.fixture
def sampler(setup, monkeypatch):
    cfg, calls, gen = setup
    sample_calls = []

    def fake_sample(model, tokenizer, prompt, n, max_new_tokens, temperature):
        sample_calls.append((n, temperature))
        return fake_sample.outputs[:n], 5.0

    monkeypatch.setattr(run_baseline, "hf_sample", fake_sample)
    return cfg, gen, fake_sample, sample_calls


def test_vote_picks_majority_result(sampler):
    cfg, gen, sample, sample_calls = sampler
    gen.outputs = [WRONG]
    sample.outputs = [GOOD, GOOD_ALT, BAD]
    rec, = _read(run_baseline.run_evaluation("finetuned", cfg, tag="t", vote_samples=4))
    assert rec["sql_pred"] == GOOD
    assert rec["vote"]["winner_votes"] == 2 and rec["vote"]["greedy_sql"] == WRONG
    assert sample_calls == [(3, 0.7)]
    assert rec["latency_ms"] == 6.0


def test_vote_tie_keeps_greedy(sampler):
    cfg, gen, sample, _ = sampler
    gen.outputs = [WRONG]
    sample.outputs = [GOOD]
    rec, = _read(run_baseline.run_evaluation("finetuned", cfg, tag="t", vote_samples=2))
    assert rec["sql_pred"] == WRONG


def test_vote_skips_failing_candidates(sampler):
    cfg, gen, sample, _ = sampler
    gen.outputs = [BAD]
    sample.outputs = [BAD, BAD, GOOD]
    rec, = _read(run_baseline.run_evaluation("finetuned", cfg, tag="t", vote_samples=4))
    assert rec["sql_pred"] == GOOD and rec["vote"]["winner_votes"] == 1


def test_vote_with_no_runnable_candidate_falls_back_to_retries(sampler):
    cfg, gen, sample, _ = sampler
    gen.outputs = [BAD, GOOD]
    sample.outputs = [BAD]
    rec, = _read(run_baseline.run_evaluation("finetuned", cfg, tag="t", vote_samples=2, retries=1))
    assert rec["vote"]["winner_votes"] == 0
    assert rec["sql_pred"] == GOOD and len(rec["failed_attempts"]) == 1


def test_no_vote_by_default(sampler):
    cfg, gen, sample, sample_calls = sampler
    gen.outputs = [GOOD]
    rec, = _read(run_baseline.run_evaluation("finetuned", cfg, tag="t"))
    assert sample_calls == [] and "vote" not in rec
