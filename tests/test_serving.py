"""Tests for the serving API and the benchmark helpers.

vLLM is replaced by fakes, so no GPU or weights are needed. SQL runs against a
real (tiny) SQLite database.
"""

import asyncio
import json
import sqlite3
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark  # noqa: E402
from serving import app as serving_app  # noqa: E402
from utils import GENERATION_STOP_STRINGS  # noqa: E402

BAD = "SELECT pet_type FROM pets"
GOOD = "SELECT PetType FROM pets"


@pytest.fixture
def api(tmp_path, monkeypatch):
    db_dir = tmp_path / "database" / "pets_1"
    db_dir.mkdir(parents=True)
    with sqlite3.connect(db_dir / "pets_1.sqlite") as conn:
        conn.execute("CREATE TABLE pets (PetType TEXT)")
        conn.executemany("INSERT INTO pets VALUES (?)", [("dog",), ("cat",)])
    monkeypatch.setattr(serving_app, "SPIDER_DB_DIR", tmp_path / "database")

    calls, queue = [], []

    async def fake_call_vllm(client, prompt, temperature=0.0):
        calls.append({"prompt": prompt, "temperature": temperature})
        return queue.pop(0)

    monkeypatch.setattr(serving_app, "call_vllm", fake_call_vllm)
    with TestClient(serving_app.app) as client:
        yield client, calls, queue


def post(client, **kw):
    body = {"question": "Pet types?", "schema_text": "Table: pets\n  PetType text"} | kw
    return client.post("/query", json=body)


# ── vLLM request format ───────────────────────────────────────────────────────

def test_call_vllm_sends_raw_prompt_to_completions_endpoint():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["body"] = str(request.url), json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"text": " SELECT 1"}]})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await serving_app.call_vllm(c, "[SQL]", 0.0)

    assert asyncio.run(go()) == " SELECT 1"
    assert seen["url"].endswith("/completions") and "chat" not in seen["url"]
    # raw text, not chat messages: a chat template would change what the model sees
    assert seen["body"]["prompt"] == "[SQL]" and "messages" not in seen["body"]
    assert seen["body"]["stop"] == GENERATION_STOP_STRINGS
    assert seen["body"]["model"] == serving_app.SERVED_MODEL


# ── /query ────────────────────────────────────────────────────────────────────

def test_generate_only_does_not_touch_the_database(api):
    client, calls, queue = api
    queue += ["SELECT count(*) FROM pets;\n\n[QUESTION]\nnext"]   # run-on output gets cut
    r = post(client)
    assert r.status_code == 200
    assert r.json()["sql"] == "SELECT count(*) FROM pets"
    assert r.json()["result_rows"] is None and len(calls) == 1


def test_execute_returns_rows(api):
    client, calls, queue = api
    queue += [GOOD]
    r = post(client, execute=True, db_id="pets_1").json()
    assert r["result_columns"] == ["PetType"] and sorted(r["result_rows"]) == [["cat"], ["dog"]]
    assert r["attempts"] == 1 and r["corrected"] is False


def test_failed_query_is_retried_with_the_error(api):
    client, calls, queue = api
    queue += [BAD, GOOD]
    r = post(client, execute=True, db_id="pets_1").json()
    assert r["sql"] == GOOD and r["corrected"] is True and r["attempts"] == 2
    assert r["correction_error"] == "no such column: pet_type"
    assert "[ERROR] The above SQL failed with: no such column: pet_type" in calls[1]["prompt"]
    assert calls[1]["temperature"] == 0.0


def test_sample_retry_mode_resamples_the_original_prompt(api, monkeypatch):
    client, calls, queue = api
    monkeypatch.setattr(serving_app, "RETRY_MODE", "sample")
    queue += [BAD, GOOD]
    r = post(client, execute=True, db_id="pets_1").json()
    assert r["corrected"] is True
    assert calls[1]["prompt"] == calls[0]["prompt"] and calls[1]["temperature"] == 0.7


def test_retries_are_bounded(api):
    client, calls, queue = api
    queue += [BAD, BAD, BAD]
    r = post(client, execute=True, db_id="pets_1").json()
    assert r["attempts"] == 1 + serving_app.MAX_RETRIES
    assert r["corrected"] is False and r["execution_error"] and r["result_rows"] is None


# ── Safety ────────────────────────────────────────────────────────────────────

def test_execute_requires_db_id(api):
    client, _, _ = api
    assert post(client, execute=True).status_code == 422


@pytest.mark.parametrize("db_id", ["../pets_1", "..\\pets_1", "pets_1/../../x", "a b", "/etc/passwd"])
def test_db_id_cannot_escape_the_database_directory(api, db_id):
    client, _, _ = api
    assert post(client, execute=True, db_id=db_id).status_code in (404, 422)


def test_unknown_db_is_404(api):
    client, _, _ = api
    assert post(client, execute=True, db_id="nope").status_code == 404


def test_run_sql_is_read_only(tmp_path):
    db = tmp_path / "d.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE t (x INT)")
        conn.execute("INSERT INTO t VALUES (1)")
    for stmt in ("DELETE FROM t", "DROP TABLE t", "INSERT INTO t VALUES (2)"):
        _, _, err = serving_app.run_sql(db, stmt)
        assert err is not None, stmt
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT count(*) FROM t").fetchone()[0] == 1


def test_run_sql_times_out(tmp_path):
    db = tmp_path / "d.sqlite"
    sqlite3.connect(db).close()
    runaway = "WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM c) SELECT count(*) FROM c"
    _, _, err = serving_app.run_sql(db, runaway, timeout_s=0.3)
    assert err == "TIMEOUT"


def test_run_sql_caps_rows(tmp_path):
    db = tmp_path / "d.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE t (x INT)")
        conn.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(50)])
    _, rows, err = serving_app.run_sql(db, "SELECT x FROM t", max_rows=10)
    assert err is None and len(rows) == 10


# ── Benchmark helpers ─────────────────────────────────────────────────────────

def test_percentile_nearest_rank():
    data = list(range(1, 101))
    assert benchmark.percentile(data, 0.50) == 50
    assert benchmark.percentile(data, 0.95) == 95
    assert benchmark.percentile(data, 0.99) == 99
    assert benchmark.percentile([7], 0.95) == 7
    assert benchmark.percentile([1, 2], 0.99) == 2     # never indexes past the end


def test_summarize():
    s = benchmark.summarize([100.0, 200.0, 300.0, 400.0], wall_s=2.0, n_requests=5, n_errors=1)
    assert s["throughput_qps"] == 2.5 and s["error_rate"] == 0.2
    assert s["latency_p50_ms"] == 200.0 and s["latency_p95_ms"] == 400.0


def test_cost_per_1k():
    c = benchmark.cost_per_1k(qps=10.0, gpu_hourly_usd=3.6)
    assert c["gpu_seconds_per_1k"] == 100.0           # 1000 queries at 10 QPS
    assert c["usd_per_1k"] == pytest.approx(0.1)      # 100 s at $3.6/h
    assert benchmark.cost_per_1k(0.0, 3.6)["usd_per_1k"] is None


def test_load_test_counts_requests_and_errors():
    records = [{"db_id": "d", "question": f"q{i}", "schema_text": "s", "sql": "SELECT 1"} for i in range(3)]
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if json.loads(request.content)["question"] == "q2":
            return httpx.Response(500)
        return httpx.Response(200, json={"sql": "SELECT 1", "attempts": 1})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await benchmark.run_load_test(c, "http://api", records, n_requests=7, concurrency=3)

    metrics, results = asyncio.run(go())
    assert state["n"] == 7 and len(results) == 7        # cycles through 3 records
    assert metrics["concurrency"] == 3
    assert metrics["error_rate"] == pytest.approx(2 / 7, abs=1e-4)  # q2 appears twice in 7 (rate is rounded)
    assert all(r["sql_pred"] == "" for r in results if r["error"])
