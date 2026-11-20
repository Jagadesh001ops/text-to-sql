"""Tests for the vLLM evaluation client (fake server, no GPU)."""

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "scripts"))

import vllm_eval  # noqa: E402


def test_request_format_and_stop_limit():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"text": " SELECT 1"}]})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await vllm_eval.generate(c, "http://x/v1", "m", ["p1", "p2"], concurrency=2)

    out = asyncio.run(go())
    assert out == [(" SELECT 1", None)] * 2
    body = seen[0]
    assert body["prompt"] in ("p1", "p2") and body["temperature"] == 0.0 and "messages" not in body
    assert 0 < len(body["stop"]) <= 4          # vLLM answers 400 to more than 4


def test_errors_are_returned_not_raised_and_order_is_kept():
    def handler(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["prompt"]
        return httpx.Response(400) if prompt == "bad" else httpx.Response(200, json={"choices": [{"text": prompt}]})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await vllm_eval.generate(c, "http://x/v1", "m", ["a", "bad", "c"], concurrency=3)

    out = asyncio.run(go())
    assert [t for t, _ in out] == ["a", "", "c"]
    assert out[1][1] is not None and out[0][1] is None


@pytest.mark.parametrize("strategy,n,expected", [("none", 0, 0), ("none", 3, 0), ("first", 2, 2)])
def test_choose_shots_sizes(strategy, n, expected):
    rows = [{"db_id": str(i), "schema_text": "s", "question": f"q{i}", "sql": "SELECT 1", "prompt": "p"} for i in range(4)]
    examples, idx = vllm_eval.choose_shots(strategy, n, 0, rows)
    assert len(examples) == expected == len(idx)


def test_random_and_diverse_use_only_the_compact_pool():
    rows = [{"db_id": f"d{i}", "schema_text": "x" * (10 if i % 2 else 5000), "question": f"q{i}",
             "sql": "SELECT a FROM t WHERE b = 1", "prompt": "p"} for i in range(40)]
    for strategy in ("random", "diverse"):
        examples, _ = vllm_eval.choose_shots(strategy, 3, 1, rows)
        assert len(examples) == 3 and all(len(e["schema_text"]) <= 1500 for e in examples)
