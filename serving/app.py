"""FastAPI serving layer in front of a vLLM server.

Start vLLM separately (see hpc/serve_bench.sbatch for the full recipe):
    vllm serve training/outputs/merged --served-model-name text2sql \
        --dtype bfloat16 --max-model-len 2048 --port 8001

Then start this API:
    uvicorn serving.app:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /query    — natural-language question + schema -> SQL (optionally executed)
    GET  /health   — liveness check
    GET  /docs     — OpenAPI docs

Safety: execution only ever touches Spider databases under SPIDER_DB_DIR,
addressed by `db_id` (never a caller-supplied path). Connections are opened
read-only, with a query timeout and a row cap.

Environment:
    VLLM_URL            vLLM OpenAI-compatible base URL   (default http://localhost:8001/v1)
    SERVED_MODEL_NAME   --served-model-name given to vLLM (default text2sql)
    SPIDER_DB_DIR       directory of <db_id>/<db_id>.sqlite files
    RETRY_MODE          feedback | sample — how a failed query is retried
"""

from __future__ import annotations

import asyncio
import os
import re
import sqlite3
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from utils import (  # noqa: E402
    GENERATION_STOP_STRINGS, build_correction_prompt, build_prompt, extract_sql, load_config,
)

cfg = load_config()
VLLM_BASE_URL = os.getenv("VLLM_URL", "http://localhost:8001/v1")
SERVED_MODEL = os.getenv("SERVED_MODEL_NAME", "text2sql")
SPIDER_DB_DIR = Path(os.getenv("SPIDER_DB_DIR", "data/raw/spider/database"))
RETRY_MODE = os.getenv("RETRY_MODE", "feedback")
MAX_RETRIES = cfg["serving"]["self_correction_retries"]
MAX_NEW_TOKENS = cfg["serving"]["max_new_tokens"]
GREEDY_TEMPERATURE = cfg["serving"]["temperature"]
SAMPLE_TEMPERATURE = 0.7
QUERY_TIMEOUT_S = cfg["eval"]["timeout_seconds"]
MAX_RESULT_ROWS = 500

_DB_ID = re.compile(r"^[A-Za-z0-9_\-]+$")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One pooled client for the life of the process: per-request clients would
    # pay connection setup on every call and distort latency measurements.
    app.state.http = httpx.AsyncClient(
        timeout=60.0, limits=httpx.Limits(max_connections=256, max_keepalive_connections=256),
    )
    yield
    await app.state.http.aclose()


app = FastAPI(
    title="Text-to-SQL API",
    description="Translate natural-language questions to SQL with a fine-tuned 3B model.",
    version="2.0.0",
    lifespan=lifespan,
)


# ── Models ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural-language question")
    schema_text: str = Field(
        ...,
        description="Database schema as plain text:\n  Table: <name>\n    <col> <type> [PK] [FK→table.col]",
    )
    db_id: str | None = Field(
        None, description="Spider database to run the SQL against (required when execute=true).",
    )
    execute: bool = Field(
        False, description="Run the SQL read-only against db_id; on failure retry with the error.",
    )


class QueryResponse(BaseModel):
    sql: str
    attempts: int = 1
    corrected: bool = False
    correction_error: str | None = None
    execution_error: str | None = None
    result_rows: list[list[Any]] | None = None
    result_columns: list[str] | None = None
    latency_ms: float


# ── vLLM client ───────────────────────────────────────────────────────────────

async def call_vllm(client: httpx.AsyncClient, prompt: str, temperature: float = GREEDY_TEMPERATURE) -> str:
    """Raw-text completion. The model was fine-tuned and evaluated on the plain
    prompt string, so this must NOT go through the chat endpoint (which would wrap
    it in Qwen's chat template and change what the model sees)."""
    resp = await client.post(
        f"{VLLM_BASE_URL}/completions",
        json={
            "model": SERVED_MODEL,
            "prompt": prompt,
            "max_tokens": MAX_NEW_TOKENS,
            "temperature": temperature,
            "stop": GENERATION_STOP_STRINGS,
        },
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["text"]


# ── Safe SQL execution ────────────────────────────────────────────────────────

def resolve_db(db_id: str) -> Path:
    """Map a db_id to its SQLite file under SPIDER_DB_DIR; reject anything else."""
    if not _DB_ID.match(db_id):
        raise HTTPException(status_code=422, detail="Invalid db_id.")
    path = SPIDER_DB_DIR / db_id / f"{db_id}.sqlite"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Unknown database '{db_id}'.")
    return path


def run_sql(
    db_file: Path, sql: str, timeout_s: float = QUERY_TIMEOUT_S, max_rows: int = MAX_RESULT_ROWS,
) -> tuple[list[str], list[list[Any]], str | None]:
    """Execute `sql` read-only with a timeout and a row cap -> (columns, rows, error)."""
    if not sql.strip():
        return [], [], "empty prediction"
    conn = None
    try:
        conn = sqlite3.connect(f"file:{db_file.as_posix()}?mode=ro", uri=True)
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        conn.execute("PRAGMA query_only = ON")
        deadline = time.monotonic() + timeout_s
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        cursor = conn.execute(sql)
        rows = cursor.fetchmany(max_rows)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        return columns, [list(r) for r in rows], None
    except sqlite3.OperationalError as e:
        return [], [], "TIMEOUT" if "interrupted" in str(e) else str(e)
    except sqlite3.Error as e:
        return [], [], str(e)
    finally:
        if conn is not None:
            conn.close()


# ── Generation with execution feedback ────────────────────────────────────────

async def generate_with_correction(
    client: httpx.AsyncClient, schema_text: str, question: str, db_file: Path | None,
) -> QueryResponse:
    """Generate SQL; if it is executed and fails, retry up to MAX_RETRIES times."""
    t0 = time.perf_counter()
    prompt = build_prompt(schema_text, question)
    sql = extract_sql(await call_vllm(client, prompt))
    attempts = 1
    first_error: str | None = None
    columns: list[str] = []
    rows: list[list[Any]] = []
    error: str | None = None

    if db_file is not None:
        columns, rows, error = await asyncio.to_thread(run_sql, db_file, sql)
        while error and attempts <= MAX_RETRIES:
            first_error = first_error or error
            if RETRY_MODE == "feedback":
                raw = await call_vllm(client, build_correction_prompt(prompt, sql, error))
            else:
                raw = await call_vllm(client, prompt, SAMPLE_TEMPERATURE)
            sql = extract_sql(raw)
            attempts += 1
            columns, rows, error = await asyncio.to_thread(run_sql, db_file, sql)

    return QueryResponse(
        sql=sql,
        attempts=attempts,
        corrected=first_error is not None and error is None,
        correction_error=first_error,
        execution_error=error,
        result_rows=rows if db_file is not None and error is None else None,
        result_columns=columns if db_file is not None and error is None else None,
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "vllm_url": VLLM_BASE_URL, "model": SERVED_MODEL, "retry_mode": RETRY_MODE}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    """Translate a question to SQL; optionally execute it read-only and self-correct."""
    db_file = None
    if req.execute:
        if not req.db_id:
            raise HTTPException(status_code=422, detail="`db_id` is required when `execute=true`.")
        db_file = resolve_db(req.db_id)

    try:
        return await generate_with_correction(app.state.http, req.schema_text, req.question, db_file)
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail=f"Cannot reach vLLM at {VLLM_BASE_URL}. Is it running?")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"vLLM error: {e}")
