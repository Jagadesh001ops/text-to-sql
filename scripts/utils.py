"""Shared utilities — config loading, logging, prompt templating."""

from __future__ import annotations

import logging
import re

import yaml
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load configs/config.yaml (or a custom path) and return as dict."""
    config_path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )
    return logging.getLogger(name)


# ── Prompt template ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert SQL assistant. Given a database schema and a natural "
    "language question, write a single valid SQL query that answers the question. "
    "Output ONLY the SQL query — no explanation, no markdown fences."
)


def schema_to_text(tables: list[dict]) -> str:
    """Convert Spider schema tables list into a concise text representation.

    Each table entry has keys: table_name, column_names_original, column_types,
    primary_keys, foreign_keys (indices into column list).

    Returns something like:
        Table: orders
          id INTEGER [PK]
          customer_id INTEGER [FK→customers.id]
          total REAL
    """
    lines: list[str] = []
    for tbl in tables:
        tname = tbl["table_name"]
        lines.append(f"Table: {tname}")
        for col_name, col_type in zip(tbl["column_names"], tbl["column_types"]):
            pk_flag = " [PK]" if col_name in tbl.get("primary_keys", []) else ""
            fk_flag = ""
            for fk in tbl.get("foreign_keys", []):
                if fk["from_col"] == col_name:
                    fk_flag = f" [FK→{fk['to_table']}.{fk['to_col']}]"
            lines.append(f"  {col_name} {col_type}{pk_flag}{fk_flag}")
    return "\n".join(lines)


def build_prompt(schema_text: str, question: str, examples: list[dict] | None = None) -> str:
    """Build the chat-style prompt sent to the model.

    Args:
        schema_text: output of schema_to_text()
        question: natural language question
        examples: optional list of {schema_text, question, sql} dicts for few-shot

    Returns:
        Formatted prompt string (already includes the system message).
    """
    parts = [f"[SYSTEM]\n{SYSTEM_PROMPT}\n"]

    if examples:
        parts.append("[FEW-SHOT EXAMPLES]")
        for ex in examples:
            parts.append(
                f"Schema:\n{ex['schema_text']}\n"
                f"Question: {ex['question']}\n"
                f"SQL: {ex['sql']}\n"
            )

    parts.append(f"[SCHEMA]\n{schema_text}\n\n[QUESTION]\n{question}\n\n[SQL]")
    return "\n".join(parts)


def build_correction_prompt(prompt: str, failed_sql: str, error: str) -> str:
    """Re-prompt after a failed query: the original prompt, the bad SQL and the
    database error, asking for a corrected query. Shared by serving and eval."""
    return (
        f"{prompt}\n{failed_sql}\n\n"
        f"[ERROR] The above SQL failed with: {error}\n"
        f"Please correct the SQL query.\n[SQL]"
    )


# Where the model starts writing past the query: a blank line or the next
# prompt section (it sometimes continues with "[QUESTION]…[SQL]…" or "[SCHEMA]…").
_SQL_STOP_MARKERS = ("\n\n", "\n[", "[QUESTION]", "[SCHEMA]", "[SQL]", "[SYSTEM]", "[ERROR]")

# Passed to model.generate(stop_strings=…) so generation ends where extraction would cut.
GENERATION_STOP_STRINGS = ["\n\n", "\n["]


def _cut_at_semicolon(sql: str) -> str:
    """Truncate at the first ';' that is not inside a quoted literal or identifier."""
    quote: str | None = None
    for i, ch in enumerate(sql):
        if quote:
            if ch == quote:
                quote = None
        elif ch in ("'", '"', "`"):
            quote = ch
        elif ch == ";":
            return sql[:i]
    return sql


def extract_sql(raw_output: str) -> str:
    """Pull a single SQL statement out of raw model output.

    Handles markdown fences (complete, opening-only or closing-only), stray
    backticks, and run-on text where the model keeps generating prompt
    sections or a second query after the first one.
    """
    # drop an opening fence (```sql or ```), then anything from a closing fence on
    raw = re.sub(r"^```(?:sql\b)?\s*", "", raw_output.strip(), flags=re.IGNORECASE)
    raw = raw.split("```", 1)[0]
    raw = raw.strip().strip("`").strip()

    for marker in _SQL_STOP_MARKERS:
        idx = raw.find(marker)
        if idx != -1:
            raw = raw[:idx]

    return _cut_at_semicolon(raw).strip()
