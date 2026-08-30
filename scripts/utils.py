"""Shared utilities — config loading, logging, prompt templating."""

from __future__ import annotations

import yaml
import logging
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


def extract_sql(raw_output: str) -> str:
    """Strip any accidental markdown fences from model output."""
    raw = raw_output.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        # drop opening fence (```sql or ```) and closing fence
        inner = []
        for line in lines[1:]:
            if line.strip().startswith("```"):
                break
            inner.append(line)
        return "\n".join(inner).strip()
    return raw
