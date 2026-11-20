"""Choosing few-shot examples from the training set: random vs diversity-covering.

`random`  — uniform sample (seeded).
`diverse` — greedy coverage of SQL constructs, Spider difficulty tiers and databases: each
            pick maximises the number of not-yet-covered features, then prefers a database not
            used yet; the seed changes the candidate subset and the tie-breaks.

Both draw from the same pool: training examples whose schema text is at most MAX_SCHEMA_CHARS,
so that multi-shot prompts fit the context window.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import sqlglot
from sqlglot import exp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hardness import spider_hardness  # noqa: E402

MAX_SCHEMA_CHARS = 1500
CANDIDATES_PER_SEED = 400


def sql_features(sql: str) -> frozenset[str]:
    """Constructs used by a query (plus its Spider difficulty tier) as coverage features."""
    feats = {f"tier:{spider_hardness(sql)}"}
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return frozenset(feats)
    checks = {
        "where": exp.Where, "group": exp.Group, "having": exp.Having, "order": exp.Order,
        "limit": exp.Limit, "join": exp.Join, "setop": exp.SetOperation, "distinct": exp.Distinct,
        "like": exp.Like, "in": exp.In, "between": exp.Between, "not": exp.Not,
        "count": exp.Count, "sum": exp.Sum, "avg": exp.Avg, "min": exp.Min, "max": exp.Max,
    }
    for name, node in checks.items():
        if tree.find(node) is not None:
            feats.add(name)
    if len(list(tree.find_all(exp.Select))) > 1:
        feats.add("subquery")
    return frozenset(feats)


def build_pool(train_rows: list[dict], max_schema_chars: int = MAX_SCHEMA_CHARS) -> list[dict]:
    """Training examples with compact schemas, as {db_id, schema_text, question, sql}."""
    return [
        {k: r[k] for k in ("db_id", "schema_text", "question", "sql")}
        for r in train_rows if len(r["schema_text"]) <= max_schema_chars
    ]


def select_random(pool: list[dict], n: int, seed: int) -> list[int]:
    """Indices of n pool examples chosen uniformly at random."""
    return random.Random(seed).sample(range(len(pool)), n)


def select_diverse(pool: list[dict], n: int, seed: int) -> list[int]:
    """Indices of n pool examples greedily covering constructs, tiers and databases."""
    rng = random.Random(seed)
    candidates = rng.sample(range(len(pool)), min(CANDIDATES_PER_SEED, len(pool)))
    feats = {i: sql_features(pool[i]["sql"]) for i in candidates}
    covered: set[str] = set()
    used_dbs: set[str] = set()
    chosen: list[int] = []
    for _ in range(n):
        remaining = [i for i in candidates if i not in chosen]
        rng.shuffle(remaining)       # seeded tie-break
        best = max(
            remaining,
            key=lambda i: (len(feats[i] - covered), pool[i]["db_id"] not in used_dbs),
        )
        chosen.append(best)
        covered |= feats[best]
        used_dbs.add(pool[best]["db_id"])
    return chosen


def select_first(train_rows: list[dict], n: int) -> list[dict]:
    """The original fixed shots: the first n training examples (what the main 3-shot run used)."""
    return [{k: r[k] for k in ("db_id", "schema_text", "question", "sql")} for r in train_rows[:n]]
