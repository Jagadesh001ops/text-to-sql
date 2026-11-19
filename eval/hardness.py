"""Spider difficulty (easy / medium / hard / extra) for a gold SQL query.

Spider's dev.json ships without difficulty labels; the official evaluation
script derives them from the gold SQL with a fixed set of counting rules
(Yu et al., 2018). This is a reimplementation of those rules on a sqlglot
parse tree, so results can be broken down by tier like published numbers.

The three counts, all taken on the top-level query only:
  component1 — WHERE, GROUP BY, ORDER BY, LIMIT, each extra joined table,
               each OR, each LIKE
  component2 — nested queries: UNION / INTERSECT / EXCEPT, and subqueries
               used as a condition value
  others     — >1 aggregate, >1 SELECT column, >1 WHERE condition,
               >1 GROUP BY column
"""

from __future__ import annotations

from functools import lru_cache

import sqlglot
from sqlglot import exp

TIERS = ("easy", "medium", "hard", "extra")


def _flatten_conditions(node: exp.Expression | None) -> tuple[list[exp.Expression], int]:
    """Split an AND/OR condition tree into leaf conditions; also count ORs."""
    if node is None:
        return [], 0
    if isinstance(node, exp.Paren):
        return _flatten_conditions(node.this)
    if isinstance(node, (exp.And, exp.Or)):
        left, lo = _flatten_conditions(node.this)
        right, ro = _flatten_conditions(node.expression)
        return left + right, lo + ro + isinstance(node, exp.Or)
    return [node], 0


def _top_level(node: exp.Expression, kind: type) -> list[exp.Expression]:
    """Nodes of `kind` under `node`, not descending into nested subqueries."""
    found = []
    stack = list(node.iter_expressions())
    while stack:
        cur = stack.pop()
        if isinstance(cur, exp.Select):
            if kind is exp.Select:
                found.append(cur)
            continue  # never look inside a subquery
        if isinstance(cur, kind):
            found.append(cur)
        stack.extend(cur.iter_expressions())
    return found


def _condition(select: exp.Select, key: str) -> exp.Expression | None:
    clause = select.args.get(key)
    return clause.this if clause is not None else None


@lru_cache(maxsize=8192)
def spider_hardness(sql: str) -> str:
    """Return 'easy' | 'medium' | 'hard' | 'extra', or 'unknown' if unparsable."""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return "unknown"

    # Set operations: Spider stores `A UNION B UNION C` as A with one nested
    # union, so any chain counts as a single nested query. Score the left SELECT.
    comp2 = 1 if isinstance(tree, exp.SetOperation) else 0
    while isinstance(tree, exp.SetOperation):
        tree = tree.this
    while isinstance(tree, (exp.Subquery, exp.Paren)):
        tree = tree.this
    if not isinstance(tree, exp.Select):
        return "unknown"
    select = tree

    joins = select.args.get("joins") or []
    join_conds: list[exp.Expression] = []
    join_ors = 0
    for j in joins:
        leaves, ors = _flatten_conditions(j.args.get("on"))
        join_conds += leaves
        join_ors += ors
    where_conds, where_ors = _flatten_conditions(_condition(select, "where"))
    having_conds, having_ors = _flatten_conditions(_condition(select, "having"))
    all_conds = join_conds + where_conds + having_conds

    group = select.args.get("group")
    group_cols = group.expressions if group is not None else []
    order = select.args.get("order")
    order_cols = order.expressions if order is not None else []

    # component1
    comp1 = 0
    comp1 += bool(where_conds)
    comp1 += bool(group_cols)
    comp1 += bool(order_cols)
    comp1 += select.args.get("limit") is not None
    comp1 += len(joins)                                  # extra table units
    comp1 += join_ors + where_ors + having_ors
    comp1 += sum(isinstance(c, exp.Like) or (isinstance(c, exp.Not) and isinstance(c.this, exp.Like))
                 for c in all_conds)

    # component2: subqueries used directly as a condition value
    for cond in all_conds:
        comp2 += len(_top_level(cond, exp.Select))

    # others. Aggregate counting follows the official script exactly, quirks
    # included: for WHERE/HAVING it checks the cond unit's first field, which is
    # the NOT flag, so NOT conditions count as aggregates and real aggregates in
    # HAVING don't; HAVING's AND/OR tokens also count.
    def _aggs(e: exp.Expression) -> int:
        return len(_top_level(e, exp.AggFunc)) + isinstance(e, exp.AggFunc)

    def _is_not(c: exp.Expression) -> bool:
        return isinstance(c, exp.Not) or any(
            isinstance(c, t) and bool(c.args.get("not")) for t in (exp.In, exp.Between))

    agg_count = sum(bool(_aggs(e)) for e in select.expressions)
    agg_count += sum(_is_not(c) for c in where_conds)
    agg_count += sum(bool(_aggs(e)) for e in group_cols)
    agg_count += sum(_aggs(e) for e in order_cols)
    agg_count += sum(_is_not(c) for c in having_conds) + max(len(having_conds) - 1, 0)
    others = 0
    others += agg_count > 1
    others += len(select.expressions) > 1
    others += len(where_conds) > 1
    others += len(group_cols) > 1

    if comp1 <= 1 and others == 0 and comp2 == 0:
        return "easy"
    if (others <= 2 and comp1 <= 1 and comp2 == 0) or (comp1 <= 2 and others < 2 and comp2 == 0):
        return "medium"
    if ((others > 2 and comp1 <= 2 and comp2 == 0)
            or (2 < comp1 <= 3 and others <= 2 and comp2 == 0)
            or (comp1 <= 1 and others == 0 and comp2 <= 1)):
        return "hard"
    return "extra"
