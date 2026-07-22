"""
sql_safety.py — Guardrail for LLM-generated SQL.

Every query produced by SQLGenerator (initial or via fix_sql) must pass through
validate_select_only() before it's handed to AnalyticsEngine.execute_sql(). This
protects against the LLM (accidentally or via a prompt-injected question) producing
a DROP/UPDATE/DELETE/CREATE/ALTER statement, a multi-statement payload, or a query
that references a table outside the ones actually loaded in this session.
"""

import sqlglot
from sqlglot import exp


class UnsafeSQLError(Exception):
    """Raised when generated SQL fails the read-only / known-table safety check."""
    pass


def validate_select_only(sql: str, allowed_tables: set) -> str:
    """
    Raises UnsafeSQLError if `sql` is not a single, read-only SELECT statement
    referencing only tables in `allowed_tables`. Returns `sql` unchanged if it
    passes. Call this on every LLM-generated query before executing it.
    """
    if not sql or not sql.strip():
        raise UnsafeSQLError("Empty SQL statement.")

    try:
        parsed = sqlglot.parse(sql, read="duckdb")
    except Exception as e:
        raise UnsafeSQLError(f"Could not parse SQL: {e}")

    parsed = [p for p in parsed if p is not None]
    if len(parsed) != 1:
        raise UnsafeSQLError("Only a single SQL statement is allowed.")

    stmt = parsed[0]

    # Allow SELECT, set operations (UNION/INTERSECT/EXCEPT) and subqueries —
    # anything read-only. Everything else is rejected outright.
    if not isinstance(stmt, (exp.Select, exp.Union, exp.Subquery)):
        raise UnsafeSQLError(
            f"Only SELECT statements are allowed, got {type(stmt).__name__}."
        )

    # Defense in depth: even within a SELECT, reject if a forbidden node type
    # shows up anywhere in the parse tree (e.g. a nested CREATE/INSERT some
    # dialect quirk let slip through the top-level type check above).
    for bad_type in (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter):
        if stmt.find(bad_type) is not None:
            raise UnsafeSQLError(f"Statement contains forbidden operation: {bad_type.__name__}")

    referenced_tables = {t.name for t in stmt.find_all(exp.Table)}
    unknown = referenced_tables - set(allowed_tables)
    if unknown:
        raise UnsafeSQLError(f"Query references unknown table(s): {unknown}")

    return sql