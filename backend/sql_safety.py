"""
sql_safety.py — Guardrails for LLM-generated and user-supplied SQL.

Two validators:
  - validate_select_only(sql, allowed_tables): for AI-generated SQL running
    against the tool's OWN registered DuckDB tables. Parsed with the DuckDB
    dialect, since that's genuinely what executes it. Checks read-only AND
    that every referenced table is one we actually know about.
  - validate_read_only_sql(sql, dialect="mysql"): for user-typed SQL running
    against an EXTERNAL source (MySQL/Snowflake connect box or table
    browser's "Advanced" custom query). Parsed under whichever dialect
    actually executes it — NOT the DuckDB dialect — since that's what
    matters concretely: MySQL identifiers are backtick-quoted
    (`` `table_name` ``), which DuckDB's dialect does not recognize as valid
    identifier syntax, and Snowflake has its own quoting/date-function
    dialect distinct from both. Validating a query under the wrong dialect
    rejected valid queries with a parse error (surfaced to the user as
    "Invalid expression / Unexpected token" — the stray "[4m"/"[0m"
    fragments in that error are ANSI underline codes sqlglot embeds to
    highlight the bad token for terminal display; harmless, just a
    rendering artifact once shown as plain text). Checks read-only only —
    there's no fixed "allowed tables" list here, since the whole point is
    letting the user query any table in their own external database.

[FIX] CTE names (e.g. `WITH ranked_institutes AS (...) SELECT ...`) were
being flagged as "unknown table(s)" in validate_select_only — sqlglot's
exp.Table.find_all() picks up CTE alias references throughout the query
body, but a CTE alias is not a real table. Fixed by collecting CTE names
(via exp.CTE) and treating them as allowed alongside the real DuckDB tables.

[EXTENSION] validate_read_only_sql() originally hardcoded dialect="mysql",
since MySQL was the only external source. Adding Snowflake as a second
external source meant this could no longer be a fixed assumption — a
Snowflake query validated under the MySQL dialect can reject perfectly
valid Snowflake-specific syntax (or worse, accept something that behaves
differently). Now takes `dialect` explicitly; callers pass whichever
dialect actually matches where the query will run ("mysql", "snowflake",
etc. — anything sqlglot's dialect list supports).
"""

import sqlglot
from sqlglot import exp


class UnsafeSQLError(Exception):
    """Raised when generated or user-supplied SQL fails the read-only safety check."""
    pass


def _parse_single_readonly_statement(sql: str, dialect: str = "duckdb"):
    """
    Shared core: parses `sql` under the given SQL `dialect`, ensures it's
    exactly one statement, that it's a SELECT/UNION/Subquery, and that no
    forbidden write/DDL node appears anywhere in the parse tree. Returns the
    parsed statement. Raises UnsafeSQLError on any violation. Does NOT check
    table names — callers decide whether/how to restrict those.

    `dialect` MUST match whatever engine will actually execute the query —
    parsing under the wrong dialect can reject syntax that's perfectly valid
    (e.g. MySQL's backtick-quoted identifiers, or Snowflake's own quoting/
    date-function syntax, under the DuckDB dialect) or, worse, silently
    accept dialect-specific syntax that behaves differently than expected.
    """
    if not sql or not sql.strip():
        raise UnsafeSQLError("Empty SQL statement.")

    try:
        parsed = sqlglot.parse(sql, read=dialect)
    except Exception as e:
        raise UnsafeSQLError(f"Could not parse SQL: {e}")

    parsed = [p for p in parsed if p is not None]
    if len(parsed) != 1:
        raise UnsafeSQLError("Only a single SQL statement is allowed.")

    stmt = parsed[0]

    if not isinstance(stmt, (exp.Select, exp.Union, exp.Subquery)):
        raise UnsafeSQLError(
            f"Only SELECT statements are allowed, got {type(stmt).__name__}."
        )

    for bad_type in (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter):
        if stmt.find(bad_type) is not None:
            raise UnsafeSQLError(f"Statement contains forbidden operation: {bad_type.__name__}")

    return stmt


def validate_select_only(sql: str, allowed_tables: set) -> str:
    """
    For AI-generated SQL against the tool's OWN DuckDB tables. Raises
    UnsafeSQLError if `sql` is not a single, read-only SELECT statement
    referencing only tables in `allowed_tables` (plus any CTEs it defines
    itself). Returns `sql` unchanged if it passes.
    """
    stmt = _parse_single_readonly_statement(sql, dialect="duckdb")

    # CTE names (WITH x AS (...)) are not real tables — collect them so they
    # don't get flagged as "unknown" when referenced later in the query body.
    cte_names = {cte.alias_or_name for cte in stmt.find_all(exp.CTE)}

    referenced_tables = {t.name for t in stmt.find_all(exp.Table)}
    unknown = referenced_tables - set(allowed_tables) - cte_names
    if unknown:
        raise UnsafeSQLError(f"Query references unknown table(s): {unknown}")

    return sql


def validate_read_only_sql(sql: str, dialect: str = "mysql") -> str:
    """
    For user-typed SQL against an EXTERNAL source (the table-browser
    click-to-connect flow, or the "Advanced" raw-query box) — MySQL,
    Snowflake, or any future external connector. Parsed under `dialect`,
    which MUST match whichever engine actually executes the query — see
    module docstring for why this matters concretely (backtick-quoting,
    Snowflake-specific syntax). Enforces read-only (no INSERT/UPDATE/
    DELETE/DROP/CREATE/ALTER, single statement only) but does NOT restrict
    which tables can be referenced. Returns `sql` unchanged if it passes;
    raises UnsafeSQLError otherwise.

    Defaults to "mysql" to stay backward compatible with every existing
    caller written before Snowflake support existed — the Snowflake route
    passes dialect="snowflake" explicitly.
    """
    _parse_single_readonly_statement(sql, dialect=dialect)
    return sql