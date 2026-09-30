from llm_client import chat_completion, MODEL_SQL, MODEL_SQL_FALLBACK


_JOIN_TYPE_SQL = {
    "inner": "INNER JOIN",
    "left":  "LEFT JOIN",
    "right": "RIGHT JOIN",
    "full":  "FULL OUTER JOIN",
}


class SQLGenerator:

    def __init__(self):
        # No client to construct up front — llm_client manages the EricAI
        # client/token lifecycle centrally.
        pass

    def generate_sql(
        self,
        question: str,
        enriched_schema: str,
        target_tables: list = None,
        needs_join: bool = False,
        join_type: str = "inner",
        join_columns: list = None,
    ) -> str:
        """
        `needs_join` / `join_type` come from IntentClassifier's own join
        decision (see its "JOIN DECISION RULES") — a single upstream
        decision about WHETHER a join is needed and WHAT TYPE, rather than
        leaving that judgment call to this prompt too. Previously this
        method only ever hinted "make sure to write an appropriate JOIN"
        with zero guidance on type, which in practice meant the model
        defaulted to INNER JOIN almost every time — even for questions that
        clearly wanted to keep unmatched rows (e.g. "customers with no
        orders"), and even for questions that didn't need a join at all.

        `join_columns`: if given, this is an EXPLICIT user choice from the
        join picker — overrides the model's own judgment about which
        column(s) to join on entirely (see _build_table_hint).
        """
        table_hint = self._build_table_hint(target_tables, needs_join, join_type, join_columns)

        prompt = f"""
        You are an expert SQL analyst. Given the schema below, write a single 
        DuckDB-compatible SQL query that answers the user's question.

        Rules:
        - Return ONLY the raw SQL query — no explanation, no markdown, no fences
        - Use ONLY table and column names that exist in the schema
        - Use readable column aliases (e.g. AS total_sales)
        - For date/time, use DuckDB syntax: DATE_TRUNC, EXTRACT, strftime
        - For top-N use ORDER BY ... DESC LIMIT N
        - Prefer CTEs over nested subqueries for readability
        - If aggregating, always GROUP BY all non-aggregated columns
        - Never use SELECT * — always name the columns you need
        - Never apply SUM or AVG to a column the schema below marks as an
          identifier/code column — those aren't measurable quantities, and
          summing one produces a meaningless number. If the user is asking
          for a count of records, use COUNT(*) or COUNT(DISTINCT column)
          instead.
        - FAN-OUT WARNING: if a column's value is already unique per group in
          one table (e.g. one budget value per department) but you're joining
          to a table with multiple rows per group (e.g. one row per employee),
          do NOT directly SUM/AVG that column after the join — it will be
          counted once per matching row on the other side, inflating the
          total by however many rows matched. Instead, either (a) aggregate
          it in a subquery/CTE BEFORE joining, (b) select DISTINCT per group
          rather than summing, or (c) skip the join entirely if the
          group-level value alone answers the question — joining in the
          finer-grained table adds nothing but risk in that case.

          Worked example of this exact mistake:
          WRONG (fans out — repeats each department's budget once per employee,
          and produces one row per employee instead of one row per department):
            SELECT b.department, b.budget AS total_budget
            FROM department_budget b
            LEFT JOIN employees e ON b.department = e.department AND b.region = e.region
          RIGHT (aggregates employee data in a subquery FIRST, so budget is
          never duplicated, and the result is genuinely one row per department):
            SELECT b.department, b.budget AS total_budget, COALESCE(emp.employee_count, 0) AS employee_count
            FROM department_budget b
            LEFT JOIN (
              SELECT department, region, COUNT(*) AS employee_count
              FROM employees
              GROUP BY department, region
            ) emp ON b.department = emp.department AND b.region = emp.region
          Even simpler and correct here: if the question only needs
          department + budget (not any actual employee-level column), skip
          the join completely — querying department_budget alone already
          answers it, with no risk of duplication at all.
        {table_hint}

        Schema & Relationships:
        {enriched_schema}

        Question: {question}
        """

        sql = chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL_SQL,
            fallback_model=MODEL_SQL_FALLBACK,
            temperature=0,
            max_tokens=1024,
        ).strip()

        # Safely clean up markdown formatting without using literal backticks in the code
        backticks = chr(96) * 3
        sql = sql.replace(backticks + "sql", "").replace(backticks, "").strip()

        return sql

    def _build_table_hint(self, target_tables: list, needs_join: bool, join_type: str, join_columns: list = None) -> str:
        """
        Turns the upstream join decision into an EXPLICIT instruction rather
        than a vague hint — this is the actual fix for "always does inner
        join": the model is now told outright whether to join at all, and
        exactly which join type to use when it should.

        When join_columns is set (an explicit user choice from the join
        picker, not the model's own inference), the instruction states
        those exact column(s) and forbids substituting any other shared
        column — this is a deliberate user decision, not a suggestion.
        """
        if not target_tables:
            return ""

        if not needs_join or len(target_tables) < 2:
            # Explicit "don't join" instruction — stops the model from
            # joining anyway just because more than one table happens to be
            # loaded in the session, when the question only actually needs
            # one of them.
            table = target_tables[0]
            return (
                f"\nAnswer this query using ONLY the table '{table}'. "
                f"Do NOT join any other table — it is not needed for this question."
            )

        join_sql = _JOIN_TYPE_SQL.get(join_type, "INNER JOIN")

        if join_columns:
            cols_str = ", ".join(join_columns)
            return (
                f"\nThis query requires combining data from: {', '.join(target_tables)}.\n"
                f"The user has EXPLICITLY chosen this join via a join picker — this is not "
                f"your decision to make. Use a {join_sql} between them, joining ONLY on: "
                f"{cols_str}. Do not substitute, add, or drop any join column, even if other "
                f"shared columns exist between the tables — the user deliberately chose these."
            )

        return (
            f"\nThis query requires combining data from: {', '.join(target_tables)}.\n"
            f"Use a {join_sql} between them.\n"
            f"Only join on a column that is explicitly listed as a shared/possible join "
            f"relationship in the schema below — never invent a join column that isn't "
            f"documented there."
        )

    def fix_sql(self, original_sql: str, error: str, enriched_schema: str) -> str:
        """Ask the LLM to fix a broken SQL query."""
        prompt = f"""
        You are a DuckDB SQL expert. The following query produced an error.
        Fix it and return ONLY the corrected SQL — no explanation, no markdown.

        Schema & Relationships:
        {enriched_schema}

        Broken SQL:
        {original_sql}

        Error:
        {error}
        """

        sql = chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL_SQL,
            fallback_model=MODEL_SQL_FALLBACK,
            temperature=0,
            max_tokens=1024,
        ).strip()

        # Safely clean up markdown formatting without using literal backticks in the code
        backticks = chr(96) * 3
        sql = sql.replace(backticks + "sql", "").replace(backticks, "").strip()

        return sql