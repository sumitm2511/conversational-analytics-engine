import re

from llm_client import chat_completion_json, MODEL_STRUCTURED

# Explicit join-type mentions in the question ALWAYS override whatever the
# LLM inferred — this is the "explicit mention wins" half of the join
# decision. Checked as a defensive net after the LLM call, same pattern as
# the schema-validation net below, since an instruction inside the prompt
# alone isn't reliably followed 100% of the time.
_EXPLICIT_JOIN_PATTERNS = [
    (re.compile(r"\bleft\s+(outer\s+)?join\b", re.I), "left"),
    (re.compile(r"\bright\s+(outer\s+)?join\b", re.I), "right"),
    (re.compile(r"\bfull\s+(outer\s+)?join\b", re.I), "full"),
    (re.compile(r"\binner\s+join\b", re.I), "inner"),
]

_VALID_JOIN_TYPES = {"inner", "left", "right", "full"}

# Soft signal for outer-join intent that ISN'T a literal "left join" mention
# — phrasing like "including departments with no employees" clearly implies
# an outer join, but the LLM's own inference doesn't always catch it
# reliably. Used only to correct an "inner" result, never to override an
# explicit join-type mention (that's _EXPLICIT_JOIN_PATTERNS above, which
# always wins outright) and never to force needs_join true on its own.
_IMPLIED_OUTER_JOIN_PATTERNS = [
    re.compile(r"\bwith no\b", re.I),
    re.compile(r"\bwithout any\b", re.I),
    re.compile(r"\beven (if|when|without)\b", re.I),
    re.compile(r"\bwhether or not\b", re.I),
    re.compile(r"\bincluding\b.{0,40}\bno\b", re.I),
    re.compile(r"\bregardless of whether\b", re.I),
    re.compile(r"\bmay not have\b", re.I),
    re.compile(r"\bnever\b.{0,20}\b(placed|assigned|made|had)\b", re.I),
]


class IntentClassifier:

    def __init__(self):
        # No client to construct up front — llm_client manages the EricAI
        # client/token lifecycle centrally.
        pass

    def classify(self, question: str, schema_info: str) -> dict:
        prompt = f"""
        You are a strict analytics query router for a senior business analyst tool.
        You MUST NOT answer the user's question. Your job is to extract metadata and map it to the database schema.

        Analyze the schema and user question, then classify the query into exactly ONE type:
        1. "sql"            - aggregations, filters, multi-table joins, top-N.
        2. "forecast"       - predict/project FUTURE values for a time series.
        3. "trend"          - rolling averages, historical growth, MoM/YoY.
        4. "anomaly"        - detect outliers, unusual spikes, drops.
        5. "regression"     - predict a NUMERIC target from feature columns.
        6. "summary"        - summarize, describe, or overview the dataset.
        7. "clustering"     - DISCOVER unlabeled natural groupings in the data (no existing category
                              column to predict — the user wants the system to find segments on its own).
        8. "classification" - predict an EXISTING categorical/label column (e.g. Risk_Tier, Status,
                              Outcome, Approved/Rejected) for new records, using OTHER columns as
                              features. The label column already exists in the data with real values —
                              this is DIFFERENT from clustering, where no such label exists yet.

        Schema and Table Relationships:
        {schema_info}

        Question: {question}

        CRITICAL COLUMN MAPPING RULES:
        1. You MUST map the user's natural language to the EXACT column names in the schema provided above.
        2. Pay strict attention to Case Sensitivity and Spaces (e.g., if the user asks for "manufacturing price", you MUST output the exact schema match like "Manufacturing Price").
        3. If the user asks for a metric not explicitly named, use your best logical judgment to map it to the closest valid schema column (e.g., "revenue" -> "Sales").
        4. NEVER hallucinate a column name. It MUST exist in the schema.

        CRITICAL TYPE-SPECIFIC FIELD RULES:
        - If type == "clustering": populate ONLY "feature_cols" with ALL columns the user wants to
          segment by (numeric AND categorical). Set "date_col", "value_col", "group_col", and
          "target_col" to null. Do NOT split clustering columns across those other fields.
        - If type == "classification": "target_col" is the EXISTING label/category column being
          predicted (e.g. "risk_tier", "status"); "feature_cols" are the predictor columns. NEVER
          include the target column inside feature_cols. Only use "classification" when the label
          column already has real category values in the data — if the user wants to DISCOVER groups
          with no existing label, that is "clustering" instead.
        - If type == "regression": "target_col" is the NUMERIC column being predicted; "feature_cols"
          are the predictor columns. NEVER include the target column inside feature_cols.
        - If type == "forecast" or "trend": "date_col" and "value_col" are required. "group_col" is
          optional — only set it if the user explicitly wants per-segment results.
        - If type == "anomaly": "value_col" is required; "date_col" is optional.
        - "domain_context": only for "clustering" or "classification" queries, if the user's question
          includes a short description of the business domain or purpose of the data (e.g. "this is
          telecom resource management data"), extract it verbatim here. Otherwise null. Do NOT invent
          domain context that wasn't stated by the user.

        JOIN DECISION RULES (populate these regardless of type — matters most for "sql"/"summary",
        but any type that reads from "tables" benefits from not pulling in an unneeded table):
        - Set "needs_join" to true ONLY if answering the question genuinely requires columns that
          live in MORE THAN ONE table. If every column needed already exists in a single table, set
          "needs_join" to false and "tables" to just that one table — even if other tables are loaded
          in this session, and even if the question MENTIONS a concept that lives elsewhere. Mentioning
          something is not the same as needing its columns as output or as a filter: "show all
          departments and their budget, including departments with no employees" needs ONLY
          department + budget — if those already live together in one table, needs_join is FALSE, even
          though the question says "employees". The word "employees" here describes what NOT to filter
          out (don't drop budget-less-of-employees departments), it does not mean employee data is
          actually required in the output. Before setting needs_join true, check: are the SPECIFIC
          columns this question needs (as output or filter) actually missing from a single table? If
          not, don't join.
        - The "Possible Join Relationships" section of the schema above lists which columns are
          actually shared between which tables — only ever join on a column listed there. Never
          invent or assume a join column that isn't documented in the schema.
        - If "needs_join" is true, set "join_type" to one of "inner" | "left" | "right" | "full":
          - Default to "inner" (keep only rows that match in both tables) unless the question implies
            otherwise.
          - Use "left" when the question wants to keep every row from the first/primary table even
            when there's no match in the other table. Watch for phrasing like this even when it
            doesn't literally say "left join": "including departments with no employees", "customers
            who have never placed an order", "X regardless of whether they have a matching Y", "X
            even if there's no Y", "whether or not they have a Z". Any of these means do NOT use
            inner join — an inner join would silently drop exactly the rows being asked to keep.
          - Use "right" for the mirrored case (keep every row from the second table).
          - Use "full" when the question wants every row from BOTH tables regardless of match.
          - If the user's question explicitly names a join type ("left join", "outer join", "inner
            join", "full join"), use EXACTLY that type — an explicit instruction always overrides
            your own inference from the question's meaning.

        Return ONLY a valid JSON object matching this exact structure (use null for missing elements):
        {{
          "type": "sql" | "forecast" | "regression" | "trend" | "anomaly" | "summary" | "clustering" | "classification",
          "tables": ["<exact_table_name_1>"], 
          "needs_join": <true_or_false>,
          "join_type": "inner" | "left" | "right" | "full",
          "date_col": "<exact_schema_column_or_null>",
          "value_col": "<exact_schema_column_or_null>",
          "group_col": "<exact_schema_column_or_null>",
          "target_col": "<exact_schema_column_or_null>",
          "feature_cols": ["<exact_col1>", "<exact_col2>"] or null,
          "domain_context": "<short verbatim description or null>",
          "periods": <integer_default_12>,
          "chart_type": "bar" | "line" | "scatter" | "pie" | "heatmap" | "histogram" | "auto"
        }}
        """

        try:
            result = chat_completion_json(
                messages=[{"role": "user", "content": prompt}],
                model=MODEL_STRUCTURED,
                temperature=0,
            )

            # Ensure required fields exist with safe defaults
            result.setdefault("type", "sql")
            result.setdefault("tables", [])
            result.setdefault("needs_join", False)
            result.setdefault("join_type", "inner")
            result.setdefault("chart_type", "auto")
            result.setdefault("date_col", None)
            result.setdefault("value_col", None)
            result.setdefault("group_col", None)
            result.setdefault("target_col", None)
            result.setdefault("feature_cols", None)
            result.setdefault("domain_context", None)
            result.setdefault("periods", 12)

            # Defensive net: don't rely on the prompt alone — LLMs sometimes
            # split clustering/regression columns across value_col/group_col
            # instead of feature_cols regardless of instructions. Fix it here.
            result = self._consolidate_fields(result, schema_info, question)

            return result

        except Exception as e:
            print(f"Intent Classification Error: {e}")
            return {
                "type": "sql",
                "tables": [],
                "needs_join": False,
                "join_type": "inner",
                "chart_type": "auto",
                "date_col": None,
                "value_col": None,
                "group_col": None,
                "target_col": None,
                "feature_cols": None,
                "domain_context": None,
                "periods": 12
            }

    def _detect_explicit_join_type(self, question: str):
        """Returns the join type literally named in the question, or None."""
        for pattern, jtype in _EXPLICIT_JOIN_PATTERNS:
            if pattern.search(question):
                return jtype
        return None

    def _consolidate_fields(self, result: dict, schema_info: str, question: str) -> dict:
        """
        Applies safety nets on top of whatever the LLM returned:

        1. Type-specific consolidation — if the model split clustering columns
           across value_col/group_col/etc. instead of putting them all in
           feature_cols, pull them back together. If it let a regression
           target leak into feature_cols, strip it back out.
        2. Schema validation — drop any column name the model returned that
           doesn't actually exist in schema_info, rather than letting it
           silently fail downstream (e.g. get filtered out of safe_cols in
           main.py with no explanation).
        3. Explicit join-type override — if the question itself names a join
           type, that wins outright over whatever the model inferred.
        4. needs_join sanity check — never true with fewer than 2 tables.
        """

        # ── 1a. Clustering: consolidate stray fields into feature_cols ──────
        if result.get("type") == "clustering" and not result.get("feature_cols"):
            stray = [
                c for c in [
                    result.get("value_col"),
                    result.get("group_col"),
                    result.get("target_col"),
                    result.get("date_col"),
                ]
                if c
            ]
            if stray:
                result["feature_cols"] = stray
            result["value_col"]  = None
            result["group_col"]  = None
            result["target_col"] = None
            result["date_col"]   = None

        # ── 1b. Regression / Classification: never let the target leak into feature_cols ─
        if result.get("type") in ("regression", "classification") and result.get("target_col") and result.get("feature_cols"):
            result["feature_cols"] = [
                c for c in result["feature_cols"] if c != result["target_col"]
            ] or None

        # ── 2. Schema validation: strip any column not actually in the schema ─
        valid_cols = self._extract_valid_columns(schema_info)
        if valid_cols:
            for key in ["date_col", "value_col", "group_col", "target_col"]:
                if result.get(key) and result[key] not in valid_cols:
                    result[key] = None
            if result.get("feature_cols"):
                result["feature_cols"] = [
                    c for c in result["feature_cols"] if c in valid_cols
                ] or None

        # ── 3. Explicit join-type mention in the question always wins ───────
        explicit_join = self._detect_explicit_join_type(question)
        if explicit_join:
            result["join_type"] = explicit_join
            if len(result.get("tables") or []) >= 2:
                result["needs_join"] = True
        elif result.get("join_type") not in _VALID_JOIN_TYPES:
            result["join_type"] = "inner"

        # ── 3b. Soft nudge: strongly-implied outer-join phrasing, if the
        # model still came back with "inner" despite it. Never overrides an
        # explicit mention (already handled above) and never turns on
        # needs_join by itself — only corrects the TYPE when a join was
        # already decided to be needed. This is what catches phrasing like
        # "including departments with no employees" when the model's own
        # inference missed it and defaulted to its "inner" baseline.
        if (
            not explicit_join
            and result.get("needs_join")
            and result.get("join_type") == "inner"
            and any(p.search(question) for p in _IMPLIED_OUTER_JOIN_PATTERNS)
        ):
            result["join_type"] = "left"

        # ── 4. needs_join only makes sense with 2+ tables ────────────────────
        if len(result.get("tables") or []) < 2:
            result["needs_join"] = False

        return result

    def _extract_valid_columns(self, schema_info: str) -> set:
        """
        Parses column names out of the schema text produced by
        AnalyticsEngine.get_all_schemas() / get_enriched_schema(), whose
        column lines look like:  '  - colname (dtype)'
        """
        cols = set()
        for line in schema_info.splitlines():
            m = re.match(r"^\s*-\s+([^\(]+)\(", line)
            if m:
                cols.add(m.group(1).strip())
        return cols