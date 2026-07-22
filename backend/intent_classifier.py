import os
import json
import re
from groq import Groq

class IntentClassifier:

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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

        Return ONLY a valid JSON object matching this exact structure (use null for missing elements):
        {{
          "type": "sql" | "forecast" | "regression" | "trend" | "anomaly" | "summary" | "clustering" | "classification",
          "tables": ["<exact_table_name_1>"], 
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
            resp = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"} # Forces strict JSON output
            )

            raw = resp.choices[0].message.content.strip()
            result = json.loads(raw)
            
            # Ensure required fields exist with safe defaults
            result.setdefault("type", "sql")
            result.setdefault("tables", [])
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
            result = self._consolidate_fields(result, schema_info)

            return result
            
        except Exception as e:
            print(f"Intent Classification Error: {e}")
            return {
                "type": "sql",
                "tables": [],
                "chart_type": "auto",
                "date_col": None,
                "value_col": None,
                "group_col": None,
                "target_col": None,
                "feature_cols": None,
                "domain_context": None,
                "periods": 12
            }

    def _consolidate_fields(self, result: dict, schema_info: str) -> dict:
        """
        Applies two safety nets on top of whatever the LLM returned:

        1. Type-specific consolidation — if the model split clustering columns
           across value_col/group_col/etc. instead of putting them all in
           feature_cols, pull them back together. If it let a regression
           target leak into feature_cols, strip it back out.
        2. Schema validation — drop any column name the model returned that
           doesn't actually exist in schema_info, rather than letting it
           silently fail downstream (e.g. get filtered out of safe_cols in
           main.py with no explanation).
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