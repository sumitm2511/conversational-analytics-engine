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
        You MUST NOT answer the user's question. Your job is to extract metadata.

        Analyze the schema and user question, then classify the query into exactly ONE type:
        1. "sql"        - aggregations, filters, multi-table joins, top-N.
        2. "forecast"   - predict/project FUTURE values for a time series.
        3. "trend"      - rolling averages, historical growth, MoM/YoY.
        4. "anomaly"    - detect outliers, unusual spikes, drops.
        5. "regression" - predict a numeric target from feature columns.
        6. "summary"    - summarize, describe, or overview the dataset.

        Schema and Table Relationships:
        {schema_info}

        Question: {question}

        Return ONLY a valid JSON object matching this exact structure (use null for missing elements):
        {{
          "type": "sql" | "forecast" | "regression" | "trend" | "anomaly" | "summary",
          "tables": ["<table_name_1>", "<table_name_2>"], 
          "date_col": "<column_name_or_null>",
          "value_col": "<column_name_or_null>",
          "group_col": "<column_name_or_null>",
          "target_col": "<column_name_or_null>",
          "feature_cols": ["<col1>", "<col2>"] or null,
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
            result.setdefault("periods", 12)
            
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
                "periods": 12
            }