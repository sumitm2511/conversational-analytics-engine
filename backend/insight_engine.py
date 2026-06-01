"""
insight_engine.py  —  Upgraded
- Intent-specific prompts for forecast / trend / anomaly / regression / sql
- Passes statistical metadata (CAGR, trend_strength, anomaly narrative) to the LLM
- Returns richer structure with kpis, narrative, and confidence
"""

import os
import json
import re
import pandas as pd
from groq import Groq


class InsightEngine:

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(
        self,
        question: str,
        intent_type: str,
        data: pd.DataFrame = None,
        metrics: dict = None,
        extra_context: str = "",
        stat_meta: dict = None,       # ← NEW: pass forecast/trend/anomaly meta here
    ) -> dict:
        """
        Returns:
          {
            "summary":        "Executive paragraph",
            "insights":       ["bullet 1", ...],
            "anomalies":      ["flag 1", ...],
            "next_questions": ["follow-up Q1", ...],
            "kpis":           {"CAGR": "12.4%", ...}   ← NEW
          }
        """
        data_str    = self._df_summary(data) if data is not None else "No data."
        metrics_str = json.dumps(metrics or {}, default=str)
        meta_str    = json.dumps(stat_meta or {}, default=str)

        prompt = self._build_prompt(
            question, intent_type, data_str, metrics_str, meta_str, extra_context
        )

        return self._call_llm(prompt)

    def summarize_dataset(self, df: pd.DataFrame, schema: str) -> dict:
        """Called on file upload — full dataset overview."""
        num_summary = ""
        if not df.empty:
            try:
                num_summary = df.describe().round(2).to_string()
            except Exception:
                pass

        prompt = f"""
You are a senior data analyst. A new dataset has been loaded.

Schema:
{schema}

Statistical summary:
{num_summary}

Respond ONLY with a JSON object — no markdown, no explanation:
{{
  "summary": "<2-3 sentence description of the dataset and its likely business purpose>",
  "insights": [
    "<data quality observation or interesting pattern>",
    "<another observation>",
    "<another observation>"
  ],
  "anomalies": ["<data quality issue or empty array>"],
  "next_questions": [
    "<best first analysis question>",
    "<second question>",
    "<third question>"
  ],
  "kpis": {{}}
}}
"""
        return self._call_llm(prompt)

    # ── Prompt builder ─────────────────────────────────────────────────────────

    def _build_prompt(self, question, intent_type, data_str, metrics_str,
                      meta_str, extra_context) -> str:

        intent_instructions = {
            "forecast": """
Focus on:
- Future trajectory and direction of the forecast
- Confidence in the prediction (mention confidence interval if available)
- Seasonality or trend patterns detected
- Key risks or caveats for the forecast
- Recommended actions based on the projected values
""",
            "trend": """
Focus on:
- Overall trend direction and strength score
- MoM/YoY growth rates and acceleration
- Notable spikes, drops, or inflection points
- CAGR if available
- What is driving the trend (infer from data context)
""",
            "anomaly": """
Focus on:
- Number and severity of anomalies found
- The worst anomaly: when it occurred and how extreme it was
- Possible business explanations for the anomalies
- Whether anomalies are isolated events or part of a pattern
- Recommended investigation steps
""",
            "regression": """
Focus on:
- Model accuracy (R2, MAPE) and what it means in plain English
- The most important predictive features and their business significance
- Practical interpretation: what drives the target variable
- Limitations and where the model may be wrong
""",
            "sql": """
Focus on:
- Key numbers and what they mean for the business
- Comparisons, rankings, or distributions visible in the data
- Notable highs/lows or unexpected patterns
- Actionable recommendations
""",
            "summary": """
Focus on:
- High-level overview of what the data shows
- Most important metrics and what they indicate
- Trends or segments worth investigating further
""",
        }

        specific = intent_instructions.get(intent_type, intent_instructions["sql"])

        return f"""
You are a senior business analyst presenting findings to the executive team.

User asked: "{question}"
Query type: {intent_type}

Statistical metadata (CAGR, trend strength, anomaly counts, model info):
{meta_str}

Metrics:
{metrics_str}

Data sample:
{data_str}

Additional context: {extra_context}

Analysis focus for {intent_type} queries:
{specific}

Respond ONLY with a JSON object — no markdown fences, no preamble:
{{
  "summary": "<2-3 sentence executive-level summary with specific numbers>",
  "insights": [
    "<key finding with a specific number or %  from the data>",
    "<second finding>",
    "<third finding>"
  ],
  "anomalies": ["<unusual finding or empty array if none>"],
  "next_questions": [
    "<smart follow-up question 1>",
    "<smart follow-up question 2>",
    "<smart follow-up question 3>"
  ],
  "kpis": {{
    "<metric name>": "<value with unit>"
  }}
}}

Rules:
- Every insight must reference an actual number from the data or metadata
- Keep summary to 2-3 sentences maximum
- next_questions should be natural follow-up analytics queries the user can ask
- kpis should surface 2-4 headline numbers (e.g. CAGR, total, peak, anomaly count)
"""

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _df_summary(self, df: pd.DataFrame, max_rows: int = 15) -> str:
        if df is None or df.empty:
            return "No data."
        try:
            return df.head(max_rows).to_string(index=False, max_colwidth=40)
        except Exception:
            return "Data preview unavailable."

    def _call_llm(self, prompt: str) -> dict:
        try:
            resp = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=700,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            result = json.loads(raw)
            result.setdefault("summary", "Analysis complete.")
            result.setdefault("insights", [])
            result.setdefault("anomalies", [])
            result.setdefault("next_questions", [])
            result.setdefault("kpis", {})
            return result
        except Exception as e:
            return {
                "summary":        "Analysis complete.",
                "insights":       [],
                "anomalies":      [],
                "next_questions": [],
                "kpis":           {},
            }