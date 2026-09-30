"""
insight_engine.py  —  Upgraded
- Intent-specific prompts for forecast / trend / anomaly / regression / sql / classification
- Passes statistical metadata (CAGR, trend_strength, anomaly narrative) to the LLM
- Returns richer structure with kpis, narrative, and confidence

Migrated from Groq to EricAI via llm_client.chat_completion().
"""

import json
import re
import traceback
import pandas as pd

from llm_client import chat_completion, MODEL_INSIGHT


class InsightEngine:

    def __init__(self):
        # No client to construct up front — llm_client manages the EricAI
        # client/token lifecycle centrally.
        pass

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(
        self,
        question: str,
        intent_type: str,
        data: pd.DataFrame = None,
        metrics: dict = None,
        extra_context: str = "",
        stat_meta: dict = None,       # forecast/trend/anomaly meta
        total_rows: int = None,       # TRUE row count of the full result, before any sample truncation
    ) -> dict:
        """
        Returns:
          {
            "summary":        "Executive paragraph",
            "insights":       ["bullet 1", ...],
            "anomalies":      ["flag 1", ...],
            "next_questions": ["follow-up Q1", ...],
            "kpis":           {"CAGR": "12.4%", ...}
          }

        `total_rows` should be the query's REAL total row count (e.g.
        main.py's response["total_rows"], computed before any truncation) —
        NOT inferred from len(data), since `data` itself may already be
        capped (main.py passes df.head(200) for the sql/summary path).
        Without this, the model has no way to know its data sample is
        truncated and will report the SAMPLE SIZE as if it were the true
        count — confirmed: a 30-row result produced a stated count of
        exactly 15, the old _df_summary() truncation limit, not the real 30.

        If `data` is empty, this short-circuits and returns a plain "no
        records matched" result WITHOUT calling the LLM at all — an empty
        result is exactly the case where a model tends to fabricate a
        plausible-sounding narrative instead of reporting nothing was
        found, since it has no real data grounding it (confirmed: a
        genuinely empty result still produced a fully fabricated summary
        with specific fake numbers).
        """
        if data is None or data.empty:
            return {
                "summary":        "No records matched this query.",
                "insights":       [],
                "anomalies":      [],
                "next_questions": [],
                "kpis":           {},
            }

        sample_size = min(len(data), 15)
        true_total  = total_rows if total_rows is not None else len(data)

        data_str    = self._df_summary(data)
        metrics_str = json.dumps(metrics or {}, default=str)
        meta_str    = json.dumps(stat_meta or {}, default=str)

        prompt = self._build_prompt(
            question, intent_type, data_str, metrics_str, meta_str, extra_context,
            true_total, sample_size,
        )

        return self._call_llm(prompt, debug_context=f"generate(question={question!r}, intent_type={intent_type!r})")

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
        return self._call_llm(prompt, debug_context="summarize_dataset()")

    # ── Prompt builder ─────────────────────────────────────────────────────────

    def _build_prompt(self, question, intent_type, data_str, metrics_str,
                      meta_str, extra_context, true_total, sample_size) -> str:

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
            "classification": """
Focus on:
- Overall accuracy and weighted F1 score, and what they mean in plain English
- Which classes the model confuses most often (infer from the confusion matrix if described)
- The most important predictive features and their business significance
- Whether class imbalance is present and how it might affect reliability
- Practical guidance on how much to trust an individual prediction from this model
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

ROW COUNT — GROUND TRUTH (read this before anything else):
This query matched exactly {true_total} row(s) in total. The "Data sample" below shows only
the first {sample_size} of those {true_total} rows, for context only — it is NOT the complete
result. Whenever your summary, insights, or kpis state a count, total, or "number of X", you
MUST use {true_total} (or a figure computed from the Metrics/Statistical metadata below) —
NEVER count rows in the data sample yourself, since the sample is truncated at {sample_size}
and does not reflect the true total whenever {true_total} is larger than that.

Statistical metadata (CAGR, trend strength, anomaly counts, model info):
{meta_str}

Metrics:
{metrics_str}

Data sample ({sample_size} of {true_total} total rows):
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
- Any count, total, or "number of X" you state MUST equal {true_total} exactly — never a
  number you arrived at by counting rows in the truncated data sample above
- Do NOT reinterpret, relabel, or assign new meaning to a metric — describe it
  using the same concept its name already indicates. A metric literally named
  something like "Employee Id" is an identifier, never a count or total, no
  matter what its numeric value happens to look like.
- Keep summary to 2-3 sentences maximum
- next_questions should be natural follow-up analytics queries the user can ask
- kpis should surface 2-4 headline numbers (e.g. CAGR, total, peak, anomaly
  count) — never invent a KPI by reinterpreting a metric whose own name
  indicates it's an identifier/code rather than a real quantity
"""

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _df_summary(self, df: pd.DataFrame, max_rows: int = 15) -> str:
        if df is None or df.empty:
            return "No data."
        try:
            return df.head(max_rows).to_string(index=False, max_colwidth=40)
        except Exception:
            return "Data preview unavailable."

    def _extract_json(self, raw: str) -> dict:
        """
        Parses the LLM's response into a dict, tolerating the common ways a
        model wraps JSON in extra text even when explicitly told not to:
        - ```json fences (or bare ``` fences)
        - leading/trailing prose ("Here is the analysis:\n{...}\nLet me know...")

        Raises (doesn't swallow) if no valid JSON object can be recovered —
        the caller is responsible for logging/handling that, so failures
        stay visible instead of silently producing identical boilerplate
        output on every call.
        """
        cleaned = re.sub(r"```json|```", "", raw).strip()

        # Fast path: the whole cleaned string is valid JSON on its own.
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Fallback: pull out the first {...} object by brace-matching, in
        # case the model added any text before/after the JSON despite being
        # told not to — this is the part the previous version didn't do,
        # so any such text caused EVERY call to silently fail and fall
        # through to the generic "Analysis complete." placeholder.
        start = cleaned.find("{")
        if start == -1:
            raise ValueError("No JSON object found in LLM response")

        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start:i + 1]
                    return json.loads(candidate)

        raise ValueError("Unbalanced JSON object in LLM response")

    def _call_llm(self, prompt: str, debug_context: str = "") -> dict:
        try:
            raw = chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=MODEL_INSIGHT,
                temperature=0.2,
                max_tokens=700,
            )
            result = self._extract_json(raw)
            result.setdefault("summary", "Analysis complete.")
            result.setdefault("insights", [])
            result.setdefault("anomalies", [])
            result.setdefault("next_questions", [])
            result.setdefault("kpis", {})
            return result
        except Exception as e:
            # Previously this was a bare `except Exception: return <static dict>`
            # with NO logging — meaning every failure (bad JSON, EricAI error,
            # etc.) was invisible and produced the exact same generic Summary
            # every time, no matter the question or data. Logging here is what
            # actually lets you find the root cause in the server console the
            # next time this fires.
            print(f"[InsightEngine] LLM call failed ({debug_context}): {type(e).__name__}: {e}")
            traceback.print_exc()
            return {
                "summary":        "Analysis complete.",
                "insights":       [],
                "anomalies":      [],
                "next_questions": [],
                "kpis":           {},
            }