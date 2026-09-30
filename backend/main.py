import os
import json
import math
import uuid
import tempfile
import concurrent.futures
from pathlib import Path
from typing import Optional, Dict

from dotenv import load_dotenv
load_dotenv()   # reads .env into os.environ — must run before any module reads
                 # MYSQL_*/ERICAI_*/etc. env vars via os.getenv()

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from analytics_engine  import AnalyticsEngine, is_identifier_like
from intent_classifier import IntentClassifier
from sql_generator     import SQLGenerator
from sql_safety        import validate_select_only, validate_read_only_sql, UnsafeSQLError
from db_connectors     import MySQLConnector, SnowflakeConnector
from prediction_engine import (
    ForecastEngine, TrendEngine, RegressionEngine, AnomalyEngine,
    ClusterEngine, SupervisedClassifierEngine,
)
from chart_engine      import (
    build_auto_chart,
    build_forecast_chart,
    build_trend_chart,
    build_regression_chart,
    build_fi_chart,
    build_anomaly_chart,
    build_classification_chart,
)
from insight_engine import InsightEngine


# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(title="Analytics AI", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Shared singletons ──────────────────────────────────────────────────────────
# NOTE: These remain module-level because they hold NO per-request mutable state —
# each call to .run()/.classify()/.generate_sql() is self-contained. ClusterEngine
# is different: it stores a trained model (self.clf, self.scaler, self.cat_mappings)
# as instance state across calls, so it CANNOT be a shared global — see get_session()
# below, where each session gets its own ClusterEngine instance instead.

classifier     = IntentClassifier()
sql_gen        = SQLGenerator()
insight_eng    = InsightEngine()
forecast_eng   = ForecastEngine()
trend_eng      = TrendEngine()
regression_eng = RegressionEngine()
anomaly_eng    = AnomalyEngine()

# Thread pool + wall-clock timeout wrapper for the LLM-generated dataset
# overview (InsightEngine.summarize_dataset). Every data-loading action
# (upload, MySQL connect, combine, table removal) previously called this
# SYNCHRONOUSLY and waited for it before returning — meaning a slow or
# cold-starting EricAI call (llm_client.py's own retry logic backs off in
# MINUTES on a 5xx) could block the entire response, making a fast local
# file read/dtype-coercion look like a multi-minute "upload." The overview
# text is a nice-to-have description, not something the rest of the app
# depends on being present, so it's safe to give up on and fall back to a
# plain, honest summary rather than block the user's whole action on it.
_overview_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

def summarize_dataset_bounded(df: pd.DataFrame, schema: str, timeout: float = 8.0) -> dict:
    future = _overview_executor.submit(insight_eng.summarize_dataset, df, schema)
    try:
        return future.result(timeout=timeout)
    except Exception:
        rows = len(df) if df is not None else 0
        cols = len(df.columns) if df is not None else 0
        return {
            "summary": f"Dataset loaded — {rows:,} rows, {cols} column(s). "
                       f"(AI-generated overview took too long and was skipped.)",
            "insights": [],
            "anomalies": [],
            "next_questions": [],
            "kpis": {},
        }


# ── In-memory session store ────────────────────────────────────────────────────

SESSIONS: dict[str, dict] = {}

def get_session(session_id: str) -> dict:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "engine":         AnalyticsEngine(),
            "cluster_engine": ClusterEngine(),   # per-session — holds trained model state
            # Supervised classifiers, keyed by the real label column each was
            # trained against (a session can train more than one classifier
            # on different label columns). Same isolation requirement as
            # ClusterEngine — holds trained model state, so must be per-session.
            "classifiers":    {},   # {label_col: SupervisedClassifierEngine}
            "history":        [],
        }
    return SESSIONS[session_id]


# ── Request / response models ──────────────────────────────────────────────────

class JoinOverride(BaseModel):
    """
    An explicit join choice from the join-picker UI — when present on a
    QueryRequest, this OVERRIDES IntentClassifier's own needs_join/join_type/
    column decision entirely for that query, rather than merely influencing
    it. `columns` must be columns shared by both tables (validated against
    AnalyticsEngine.get_shared_columns() before use).
    """
    table_a:    str
    table_b:    str
    join_type:  str            # "inner" | "left" | "right" | "full"
    columns:    list[str]

class QueryRequest(BaseModel):
    question:                str
    chart_type:               Optional[str] = "auto"
    join_override:            Optional[JoinOverride] = None
    # When the user clicks "Skip — use AI's best guess" on the inline join
    # clarification card, this tells handle_query() not to pause and ask
    # again for THIS specific resubmission — otherwise the exact same
    # needs_join condition would just trigger the clarification a second
    # time in an infinite loop.
    skip_join_clarification: bool = False

class PredictPointRequest(BaseModel):
    point: Dict[str, float | int | str]

class PredictLabelRequest(BaseModel):
    label_col: str
    point: Dict[str, float | int | str]

class MySQLQueryRequest(BaseModel):
    query:      str
    table_name: Optional[str] = "mysql_data"

class SnowflakeQueryRequest(BaseModel):
    query:      str
    table_name: Optional[str] = "snowflake_data"

class CombineTablesRequest(BaseModel):
    table_a:        str
    table_b:        str
    columns:        list[str]
    new_table_name: Optional[str] = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _deep_sanitize(obj):
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass

    if isinstance(obj, dict):
        return {k: _deep_sanitize(v) for k, v in obj.items()}

    if isinstance(obj, list) or isinstance(obj, tuple):
        return [_deep_sanitize(v) for v in obj]

    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None

    if isinstance(obj, pd.Timestamp):
        return str(obj)

    return obj

def df_to_rows(df: pd.DataFrame, max_rows: int = 500) -> dict:
    if df is None or df.empty:
        return {"columns": [], "rows": [], "total_rows": 0}

    total   = len(df)
    display = df.head(max_rows).copy()

    for col in display.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns:
        display[col] = display[col].astype(str)

    display = display.replace([float("inf"), float("-inf")], None)
    rows    = [[_deep_sanitize(cell) for cell in row] for row in display.values.tolist()]

    return {
        "columns":    display.columns.tolist(),
        "rows":       rows,
        "total_rows": total,
    }

def _explanation_sentence(explanation: list, explanation_type: str) -> str:
    """Turns a SHAP explanation (or global-importance fallback) into one plain-English
    sentence for the narrative. Phrasing differs for per-prediction vs. global fallback,
    since the latter isn't specific to this exact record and shouldn't be presented as if it were."""
    if not explanation:
        return ""
    top = explanation[0]
    if explanation_type == "per_prediction":
        direction = top.get("direction", "influenced")
        rest = explanation[1:3]
        rest_str = "".join(
            f", with {r['feature']} also having {'increased' if (r.get('contribution') or 0) > 0 else 'decreased'} it"
            for r in rest[:1]
        )
        return (
            f" This was driven primarily by **{top['feature']}**, which {direction} the prediction"
            f"{rest_str}."
        )
    elif explanation_type == "global_fallback":
        return (
            f" Based on this model's overall feature importance (not specific to this exact record — "
            f"a fast per-prediction explanation wasn't available for this model type), "
            f"**{top['feature']}** is typically the strongest driver."
        )
    return ""


def extract_metrics(df: pd.DataFrame) -> dict:
    """
    Picks up to 4 headline numeric metrics from a query result. Uses the
    SAME is_identifier_like() check AnalyticsEngine uses to annotate the
    schema for the LLMs — one shared definition of "safe to aggregate",
    not a second copy that could quietly disagree with the first (that
    disagreement is exactly what previously let SUM(employee_id) become a
    KPI and then get mislabeled "Total Employees" by the insight narrative).
    """
    if df is None or df.empty or len(df) > 20:
        return {}

    metrics = {}
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        if is_identifier_like(col, df[col]):
            continue

        if len(metrics) >= 4:
            break

        try:
            val = df[col].sum() if len(df) > 1 else df[col].iloc[0]
            if math.isnan(val) or math.isinf(val):
                continue
            metrics[col.replace("_", " ").title()] = (
                f"{val:,.0f}" if abs(val) > 100 else f"{val:,.2f}"
            )
        except Exception:
            continue
    return metrics


# ── Master Dispatcher ──────────────────────────────────────────────────────────

class AnalyticsDispatcher:
    def __init__(self, engine: AnalyticsEngine, cluster_engine: ClusterEngine, classifiers: dict):
        self.engine = engine
        self.cluster_engine = cluster_engine
        self.classifiers = classifiers   # {label_col: SupervisedClassifierEngine}, session-scoped, mutated in place

    def _get_clean_dataframe(self, intent: dict, target_tables: list) -> pd.DataFrame:
        """
        Builds the base dataframe an engine (forecast/trend/anomaly/
        regression/clustering/classification) will run on. When the
        intent's own join decision (IntentClassifier's "JOIN DECISION
        RULES" — needs_join/join_type, computed for every query type, not
        just sql/summary) says the columns needed span two tables, this
        now builds a live join via AnalyticsEngine.get_joined_dataframe()
        instead of silently only ever looking at target_tables[0]. That
        was the actual gap before this: these engines had NO join
        capability at all, so a question like "forecast total budget by
        department" (budget in one table, nothing to forecast on in the
        other) could only ever work if the two tables had already been
        manually combined into one persistent table first.
        """
        needs_join = intent.get("needs_join", False)

        if needs_join and len(target_tables) >= 2:
            join_result = self.engine.get_joined_dataframe(
                target_tables[0], target_tables[1],
                join_type=intent.get("join_type", "inner"),
                on_columns=intent.get("join_columns") or None,
            )
            if join_result["error"]:
                # Degrade gracefully to the primary table alone rather than
                # failing the whole query outright over a join that didn't
                # pan out (e.g. no shared columns after all).
                base_df        = self.engine.get_dataframe(target_tables[0])
                available_cols = self.engine.tables.get(target_tables[0], {}).get("columns", [])
            else:
                base_df        = join_result["data"]
                available_cols = list(base_df.columns)
        else:
            base_df        = self.engine.get_dataframe(target_tables[0])
            available_cols = self.engine.tables.get(target_tables[0], {}).get("columns", [])

        cols = [
            intent.get("date_col"),
            intent.get("value_col"),
            intent.get("group_col"),
            intent.get("target_col"),
        ]
        cols      = [c for c in cols if c] + (intent.get("feature_cols") or [])
        safe_cols = [c for c in set(cols) if c in available_cols] if cols else []

        if not safe_cols:
            return base_df

        return base_df[safe_cols].copy()

    def _run_engine_safely(self, engine_call, engine_label: str, *args, **kwargs):
        """
        Runs a prediction/analysis engine call and converts ANY unhandled
        exception into a clear, actionable message instead of letting a raw
        low-level error (e.g. "list index out of range" from deep inside a
        forecasting library given too little/irregular data) leak straight
        through to the user. Engines that already return their own
        {"error": ...} dict are unaffected — this only catches what they
        DON'T already handle themselves, and is applied uniformly to every
        ML engine call rather than patched in one at a time per engine.

        Returns (result, error_message) — exactly one of the two is None.
        """
        try:
            return engine_call(*args, **kwargs), None
        except Exception as e:
            return None, (
                f"{engine_label} failed on this data ({type(e).__name__}: {e}). "
                f"This usually means the selected columns don't have enough "
                f"consistent structure for this kind of analysis — try a "
                f"different column combination, more historical data points, "
                f"or a simpler query (e.g. a trend or summary instead)."
            )

    def handle_query(
        self, question: str, chart_type_override: str,
        join_override: dict = None, skip_join_clarification: bool = False,
    ) -> dict:
        enriched_schema = getattr(self.engine, "get_enriched_schema", self.engine.get_all_schemas)()

        q_lower = question.lower()

        # ── Disambiguating unsupervised vs. supervised point queries ──────────
        # Both ClusterEngine.predict_point and SupervisedClassifierEngine.predict_point
        # answer "what would this record be classified as", so generic keywords like
        # "belong"/"match"/"classify" alone can't tell them apart once a session has
        # trained BOTH kinds of model. Instead: for supervised models, we check whether
        # the question references the ACTUAL trained label column's name (e.g. a
        # classifier trained on "risk_tier" gets matched by "which risk tier..."). This
        # scales cleanly to multiple trained classifiers on different label columns,
        # since each is disambiguated by its own label name rather than shared keywords.
        import re as _re
        # "classif" (not "classify") catches classify/classified/classification;
        # "would" (not the fragile "what would" bigram) catches "what X would...".
        action_words = ["belong", "match", "classif", "predict", "would", "which"]
        has_action_word   = any(w in q_lower for w in action_words)
        has_numeric_value = bool(_re.search(r'\d', question))

        matched_label_col = None
        # Require a concrete numeric value in the question, not just the label
        # name + an action word — otherwise a plain aggregation question like
        # "which region has the highest utilization" or a training request like
        # "predict billable hours from FTE and region" (region mentioned only
        # as a FEATURE, not as the thing being predicted for one record) would
        # incorrectly route to point-classification instead of SQL/training.
        if has_action_word and has_numeric_value:
            for label_col in self.classifiers.keys():
                # Match on individual underscore-separated words (each checked as a
                # substring), not the whole phrase — a column named "emp_category"
                # must still match natural phrasing like "employee category", where
                # "emp" is a substring of "employee" but "emp category" as a literal
                # phrase never appears. Whole-phrase matching silently failed on any
                # abbreviated column name, which meant the routing check could NEVER
                # succeed for that column — every question fell through to the LLM
                # classifier and got treated as a fresh training request instead of
                # reusing the already-trained model.
                label_parts = [p for p in label_col.lower().split("_") if p]
                if label_parts and all(part in q_lower for part in label_parts):
                    matched_label_col = label_col
                    break

        is_prediction_intent = ("cluster" in q_lower or "persona" in q_lower) and any(w in q_lower for w in ["belong", "match", "predict", "classify", "if a"])

        if matched_label_col:
            intent_type = "predict_label"
            target_tables = [self.engine.active_table]
        elif is_prediction_intent and self.cluster_engine.clf is not None:
            intent_type = "predict_point"
            target_tables = [self.engine.active_table]
        else:
            intent          = classifier.classify(question, enriched_schema)
            intent_type     = intent.get("type", "sql")
            chart_type      = chart_type_override if chart_type_override != "auto" else intent.get("chart_type", "bar")
            target_tables   = intent.get("tables", [])

            # ── Explicit join override from the join-picker UI ─────────────
            # Takes priority over whatever IntentClassifier decided — this
            # is a deliberate user choice, not a suggestion to weigh. Skips
            # the LLM's own join reasoning entirely for this query rather
            # than merely nudging it, since the user has already made this
            # decision explicitly through the picker.
            if join_override:
                intent["needs_join"]   = True
                intent["join_type"]    = join_override.get("join_type", "inner")
                intent["tables"]       = [join_override["table_a"], join_override["table_b"]]
                intent["join_columns"] = join_override.get("columns") or []
                target_tables = intent["tables"]

        if not target_tables or not isinstance(target_tables, list):
            target_tables = [self.engine.active_table]

        # ── Join clarification pause ─────────────────────────────────────────
        # If this question needs data from two tables, and the user hasn't
        # already told us how (join_override, from a resolved clarification
        # card) or explicitly said to skip asking (skip_join_clarification,
        # from clicking "use AI's best guess"), pause HERE and hand the
        # decision back to the user instead of silently committing to it —
        # the same spirit as asking a clarifying question before answering.
        # intent_type is checked first so this never fires for
        # predict_point/predict_label, where `intent` was never built at all
        # (short-circuit evaluation means intent.get(...) below is never
        # reached for those two, avoiding a NameError).
        if (
            intent_type not in ("predict_point", "predict_label")
            and intent.get("needs_join")
            and len(target_tables) >= 2
            and not join_override
            and not skip_join_clarification
        ):
            table_a, table_b = target_tables[0], target_tables[1]
            shared = self.engine.get_shared_columns(table_a, table_b)
            return {
                "question":            question,
                "type":                "join_clarification",
                "table_a":             table_a,
                "table_b":             table_b,
                "suggested_join_type": intent.get("join_type", "inner"),
                "join_types":          ["inner", "left", "right", "full"],
                "shared_columns":      shared.get("columns", []),
                "error":               shared.get("error"),
            }

        response = {
            "question":           question,
            "type":               intent_type,
            "chart":              None,
            "sql":                None,
            "columns":            [],
            "rows":               [],
            "total_rows":         0,
            "metrics":            {},
            "feature_importance": None,
            "insights":           None,
            "error":              None,
        }

        try:
            # ── SUPERVISED POINT CLASSIFICATION (IN-CHAT) ─────────────────────
            if intent_type == "predict_label":
                clf_engine = self.classifiers[matched_label_col]
                result = clf_engine.predict_point_from_text(question)

                if "error" in result:
                    response["error"] = result["error"]
                else:
                    label = result["predicted_label"]
                    conf  = result["match_confidence"] * 100

                    response["metrics"] = {
                        f"Predicted {matched_label_col.replace('_', ' ').title()}": str(label),
                        "Match Confidence": f"{conf:.1f}%",
                    }

                    matrix = " | ".join([f"{k} ({v*100:.1f}%)" for k, v in result["confidence_matrix"].items()])
                    narrative = (
                        f"Based on the trained classifier for **{matched_label_col}**, this record is "
                        f"predicted as **{label}** with {conf:.1f}% confidence. Probability Matrix: [ {matrix} ]"
                    )
                    narrative += _explanation_sentence(result.get("explanation"), result.get("explanation_type"))
                    if result.get("defaulted_fields"):
                        narrative += f" ⚠️ {result['warning']}"
                    response["stat_narrative"] = narrative
                    response["explanation"]      = result.get("explanation")
                    response["explanation_type"] = result.get("explanation_type")

                    extracted = result["extracted_features"]
                    response["columns"] = list(extracted.keys())
                    response["rows"] = [[str(v) for v in extracted.values()]]
                    response["total_rows"] = 1

                    response["insights"] = [
                        f"The Natural Language engine successfully extracted {len(extracted)} parameters from your query.",
                        f"The trained classifier matched these parameters against learned patterns for '{matched_label_col}'."
                    ]
                    if result.get("defaulted_fields"):
                        response["insights"].append(
                            f"⚠️ Defaulted (not specified in your question): {', '.join(result['defaulted_fields'])}. "
                            f"Prediction confidence should be treated as lower than usual."
                        )

                    response["type"] = "summary"

            # ── POINT PREDICTION (IN-CHAT) ────────────────────────────────────
            elif intent_type == "predict_point":
                result = self.cluster_engine.predict_point_from_text(question)

                if "error" in result:
                    response["error"] = result["error"]
                else:
                    persona = result["assigned_persona"]
                    conf = result["match_confidence"] * 100

                    # 1. Feed the KPI Cards (Safe for React rendering)
                    response["metrics"] = {
                        "Assigned Persona": str(persona).title(),
                        "Match Confidence": f"{conf:.1f}%"
                    }

                    # 2. Feed the Summary Narrative (Renders in the italic ~ strip)
                    matrix = " | ".join([f"{k} ({v*100:.1f}%)" for k, v in result["confidence_matrix"].items()])
                    narrative = (
                        f"Based on the trained Random Forest model, this customer profile belongs to the **{persona}** persona "
                        f"with {conf:.1f}% confidence. Probability Matrix: [ {matrix} ]"
                    )
                    narrative += _explanation_sentence(result.get("explanation"), result.get("explanation_type"))
                    if result.get("defaulted_fields"):
                        narrative += f" ⚠️ {result['warning']}"
                    response["stat_narrative"] = narrative
                    response["explanation"]      = result.get("explanation")
                    response["explanation_type"] = result.get("explanation_type")

                    # 3. Feed the Data Table (Proves to the user what the LLM extracted)
                    extracted = result["extracted_features"]
                    response["columns"] = list(extracted.keys())
                    response["rows"] = [[str(v) for v in extracted.values()]]
                    response["total_rows"] = 1

                    # 4. Safely mock the insights array to prevent .map() crashes in the frontend
                    response["insights"] = [
                        f"The Natural Language engine successfully extracted {len(extracted)} parameters from your query.",
                        f"The classifier matched these parameters to the geometric boundaries of the '{persona}' cluster."
                    ]
                    if result.get("defaulted_fields"):
                        response["insights"].append(
                            f"⚠️ Defaulted (not specified in your question): {', '.join(result['defaulted_fields'])}. "
                            f"Prediction confidence should be treated as lower than usual."
                        )

                    response["type"] = "summary"

            elif intent_type in ("sql", "summary"):
                # needs_join / join_type come from IntentClassifier's own
                # join decision (see its "JOIN DECISION RULES") — grounded
                # in the schema's actual shared-column data and overridden
                # by any join type the user named explicitly in the
                # question, or by an explicit join_override from the
                # join-picker UI (spliced into `intent` above, including
                # join_columns) which takes priority over both.
                sql = sql_gen.generate_sql(
                    question,
                    enriched_schema,
                    target_tables,
                    needs_join=intent.get("needs_join", False),
                    join_type=intent.get("join_type", "inner"),
                    join_columns=intent.get("join_columns") or None,
                )

                try:
                    validate_select_only(sql, set(self.engine.tables.keys()))
                except UnsafeSQLError as e:
                    response["error"] = f"Generated query was rejected for safety: {e}"
                    return response

                response["sql"] = sql
                result = self.engine.execute_sql(sql)

                if result["error"]:
                    fixed_sql = sql_gen.fix_sql(sql, result["error"], enriched_schema)
                    try:
                        validate_select_only(fixed_sql, set(self.engine.tables.keys()))
                    except UnsafeSQLError as e:
                        response["error"] = f"Corrected query was rejected for safety: {e}"
                        return response
                    response["sql"] = fixed_sql
                    result = self.engine.execute_sql(fixed_sql)

                if result["error"]:
                    response["error"] = result["error"]
                    return response

                df = result["data"]
                response.update(df_to_rows(df))
                response["metrics"] = extract_metrics(df)
                response["chart"]   = build_auto_chart(df, chart_type, question)
                # total_rows passed explicitly — response["total_rows"] is the
                # TRUE, untruncated count from df_to_rows(); df.head(200) below
                # only caps what InsightEngine samples for context, and it must
                # know the real total rather than inferring it from the sample
                # size (see insight_engine.py's generate() docstring for the
                # exact bug this fixes: an under-15/30/etc-row sample getting
                # reported as if it were the complete result).
                response["insights"] = insight_eng.generate(
                    question, intent_type, df.head(200), response["metrics"],
                    total_rows=response["total_rows"],
                )

            elif intent_type == "forecast":
                date_col, value_col = intent.get("date_col"), intent.get("value_col")
                if not date_col or not value_col:
                    return {**response, "error": "Could not identify date and value columns for forecast."}

                df_clean = self._get_clean_dataframe(intent, target_tables)

                # Pre-flight check: forecasting needs a genuine time series —
                # a column with only 1-2 distinct dates (e.g. a per-record
                # date that doesn't represent repeated observations over
                # time, like joining in per-employee hire dates to forecast
                # a per-department budget) isn't forecastable at all, and
                # was the root cause of a raw "list index out of range"
                # crash surfacing straight to the user instead of a clear
                # explanation.
                distinct_periods = df_clean[date_col].dropna().nunique() if date_col in df_clean.columns else 0
                if distinct_periods < 3:
                    return {
                        **response,
                        "error": (
                            f"Not enough historical time points to forecast — found only "
                            f"{distinct_periods} distinct value(s) in '{date_col}'. Forecasting "
                            f"needs a genuine time series (at least 3-4 periods) where "
                            f"'{value_col}' actually varies over time."
                        ),
                    }

                periods = intent.get("periods", 12)
                result, engine_err = self._run_engine_safely(
                    forecast_eng.run, "Forecast", df_clean, date_col, value_col, periods, intent.get("group_col")
                )
                if engine_err:
                    return {**response, "error": engine_err}

                if "error" in result:
                    return {**response, "error": result["error"]}

                data      = result["data"]
                stat_meta = result.get("meta", {})

                if isinstance(data, pd.DataFrame):
                    response.update(df_to_rows(data))
                else:
                    first = list(data.values())[0]
                    response.update(df_to_rows(first) if isinstance(first, pd.DataFrame) else {})

                response["forecast_leaderboard"] = stat_meta.get("leaderboard", [])
                response["primary_metric"]       = stat_meta.get("primary_metric", "SMAPE")
                response["series_profile"]       = stat_meta.get("series_profile", {})
                response["stat_narrative"]       = stat_meta.get("narrative", "")

                response["chart"]   = build_forecast_chart(result, date_col, value_col)

                cov     = stat_meta.get("empirical_coverage")
                z_mult  = stat_meta.get("z_multiplier", 1.645)
                ci_pct  = "95%" if z_mult >= 2.0 else "90%"
                cov_str = f"{cov:.1f}%" if cov is not None else "—"

                response["metrics"] = {
                    "Periods Forecasted":  periods,
                    "Model":               stat_meta.get("model_display", stat_meta.get("model", "—")).title(),
                    "Granularity":         stat_meta.get("granularity", "monthly").title(),
                    "Direction":           stat_meta.get("direction", "—").title(),
                    "CAGR":                f"{stat_meta['cagr']*100:.1f}%" if stat_meta.get("cagr") else "—",
                    "CV Coverage":         cov_str,
                    "CI Width":            ci_pct,
                }
                response["insights"] = insight_eng.generate(
                    question, "forecast",
                    data if isinstance(data, pd.DataFrame) else None,
                    response["metrics"],
                    extra_context=stat_meta.get("narrative", ""),
                    stat_meta=stat_meta,
                )

            elif intent_type == "trend":
                date_col, value_col = intent.get("date_col"), intent.get("value_col")
                if not date_col or not value_col:
                    return {**response, "error": "Could not identify date and value columns for trend analysis."}

                df_clean = self._get_clean_dataframe(intent, target_tables)
                result, engine_err = self._run_engine_safely(
                    trend_eng.run, "Trend analysis", df_clean, date_col, value_col, intent.get("group_col")
                )
                if engine_err:
                    return {**response, "error": engine_err}

                if "error" in result:
                    return {**response, "error": result["error"]}

                df_trend  = result["data"]
                stat_meta = result.get("meta", {})
                response["stat_narrative"] = stat_meta.get("narrative", "")

                response.update(df_to_rows(df_trend))
                response["chart"]   = build_trend_chart(df_trend, date_col, value_col, intent.get("group_col"))
                response["metrics"] = {
                    "Rolling Window":  f"{result.get('window', 3)} periods",
                    "Direction":       stat_meta.get("direction", "—").title(),
                    "Trend Strength":  f"{stat_meta['trend_strength']:.2f}" if stat_meta.get("trend_strength") else "—",
                    "CAGR":            f"{stat_meta['cagr']*100:.1f}%" if stat_meta.get("cagr") else "—",
                    "Avg MoM Growth":  f"{stat_meta['avg_mom_pct']:+.1f}%" if stat_meta.get("avg_mom_pct") else "—",
                    "Spikes Detected": str(stat_meta.get("spike_count", 0)),
                }
                response["insights"] = insight_eng.generate(
                    question, "trend", df_trend.head(100), response["metrics"],
                    extra_context=stat_meta.get("narrative", ""), stat_meta=stat_meta,
                )

            elif intent_type == "anomaly":
                df_clean  = self._get_clean_dataframe(intent, target_tables)
                value_col = intent.get("value_col")

                if not value_col:
                    numeric_cols = df_clean.select_dtypes(include="number").columns.tolist()
                    value_col    = numeric_cols[0] if numeric_cols else None

                if not value_col:
                    return {**response, "error": "Could not identify a numeric column for anomaly detection."}

                date_col = intent.get("date_col")
                result, engine_err = self._run_engine_safely(
                    anomaly_eng.run, "Anomaly detection", df_clean, value_col, date_col,
                    intent.get("group_col"), contamination="auto",
                )
                if engine_err:
                    return {**response, "error": engine_err}

                if "error" in result:
                    return {**response, "error": result["error"]}

                stat_meta = result.get("meta", {})
                response["stat_narrative"] = stat_meta.get("narrative", "")
                response.update(df_to_rows(result["anomalies"]))
                response["metrics"] = {
                    "Anomalies Found":  result.get("anomaly_count", 0),
                    "High / Critical":  stat_meta.get("high_severity", 0),
                    "Contextual":       stat_meta.get("contextual", 0),
                    "Mean":             result.get("mean", 0),
                    "Std Dev":          result.get("std", 0),
                    "Method":           result.get("method", "rolling_z").replace("_", " ").title(),
                    "Contamination %":  f"{result.get('contamination', 0)*100:.1f}%",
                }
                response["chart"]    = build_anomaly_chart(result, date_col, value_col)
                response["insights"] = insight_eng.generate(
                    question, "anomaly",
                    result.get("anomalies", pd.DataFrame()).head(50),
                    response["metrics"],
                    extra_context=stat_meta.get("narrative", ""),
                    stat_meta=stat_meta,
                )

            elif intent_type == "regression":
                target_col, feature_cols = intent.get("target_col"), intent.get("feature_cols")
                if not target_col or not feature_cols:
                    return {**response, "error": "Could not identify target and feature columns for regression."}

                # Defensive guard, in addition to the classifier-side consolidation:
                # never let the prediction target sit inside its own feature list.
                feature_cols = [c for c in feature_cols if c != target_col]
                if not feature_cols:
                    return {**response, "error": "No usable feature columns remain after excluding the target column."}

                df_clean = self._get_clean_dataframe(intent, target_tables)
                result, engine_err = self._run_engine_safely(
                    regression_eng.run, "Regression", df_clean, target_col, feature_cols
                )
                if engine_err:
                    return {**response, "error": engine_err}

                if "error" in result:
                    return {**response, "error": result["error"]}

                response.update(df_to_rows(result["data"]))
                response["metrics"] = {k: str(v) for k, v in result["metrics"].items()}
                response["chart"]   = build_regression_chart(result)

                fi_df = result.get("feature_importance")
                if fi_df is not None:
                    response["feature_importance"] = {
                        "columns": fi_df.columns.tolist(),
                        "rows":    fi_df.values.tolist(),
                    }
                    fi_chart = build_fi_chart(fi_df)
                    if fi_chart:
                        response["chart"] = fi_chart

                response["insights"] = insight_eng.generate(
                    question, "regression", result["data"], result["metrics"]
                )

            elif intent_type == "clustering":
                df_clean     = self._get_clean_dataframe(intent, target_tables)
                feature_cols = intent.get("feature_cols")

                if not feature_cols:
                    feature_cols = df_clean.columns.tolist()

                # Explicit context in the question wins if present; otherwise
                # fall back to whatever was set once via the UI at upload time.
                # Domain context, if the user included it directly in the question
                # (e.g. "Segment by X — this is telecom resourcing data").
                domain_context = intent.get("domain_context")
                result, engine_err = self._run_engine_safely(
                    self.cluster_engine.run, "Clustering", df_clean, feature_cols, domain_context=domain_context
                )
                if engine_err:
                    return {**response, "error": engine_err}

                if "error" in result:
                    return {**response, "error": result["error"]}

                df_cluster = result["data"]
                meta       = result["meta"]
                fi_df      = result.get("feature_importance")

                response.update(df_to_rows(df_cluster))
                response["metrics"] = {
                    "Optimal Clusters":  str(meta["k"]),
                    "Silhouette Score":  f"{meta['silhouette']:.2f}",
                    "Algorithm":         meta["algorithm"],
                    "#1 Split Factor":   meta["top_driver"].replace("_", " ").title(),
                    "Outliers Isolated": str(meta.get("outlier_count", 0)),
                    "PCA Var (PC1+PC2)": (
                        f"{sum(meta['pca_variance'][:2])*100:.1f}%"
                        if meta.get("pca_variance") else "—"
                    ),
                }
                response["stat_narrative"] = meta["narrative"]
                response["clustering_leaderboard"] = meta.get("clustering_leaderboard", [])
                response["classifier_leaderboard"]  = meta.get("classifier_leaderboard", [])
                response["classifier_model"]        = meta.get("classifier_model")
                response["classifier_skipped_reason"] = meta.get("classifier_skipped_reason")
                response["kprototypes_note"]        = meta.get("kprototypes_note")

                if fi_df is not None:
                    response["feature_importance"] = {
                        "columns": fi_df.columns.tolist(),
                        "rows":    fi_df.values.tolist(),
                    }

                response["chart"]    = build_auto_chart(df_cluster, "scatter", question)
                response["insights"] = insight_eng.generate(
                    question, "clustering", df_cluster.head(100),
                    response["metrics"],
                    extra_context=meta["narrative"],
                )

            # ── SUPERVISED CLASSIFICATION (TRAINING) ──────────────────────────
            # Trains against a REAL label column already in the data — distinct
            # from "clustering" above, which discovers groups with no ground truth.
            elif intent_type == "classification":
                target_col, feature_cols = intent.get("target_col"), intent.get("feature_cols")
                if not target_col or not feature_cols:
                    return {
                        **response,
                        "error": (
                            "Could not identify a label column and feature columns for classification. "
                            "Specify which existing column holds the category you want to predict."
                        ),
                    }

                # Defensive guard, same pattern as regression: never let the
                # label sit inside its own feature list.
                feature_cols = [c for c in feature_cols if c != target_col]
                if not feature_cols:
                    return {**response, "error": "No usable feature columns remain after excluding the label column."}

                df_clean = self._get_clean_dataframe(intent, target_tables)

                # Reuse an existing trained classifier for this label column if one
                # exists in the session, otherwise train a fresh one. Re-running the
                # same classification question re-trains (intentional — lets the user
                # retrain after uploading corrected data without restarting the session).
                clf_engine = SupervisedClassifierEngine()
                result, engine_err = self._run_engine_safely(
                    clf_engine.run, "Classification", df_clean, feature_cols, target_col
                )
                if engine_err:
                    return {**response, "error": engine_err}

                if "error" in result:
                    return {**response, "error": result["error"]}

                self.classifiers[target_col] = clf_engine   # persist trained model in session

                df_result = result["data"]
                meta      = result["meta"]
                fi_df     = result.get("feature_importance")

                response.update(df_to_rows(df_result))
                response["metrics"] = {
                    "Accuracy":        f"{meta['accuracy']*100:.1f}%",
                    "F1 (weighted)":   f"{meta['f1_weighted']:.3f}",
                    "Model":           meta.get("classifier_model", "—"),
                    "Classes":         str(meta["n_classes"]),
                    "Label Column":    meta["label_col"],
                }
                response["stat_narrative"]     = meta["narrative"]
                response["confusion_matrix"]   = {"classes": meta["classes"], "matrix": meta["confusion_matrix"]}
                response["label_distribution"] = meta["label_distribution"]
                response["classifier_leaderboard"] = meta.get("classifier_leaderboard", [])
                response["classifier_skipped_reason"] = meta.get("classifier_skipped_reason")

                if fi_df is not None:
                    response["feature_importance"] = {
                        "columns": fi_df.columns.tolist(),
                        "rows":    fi_df.values.tolist(),
                    }

                response["chart"]    = build_classification_chart(result)
                response["insights"] = insight_eng.generate(
                    question, "classification", df_result.head(100),
                    response["metrics"],
                    extra_context=meta["narrative"],
                )

        except Exception as e:
            response["error"] = str(e)

        return _deep_sanitize(response)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.post("/inspect_file")
async def inspect_file(file: UploadFile = File(...)):
    """
    Reads just the sheet NAMES from an uploaded Excel file (no data is loaded,
    no session is touched) so the frontend can show a sheet picker before
    committing to loading any of them. CSVs are always single-table, so they
    report multi_sheet: False and an empty sheet list — the frontend should
    skip the picker entirely for those and just call /upload directly.
    """
    suffix = Path(file.filename).suffix
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(400, "Only CSV and Excel files are supported.")

    if suffix == ".csv":
        return {"multi_sheet": False, "sheets": []}

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        sheets = AnalyticsEngine().inspect_excel_sheets(tmp_path)
    except Exception as e:
        raise HTTPException(500, f"Failed to read Excel file: {e}")
    finally:
        import time, gc
        gc.collect()
        time.sleep(0.2)
        try:
            os.remove(tmp_path)
        except Exception as cleanup_err:
            print("Temp cleanup failed:", cleanup_err)

    return {"multi_sheet": len(sheets) > 1, "sheets": sheets}


@app.post("/upload/{session_id}")
async def upload_file(
    session_id: str,
    file: UploadFile = File(...),
    # JSON-encoded list of sheet names to load, e.g. '["acr"]' — sent as a
    # regular multipart form field alongside the file. Optional and backward
    # compatible: omitted (or null) still loads every sheet, same as before.
    # Only meaningful for Excel files; ignored for CSV.
    sheets: Optional[str] = Form(None),
):
    session = get_session(session_id)
    engine: AnalyticsEngine = session["engine"]

    suffix = Path(file.filename).suffix
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(400, "Only CSV and Excel files are supported.")

    selected_sheets = None
    if sheets:
        try:
            selected_sheets = json.loads(sheets)
            if not isinstance(selected_sheets, list):
                raise ValueError
        except (json.JSONDecodeError, ValueError):
            raise HTTPException(400, "`sheets` must be a JSON array of sheet names.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # original_filename passed through explicitly — tmp_path is a random
        # temp-file name, so without this a CSV's table name had nothing
        # real to derive from (see load_file()'s docstring for the bug this
        # was causing: every CSV silently overwrote the previous one).
        info = engine.load_file(tmp_path, original_filename=file.filename, sheets=selected_sheets)
    except Exception as e:
        raise HTTPException(500, f"Failed to load file: {e}")
    finally:
        import time, gc
        gc.collect()
        # Only retry-with-a-brief-pause if deletion ACTUALLY fails (the
        # Windows file-lock case this was guarding against) — previously
        # this slept a flat 0.5s on every single upload regardless of
        # whether it was ever needed.
        for attempt in range(5):
            try:
                os.remove(tmp_path)
                break
            except PermissionError:
                time.sleep(0.1)
            except Exception as cleanup_err:
                print("Temp cleanup failed:", cleanup_err)
                break

    active_df = engine.get_dataframe(info["active_table"])
    schema    = engine.get_all_schemas()
    overview  = summarize_dataset_bounded(active_df, schema)

    return {**info, "overview": overview}


@app.get("/connect/mysql/tables")
async def list_mysql_tables():
    """
    Lists tables in the configured MySQL database, for the table-browser
    sidebar UI — lets the user click a table instead of typing SQL. Doesn't
    touch session state; the actual data load still goes through
    /connect/mysql/{session_id} below.
    """
    try:
        tables = MySQLConnector().list_tables()
    except Exception as e:
        raise HTTPException(500, f"Failed to list MySQL tables: {e}")
    return {"tables": tables}


@app.post("/connect/mysql/{session_id}")
async def connect_mysql(session_id: str, req: MySQLQueryRequest):
    """
    Loads data from MySQL via db_connectors.MySQLConnector, then registers
    it the same way /upload registers a file — same response shape.

    The query is validated read-only (no INSERT/UPDATE/DELETE/DROP/CREATE/
    ALTER, single statement) before it's allowed to run — this closes the
    gap where a user-typed query in the "Advanced" raw-SQL box could
    otherwise execute destructive SQL directly against MySQL with no check.
    """
    session = get_session(session_id)
    engine: AnalyticsEngine = session["engine"]

    try:
        validate_read_only_sql(req.query, dialect="mysql")
    except UnsafeSQLError as e:
        raise HTTPException(400, f"Query rejected for safety: {e}")

    try:
        info = engine.load_from_mysql(req.query, req.table_name)
    except Exception as e:
        raise HTTPException(500, f"Failed to load data from MySQL: {e}")

    active_df = engine.get_dataframe(info["active_table"])
    schema    = engine.get_all_schemas()
    overview  = summarize_dataset_bounded(active_df, schema)

    return {**info, "overview": overview}


@app.get("/connect/snowflake/tables")
async def list_snowflake_tables():
    """
    Lists tables in the configured Snowflake database/schema, for the
    table-browser sidebar UI — identical role to list_mysql_tables() above,
    just against a different external source. Doesn't touch session state.
    """
    try:
        tables = SnowflakeConnector().list_tables()
    except Exception as e:
        raise HTTPException(500, f"Failed to list Snowflake tables: {e}")
    return {"tables": tables}


@app.post("/connect/snowflake/{session_id}")
async def connect_snowflake(session_id: str, req: SnowflakeQueryRequest):
    """
    Loads data from Snowflake via db_connectors.SnowflakeConnector, then
    registers it the same way /upload and /connect/mysql register data —
    identical response shape, so the frontend (and everything downstream:
    joins, combine, dtype detection, health scoring) treats a Snowflake
    table exactly like any other loaded table, no special-casing needed.

    The query is validated read-only under the Snowflake SQL dialect
    (distinct from MySQL's — different quoting/date-function syntax) before
    it's allowed to run, same safety gate as the MySQL route.
    """
    session = get_session(session_id)
    engine: AnalyticsEngine = session["engine"]

    try:
        validate_read_only_sql(req.query, dialect="snowflake")
    except UnsafeSQLError as e:
        raise HTTPException(400, f"Query rejected for safety: {e}")

    try:
        info = engine.load_from_snowflake(req.query, req.table_name)
    except Exception as e:
        raise HTTPException(500, f"Failed to load data from Snowflake: {e}")

    active_df = engine.get_dataframe(info["active_table"])
    schema    = engine.get_all_schemas()
    overview  = summarize_dataset_bounded(active_df, schema)

    return {**info, "overview": overview}


@app.get("/suggest_joins/{session_id}")
async def suggest_joins(session_id: str):
    """
    Returns columns shared by exactly two currently-loaded tables in this
    session — the data behind the frontend's "combine these tables?"
    suggestion cards. No SQL, no join actually happens here; see
    /combine_tables for that.
    """
    session = get_session(session_id)
    engine: AnalyticsEngine = session["engine"]
    return {"suggestions": engine.suggest_joins()}


@app.post("/combine_tables/{session_id}")
async def combine_tables(session_id: str, req: CombineTablesRequest):
    """
    Joins two already-loaded tables on a shared column and registers the
    result as a new table in this session, the same way /upload and
    /connect/mysql register data. This is the one-click "Combine" action —
    the user never sees or writes SQL for this.
    """
    session = get_session(session_id)
    engine: AnalyticsEngine = session["engine"]

    result = engine.combine_tables(req.table_a, req.table_b, req.columns, req.new_table_name)
    if "error" in result:
        raise HTTPException(400, result["error"])

    active_df = engine.get_dataframe(result["active_table"])
    schema    = engine.get_all_schemas()
    overview  = summarize_dataset_bounded(active_df, schema)

    return {**result, "overview": overview}


@app.get("/table/{session_id}/{table_name}/export")
async def export_table(session_id: str, table_name: str, format: str = "csv"):
    """
    Exports a whole loaded table (not a query result — that's ExportMenu's
    job, and is capped at 500 rows via df_to_rows). Backs the top bar's
    Export button, which previously had no handler at all.
    """
    session = get_session(session_id)
    engine: AnalyticsEngine = session["engine"]

    df = engine.get_dataframe(table_name)
    if df is None:
        raise HTTPException(404, f"Table '{table_name}' is not loaded.")

    import io

    if format == "csv":
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        content, media_type, filename = buf.getvalue(), "text/csv", f"{table_name}.csv"
    elif format == "json":
        content = df.to_json(orient="records", indent=2, date_format="iso")
        media_type, filename = "application/json", f"{table_name}.json"
    elif format == "xlsx":
        buf = io.BytesIO()
        df.to_excel(buf, index=False, engine="openpyxl")
        content, media_type, filename = buf.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", f"{table_name}.xlsx"
    else:
        raise HTTPException(400, f"Unsupported format: {format}")

    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.delete("/table/{session_id}/{table_name}")
async def remove_table(session_id: str, table_name: str):
    """
    Unloads a single table from this session — the "click again to remove"
    counterpart to /upload and /connect/mysql. Works for any table
    regardless of source (an uploaded sheet, a MySQL table, or a combined
    table), since AnalyticsEngine.remove_table() has no notion of origin.
    """
    session = get_session(session_id)
    engine: AnalyticsEngine = session["engine"]

    result = engine.remove_table(table_name)
    if "error" in result:
        raise HTTPException(404, result["error"])

    if result["active_table"]:
        active_df = engine.get_dataframe(result["active_table"])
        schema    = engine.get_all_schemas()
        overview  = summarize_dataset_bounded(active_df, schema)
        return {**result, "overview": overview}

    # No tables left at all — nothing to summarize.
    return {**result, "overview": None}


@app.get("/shared_columns/{session_id}/{table_a}/{table_b}")
async def shared_columns(session_id: str, table_a: str, table_b: str):
    """
    Powers the join-picker UI: once the user selects two tables, this
    returns every non-generic column shared between them, so the picker can
    offer real choices instead of a free-text field. Unlike suggest_joins(),
    this ignores combined_pairs — the picker should work for any two
    tables, even ones already manually combined before.
    """
    session = get_session(session_id)
    engine: AnalyticsEngine = session["engine"]
    result = engine.get_shared_columns(table_a, table_b)
    if result.get("error"):
        raise HTTPException(404, result["error"])
    return result


@app.post("/query/{session_id}")
async def query(session_id: str, req: QueryRequest):
    session = get_session(session_id)
    engine: AnalyticsEngine       = session["engine"]
    cluster_engine: ClusterEngine = session["cluster_engine"]
    classifiers: dict             = session["classifiers"]

    if not engine.tables:
        raise HTTPException(400, "No file uploaded yet. Please upload a CSV or Excel file first.")

    dispatcher = AnalyticsDispatcher(engine, cluster_engine, classifiers)
    join_override = req.join_override.dict() if req.join_override else None
    response   = dispatcher.handle_query(
        req.question.strip(), req.chart_type,
        join_override=join_override,
        skip_join_clarification=req.skip_join_clarification,
    )

    session["history"].append({
        "question": response["question"],
        "type":     response["type"],
    })
    if len(session["history"]) > 20:
        session["history"] = session["history"][-20:]

    return response


@app.post("/predict_cluster/{session_id}")
async def predict_cluster(session_id: str, req: PredictPointRequest):
    session = get_session(session_id)
    cluster_engine: ClusterEngine = session["cluster_engine"]

    try:
        result = cluster_engine.predict_point(req.point)
        return _deep_sanitize(result)
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}


@app.post("/predict_label/{session_id}")
async def predict_label(session_id: str, req: PredictLabelRequest):
    session = get_session(session_id)
    classifiers: dict = session["classifiers"]

    clf_engine = classifiers.get(req.label_col)
    if clf_engine is None:
        return {
            "error": (
                f"No trained classifier found for label column '{req.label_col}'. "
                "Train one first by asking a classification question against labeled data."
            )
        }

    try:
        result = clf_engine.predict_point(req.point)
        return _deep_sanitize(result)
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}


@app.get("/history/{session_id}")
async def get_history(session_id: str):
    session = get_session(session_id)
    return {"history": session["history"]}


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    if session_id in SESSIONS:
        del SESSIONS[session_id]
    return {"cleared": True}


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.1.0"}