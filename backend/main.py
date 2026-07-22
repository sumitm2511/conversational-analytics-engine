import os
import math
import uuid
import tempfile
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from analytics_engine  import AnalyticsEngine
from intent_classifier import IntentClassifier
from sql_generator     import SQLGenerator
from sql_safety        import validate_select_only, UnsafeSQLError
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

class QueryRequest(BaseModel):
    question:   str
    chart_type: Optional[str] = "auto"

class PredictPointRequest(BaseModel):
    point: Dict[str, float | int | str]

class PredictLabelRequest(BaseModel):
    label_col: str
    point: Dict[str, float | int | str]


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

def extract_metrics(df: pd.DataFrame) -> dict:
    if df is None or df.empty or len(df) > 20:
        return {}
    metrics = {}
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols[:4]:
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
        cols = [
            intent.get("date_col"),
            intent.get("value_col"),
            intent.get("group_col"),
            intent.get("target_col"),
        ]
        cols       = [c for c in cols if c] + (intent.get("feature_cols") or [])
        valid_cols = self.engine.tables.get(target_tables[0], {}).get("columns", [])
        safe_cols  = [c for c in set(cols) if c in valid_cols] if cols else []

        if not safe_cols:
            return self.engine.get_dataframe(target_tables[0])

        select_clause = ", ".join([f'"{c}"' for c in safe_cols])
        sql_res       = self.engine.execute_sql(f"SELECT {select_clause} FROM {target_tables[0]}")

        if sql_res["error"] or sql_res["data"] is None:
            return self.engine.get_dataframe(target_tables[0])

        return sql_res["data"]

    def handle_query(self, question: str, chart_type_override: str) -> dict:
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
                label_words = label_col.replace("_", " ").lower()
                if label_words and label_words in q_lower:
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

        if not target_tables or not isinstance(target_tables, list):
            target_tables = [self.engine.active_table]

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
                    if result.get("defaulted_fields"):
                        narrative += f" ⚠️ {result['warning']}"
                    response["stat_narrative"] = narrative

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
                    if result.get("defaulted_fields"):
                        narrative += f" ⚠️ {result['warning']}"
                    response["stat_narrative"] = narrative

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
                sql = sql_gen.generate_sql(question, enriched_schema, target_tables)

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
                response["insights"] = insight_eng.generate(
                    question, intent_type, df.head(200), response["metrics"]
                )

            elif intent_type == "forecast":
                date_col, value_col = intent.get("date_col"), intent.get("value_col")
                if not date_col or not value_col:
                    return {**response, "error": "Could not identify date and value columns for forecast."}

                df_clean = self._get_clean_dataframe(intent, target_tables)
                periods  = intent.get("periods", 12)
                result   = forecast_eng.run(df_clean, date_col, value_col, periods, intent.get("group_col"))

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
                result   = trend_eng.run(df_clean, date_col, value_col, intent.get("group_col"))

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
                result   = anomaly_eng.run(
                    df_clean, value_col, date_col,
                    intent.get("group_col"),
                    contamination="auto",
                )

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
                result   = regression_eng.run(df_clean, target_col, feature_cols)

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

                domain_context = intent.get("domain_context")
                result = self.cluster_engine.run(df_clean, feature_cols, domain_context=domain_context)

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
                result = clf_engine.run(df_clean, feature_cols, target_col)

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

@app.post("/upload/{session_id}")
async def upload_file(session_id: str, file: UploadFile = File(...)):
    session = get_session(session_id)
    engine: AnalyticsEngine = session["engine"]

    suffix = Path(file.filename).suffix
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(400, "Only CSV and Excel files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        info = engine.load_file(tmp_path)
    except Exception as e:
        raise HTTPException(500, f"Failed to load file: {e}")
    finally:
        import time, gc
        gc.collect()
        time.sleep(0.5)
        try:
            os.remove(tmp_path)
        except Exception as cleanup_err:
            print("Temp cleanup failed:", cleanup_err)

    active_df = engine.get_dataframe(info["active_table"])
    schema    = engine.get_all_schemas()
    overview  = insight_eng.summarize_dataset(active_df, schema)

    return {**info, "overview": overview}


@app.post("/query/{session_id}")
async def query(session_id: str, req: QueryRequest):
    session = get_session(session_id)
    engine: AnalyticsEngine       = session["engine"]
    cluster_engine: ClusterEngine = session["cluster_engine"]
    classifiers: dict             = session["classifiers"]

    if not engine.tables:
        raise HTTPException(400, "No file uploaded yet. Please upload a CSV or Excel file first.")

    dispatcher = AnalyticsDispatcher(engine, cluster_engine, classifiers)
    response   = dispatcher.handle_query(req.question.strip(), req.chart_type)

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