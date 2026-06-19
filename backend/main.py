"""
main.py  —  FastAPI backend for the Analytics AI platform
"""

import os
import math
import uuid
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from analytics_engine  import AnalyticsEngine
from intent_classifier import IntentClassifier
from sql_generator     import SQLGenerator
from prediction_engine import ForecastEngine, TrendEngine, RegressionEngine, AnomalyEngine, ClusterEngine
from chart_engine      import (
    build_auto_chart,
    build_forecast_chart,
    build_trend_chart,
    build_regression_chart,
    build_fi_chart,
    build_anomaly_chart,
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
# NOTE: ClusterEngine.__init__ no longer touches Groq — safe to instantiate
#       even when GROQ_API_KEY is absent from the environment.

classifier     = IntentClassifier()
sql_gen        = SQLGenerator()
insight_eng    = InsightEngine()
forecast_eng   = ForecastEngine()
trend_eng      = TrendEngine()
regression_eng = RegressionEngine()
anomaly_eng    = AnomalyEngine()
cluster_eng    = ClusterEngine()


# ── In-memory session store ────────────────────────────────────────────────────

SESSIONS: dict[str, dict] = {}

def get_session(session_id: str) -> dict:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "engine":  AnalyticsEngine(),
            "history": [],
        }
    return SESSIONS[session_id]


# ── Request / response models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:   str
    chart_type: Optional[str] = "auto"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _deep_sanitize(obj):
    """
    Recursively sweeps dictionaries, lists, and values to safely unwrap 
    any rogue NumPy types or NaNs before FastAPI converts them to JSON.
    """
    # 1. Safely unwrap NumPy wrappers (np.bool_, np.int64, np.float64)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
            
    # 2. Recursively sweep dictionaries
    if isinstance(obj, dict):
        return {k: _deep_sanitize(v) for k, v in obj.items()}
        
    # 3. Recursively sweep lists/tuples
    if isinstance(obj, list) or isinstance(obj, tuple):
        return [_deep_sanitize(v) for v in obj]
        
    # 4. Handle invalid JSON floats (NaN, Infinity)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
        
    # 5. Stringify Pandas timestamps
    if isinstance(obj, pd.Timestamp):
        return str(obj)
        
    return obj

def df_to_rows(df: pd.DataFrame, max_rows: int = 500) -> dict:
    """Serialise a DataFrame to JSON-safe rows/columns."""
    if df is None or df.empty:
        return {"columns": [], "rows": [], "total_rows": 0}

    total   = len(df)
    display = df.head(max_rows).copy()

    for col in display.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns:
        display[col] = display[col].astype(str)

    display = display.replace([float("inf"), float("-inf")], None)
    
    # Apply deep sanitization to every cell
    rows    = [[_deep_sanitize(cell) for cell in row] for row in display.values.tolist()]

    return {
        "columns":    display.columns.tolist(),
        "rows":       rows,
        "total_rows": total,
    }



def extract_metrics(df: pd.DataFrame) -> dict:
    """Pull simple headline metrics from a small result DF."""
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
    """
    Routes intent → engine → response. All engine calls are isolated
    inside try/except so one broken path can't crash the others.
    """
    def __init__(self, engine: AnalyticsEngine):
        self.engine = engine

    def _get_clean_dataframe(self, intent: dict, target_tables: list) -> pd.DataFrame:
        cols = [
            intent.get("date_col"),
            intent.get("value_col"),
            intent.get("group_col"),
            intent.get("target_col"),
        ]
        cols      = [c for c in cols if c] + (intent.get("feature_cols") or [])
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
            # ── SQL & SUMMARY ──────────────────────────────────────────────────
            if intent_type in ("sql", "summary"):
                sql    = sql_gen.generate_sql(question, enriched_schema, target_tables)
                response["sql"] = sql
                result = self.engine.execute_sql(sql)

                if result["error"]:
                    fixed_sql = sql_gen.fix_sql(sql, result["error"], enriched_schema)
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

            # ── FORECAST ──────────────────────────────────────────────────────
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

                # Surface empirical coverage so the frontend can show it
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

            # ── TREND ─────────────────────────────────────────────────────────
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

            # ── ANOMALY ───────────────────────────────────────────────────────
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
                    contamination="auto",   # always auto-estimate
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

            # ── REGRESSION ────────────────────────────────────────────────────
            elif intent_type == "regression":
                target_col, feature_cols = intent.get("target_col"), intent.get("feature_cols")
                if not target_col or not feature_cols:
                    return {**response, "error": "Could not identify target and feature columns for regression."}

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

            # ── CLUSTERING ────────────────────────────────────────────────────
            elif intent_type == "clustering":
                df_clean     = self._get_clean_dataframe(intent, target_tables)
                feature_cols = intent.get("feature_cols")

                if not feature_cols:
                    feature_cols = df_clean.columns.tolist()

                result = cluster_eng.run(df_clean, feature_cols)

                if "error" in result:
                    return {**response, "error": result["error"]}

                df_cluster = result["data"]
                meta       = result["meta"]
                fi_df      = result.get("feature_importance")

                response.update(df_to_rows(df_cluster))
                response["metrics"] = {
                    "Optimal Clusters":  str(meta["k"]),
                    "Silhouette Score":  f"{meta['silhouette']:.2f}",
                    "#1 Split Factor":   meta["top_driver"].replace("_", " ").title(),
                    "Outliers Isolated": str(meta.get("outlier_count", 0)),
                    "PCA Var (PC1+PC2)": (
                        f"{sum(meta['pca_variance'][:2])*100:.1f}%"
                        if meta.get("pca_variance") else "—"
                    ),
                }
                response["stat_narrative"] = meta["narrative"]

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
    engine: AnalyticsEngine = session["engine"]

    if not engine.tables:
        raise HTTPException(400, "No file uploaded yet. Please upload a CSV or Excel file first.")

    dispatcher = AnalyticsDispatcher(engine)
    response   = dispatcher.handle_query(req.question.strip(), req.chart_type)

    session["history"].append({
        "question": response["question"],
        "type":     response["type"],
    })
    if len(session["history"]) > 20:
        session["history"] = session["history"][-20:]

    return response


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