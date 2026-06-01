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
from prediction_engine import ForecastEngine, TrendEngine, RegressionEngine, AnomalyEngine
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

app = FastAPI(title="Analytics AI", version="2.0.0")

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
            "engine":  AnalyticsEngine(),
            "history": [],
        }
    return SESSIONS[session_id]


# ── Request / response models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    chart_type: Optional[str] = "auto"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sanitize(val):
    """Convert nan/inf/NaT to None so JSON serializes cleanly."""
    if val is None:
        return None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    if isinstance(val, pd.Timestamp):
        return str(val)
    return val

def df_to_rows(df: pd.DataFrame, max_rows: int = 500) -> dict:
    """Serialise a DataFrame to JSON-safe rows/columns."""
    if df is None or df.empty:
        return {"columns": [], "rows": [], "total_rows": 0}

    total   = len(df)
    display = df.head(max_rows).copy()

    for col in display.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns:
        display[col] = display[col].astype(str)

    display = display.replace([float("inf"), float("-inf")], None)

    rows = [[_sanitize(cell) for cell in row] for row in display.values.tolist()]

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


# ── Master Dispatcher Class ────────────────────────────────────────────────────

class AnalyticsDispatcher:
    """
    Handles strict LLM routing, safe data extraction, and deterministic engine execution.
    Keeps the FastAPI endpoints clean and modular.
    """
    def __init__(self, engine: AnalyticsEngine):
        self.engine = engine

    def _get_clean_dataframe(self, intent: dict, target_tables: list) -> pd.DataFrame:
        """Safely extracts only the required columns from DuckDB to feed Python models."""
        cols = [
            intent.get("date_col"), 
            intent.get("value_col"), 
            intent.get("group_col"), 
            intent.get("target_col")
        ]
        cols = [c for c in cols if c] + (intent.get("feature_cols") or [])
        
        if not cols:
            return self.engine.get_dataframe(target_tables[0])
            
        # Strict validation against schema to prevent SQL injection or bad column requests
        valid_cols = self.engine.tables.get(target_tables[0], {}).get("columns", [])
        safe_cols = [c for c in set(cols) if c in valid_cols]
        
        if not safe_cols:
            return self.engine.get_dataframe(target_tables[0])
            
        select_clause = ", ".join([f'"{c}"' for c in safe_cols])
        sql_res = self.engine.execute_sql(f"SELECT {select_clause} FROM {target_tables[0]}")
        
        if sql_res["error"] or sql_res["data"] is None:
            return self.engine.get_dataframe(target_tables[0])
            
        return sql_res["data"]

    def handle_query(self, question: str, chart_type_override: str) -> dict:
        # Get enriched schema containing foreign key / join relationships
        enriched_schema = getattr(self.engine, 'get_enriched_schema', self.engine.get_all_schemas)()
        
        # 1. Semantic Routing (Strict JSON)
        intent        = classifier.classify(question, enriched_schema)
        intent_type   = intent.get("type", "sql")
        chart_type    = chart_type_override if chart_type_override != "auto" else intent.get("chart_type", "bar")
        target_tables = intent.get("tables", [])
        
        if not target_tables or not isinstance(target_tables, list):
            target_tables = [self.engine.active_table]

        # Base response structure
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
            # ── SQL & MULTI-TABLE PATH ──
            if intent_type in ("sql", "summary"):
                sql = sql_gen.generate_sql(question, enriched_schema, target_tables)
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
                response["insights"] = insight_eng.generate(question, intent_type, df.head(200), response["metrics"])

            # ── FORECAST PATH ──
            elif intent_type == "forecast":
                date_col, value_col = intent.get("date_col"), intent.get("value_col")
                if not date_col or not value_col:
                    return {**response, "error": "Could not identify date and value columns for forecast."}

                df_clean = self._get_clean_dataframe(intent, target_tables)
                periods = intent.get("periods", 12)
                result = forecast_eng.run(df_clean, date_col, value_col, periods, intent.get("group_col"))

                if "error" in result:
                    return {**response, "error": result["error"]}

                data, stat_meta = result["data"], result.get("meta", {})
                
                if isinstance(data, pd.DataFrame):
                    response.update(df_to_rows(data))
                else:
                    first = list(data.values())[0]
                    response.update(df_to_rows(first) if isinstance(first, pd.DataFrame) else {})

                response["chart"]   = build_forecast_chart(result, date_col, value_col)
                response["metrics"] = {
                    "Periods Forecasted": periods,
                    "Model":              stat_meta.get("model", "Prophet").title(),
                    "Granularity":        stat_meta.get("granularity", "monthly").title(),
                    "Direction":          stat_meta.get("direction", "—").title(),
                    "CAGR":               f"{stat_meta['cagr']*100:.1f}%" if stat_meta.get("cagr") else "—",
                }
                response["insights"] = insight_eng.generate(
                    question, "forecast", data if isinstance(data, pd.DataFrame) else None,
                    response["metrics"], extra_context=stat_meta.get("narrative", ""), stat_meta=stat_meta
                )

            # ── TREND PATH ──
            elif intent_type == "trend":
                date_col, value_col = intent.get("date_col"), intent.get("value_col")
                if not date_col or not value_col:
                    return {**response, "error": "Could not identify date and value columns for trend analysis."}

                df_clean = self._get_clean_dataframe(intent, target_tables)
                result = trend_eng.run(df_clean, date_col, value_col, intent.get("group_col"))

                if "error" in result:
                    return {**response, "error": result["error"]}

                df_trend, stat_meta = result["data"], result.get("meta", {})
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
                    extra_context=stat_meta.get("narrative", ""), stat_meta=stat_meta
                )

            # ── ANOMALY PATH ──
            elif intent_type == "anomaly":
                df_clean = self._get_clean_dataframe(intent, target_tables)
                
                value_col = intent.get("value_col")
                if not value_col:
                    numeric_cols = df_clean.select_dtypes(include="number").columns.tolist()
                    value_col = numeric_cols[0] if numeric_cols else None

                if not value_col:
                    return {**response, "error": "Could not identify a numeric column for anomaly detection."}

                date_col = intent.get("date_col")
                result = anomaly_eng.run(df_clean, value_col, date_col, intent.get("group_col"))

                if "error" in result:
                    return {**response, "error": result["error"]}

                stat_meta = result.get("meta", {})
                response.update(df_to_rows(result["anomalies"]))
                response["metrics"] = {
                    "Anomalies Found": result.get("anomaly_count", 0),
                    "High/Critical":   stat_meta.get("high_severity", 0),
                    "Mean":            result.get("mean", 0),
                    "Std Dev":         result.get("std", 0),
                    "Method":          result.get("method", "rolling_z").replace("_", " ").title(),
                }
                response["chart"]    = build_anomaly_chart(result, date_col, value_col)
                response["insights"] = insight_eng.generate(
                    question, "anomaly", result.get("anomalies", pd.DataFrame()).head(50),
                    response["metrics"], extra_context=f"{result.get('anomaly_count', 0)} anomalies detected. {stat_meta.get('narrative', '')}",
                    stat_meta=stat_meta
                )
                
            # ── REGRESSION PATH ──
            elif intent_type == "regression":
                target_col, feature_cols = intent.get("target_col"), intent.get("feature_cols")
                if not target_col or not feature_cols:
                    return {**response, "error": "Could not identify target and feature columns for regression."}

                df_clean = self._get_clean_dataframe(intent, target_tables)
                result = regression_eng.run(df_clean, target_col, feature_cols)

                if "error" in result:
                    return {**response, "error": result["error"]}

                response.update(df_to_rows(result["data"]))
                response["metrics"] = {k: str(v) for k, v in result["metrics"].items()}
                response["chart"]   = build_regression_chart(result)

                fi_df = result.get("feature_importance")
                if fi_df is not None:
                    response["feature_importance"] = {"columns": fi_df.columns.tolist(), "rows": fi_df.values.tolist()}
                    fi_chart = build_fi_chart(fi_df)
                    if fi_chart:
                        response["chart"] = fi_chart

                response["insights"] = insight_eng.generate(question, "regression", result["data"], result["metrics"])

        except Exception as e:
            response["error"] = str(e)

        return response


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
        except Exception as e:
            print("Temp cleanup failed:", e)

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

    # Utilize the Dispatcher class
    dispatcher = AnalyticsDispatcher(engine)
    response = dispatcher.handle_query(req.question.strip(), req.chart_type)

    # Manage conversational history
    session["history"].append({"question": response["question"], "type": response["type"]})
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
    return {"status": "ok", "version": "2.0.0"}