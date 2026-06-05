"""
prediction_engine.py — AutoML Forecast Engine v10 (Auto-ARIMA Integration)
Fixes: Scrubbed all indentation/invisible space errors.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Optional

# ── Shared helpers ─────────────────────────────────────────────────────────────

def _infer_freq(dates: pd.Series) -> str:
    dts   = pd.to_datetime(dates, errors="coerce").dropna()
    dts   = dts.drop_duplicates().sort_values()
    diffs = dts.diff().dt.days.dropna()
    if diffs.empty: return "MS"
    med = diffs.median()
    if med <= 4:   return "D"
    if med <= 14:  return "W"
    if med <= 45:  return "MS"
    if med <= 110: return "QS"
    return "YS"

def _freq_label(freq: str) -> str:
    return {"D": "daily", "W": "weekly", "MS": "monthly", "QS": "quarterly", "YS": "yearly"}.get(freq, "monthly")

def _cagr(start_val, end_val, n_periods, freq="MS") -> Optional[float]:
    try:
        ppy   = {"D": 365, "W": 52, "MS": 12, "QS": 4, "YS": 1}.get(freq, 12)
        years = n_periods / ppy
        if start_val <= 0 or end_val <= 0 or years <= 0: return None
        return round((end_val / start_val) ** (1 / years) - 1, 4)
    except Exception: return None

def _get_agg_func(col_name: str) -> str:
    col = str(col_name).lower()
    if any(w in col for w in ["%", "percent", "utilization", "rate", "avg", "average", "ratio", "score", "compliance"]):
        return "mean"
    return "sum"

# ── Dynamic Evaluation Metrics ─────────────────────────────────────────────────

def _smape(actual, predicted) -> float:
    actual, predicted = np.array(actual, dtype=float), np.array(predicted, dtype=float)
    denom = (np.abs(actual) + np.abs(predicted)) / 2 + 1e-9
    return float(np.mean(np.abs(actual - predicted) / denom) * 100)

def _mae(actual, predicted) -> float:
    return float(np.mean(np.abs(np.array(actual, dtype=float) - np.array(predicted, dtype=float))))

def _rmse(actual, predicted) -> float:
    return float(np.sqrt(np.mean((np.array(actual, dtype=float) - np.array(predicted, dtype=float)) ** 2)))

def _select_eval_metric(full_series: np.ndarray) -> str:
    """Dynamically chooses the fairest referee by scanning the ENTIRE historical dataset."""
    if len(full_series) == 0: return "mae"
    
    # 1. The Zero Problem
    if np.any(full_series <= 0.01):
        return "mae"
        
    # 2. The Volatility Problem
    cv = np.std(full_series) / (np.mean(full_series) + 1e-9)
    if cv > 0.8:
        return "rmse"
        
    # 3. Default standard
    return "smape"

def _evaluate_forecast(actual, predicted, metric: str) -> float:
    if metric == "mae": return _mae(actual, predicted)
    if metric == "rmse": return _rmse(actual, predicted)
    return _smape(actual, predicted)

def _check_stationarity(series: pd.Series, freq: str) -> dict:
    n = len(series)
    result = {"is_stationary": True, "trend_strength": 0.0, "has_seasonality": False, "needs_log": False}
    if n < 4: return result

    x      = np.arange(n)
    y      = series.fillna(series.mean()).values
    
    coeffs_lin = np.polyfit(x, y, 1)
    pred_lin   = np.polyval(coeffs_lin, x)
    var_y      = np.var(y) + 1e-9
    r2_lin     = 1 - np.var(y - pred_lin) / var_y
    result["trend_strength"] = round(max(0, r2_lin), 3)

    if r2_lin > 0.3: result["is_stationary"] = False

    if y.min() > 0:
        coeffs_log = np.polyfit(x, np.log(y), 1)
        pred_log   = np.exp(np.polyval(coeffs_log, x))
        r2_log     = 1 - np.var(y - pred_log) / var_y
        if r2_log > max(0, r2_lin) + 0.05: result["needs_log"] = True

    sp = {"D": 7, "W": 52, "MS": 12, "QS": 4, "YS": 1}.get(freq, 12)
    if n >= sp * 2:
        try:
            detrended = y - pred_lin
            acf = pd.Series(detrended).autocorr(lag=sp)
            if not np.isnan(acf) and acf > 0.3:
                result["has_seasonality"] = True
        except Exception: pass

    return result

# ── Forecast Engine ────────────────────────────────────────────────────────────

class ForecastEngine:

    def run(self, df, date_col, value_col, periods=6, group_col=None):
        try: periods = max(int(periods), 3)
        except: periods = 6

        if date_col not in df.columns or value_col not in df.columns:
            return {"error": f"Columns '{date_col}' or '{value_col}' not found."}

        cols = [date_col, value_col] + ([group_col] if group_col in df.columns else [])
        df = df[cols].copy()
        df[date_col]  = pd.to_datetime(df[date_col], errors="coerce")
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=[date_col, value_col])

        freq     = _infer_freq(df[date_col])
        agg_func = _get_agg_func(value_col)
        groups   = df[group_col].unique() if group_col in df.columns else [None]
        results  = {}

        for group in groups:
            sub = (df[df[group_col] == group][[date_col, value_col]].copy() if group is not None else df[[date_col, value_col]].copy())
            
            sub = sub.groupby(pd.Grouper(key=date_col, freq=freq))[value_col].agg(agg_func).reset_index()
            full_idx = pd.date_range(sub[date_col].min(), sub[date_col].max(), freq=freq)
            sub = sub.set_index(date_col).reindex(full_idx).rename_axis(date_col).reset_index()

            if agg_func == "sum": sub[value_col] = sub[value_col].fillna(0)
            else: sub[value_col] = sub[value_col].interpolate(method="linear").bfill().ffill()

            n = len(sub)
            label = str(group) if group is not None else "forecast"

            if n < 3:
                results[label] = {"error": f"Need at least 3 time periods (got {n})."}
                continue

            try:
                result_df, meta = self._fit_best_model(sub, date_col, value_col, periods, freq, n)

                if sub[value_col].min() >= 0:
                    for col in ["predicted", "lower_bound", "upper_bound"]:
                        if col in result_df.columns: result_df[col] = result_df[col].clip(lower=0)

                result_df = self._add_rolling(result_df, "actual")
                actual_vals = result_df.loc[~result_df["is_forecast"], "actual"].dropna()

                if len(actual_vals) >= 2:
                    meta["cagr"]      = _cagr(actual_vals.iloc[0], actual_vals.iloc[-1], len(actual_vals), freq)
                    meta["avg_mom"]   = round(actual_vals.pct_change().mean() * 100, 2)
                    meta["direction"] = "upward" if actual_vals.iloc[-1] > actual_vals.iloc[0] else "downward"

                meta.update({"granularity": _freq_label(freq), "periods": periods})
                meta["narrative"] = self._narrative(meta)
                results[label] = {"data": result_df, "meta": meta}
            except Exception as e:
                results[label] = {"error": str(e)}

        if not results: return {"error": "Not enough data to forecast."}
        if len(results) == 1:
            val = list(results.values())[0]
            return val if "error" in val else {"data": val["data"], "meta": val["meta"], "groups": None, "periods": periods}
        return {"data": {k: v["data"] for k, v in results.items() if "data" in v}, "meta": {k: v.get("meta", {}) for k, v in results.items()}, "groups": list(results.keys()), "periods": periods}

    def _fit_best_model(self, sub, date_col, value_col, periods, freq, n):
        series = sub.set_index(date_col)[value_col]
        stats  = _check_stationarity(series, freq)
        model_map = {"prophet": self._run_prophet, "ets": self._run_ets, "arima": self._run_arima, "linear": self._run_linear}

        if n < 6:
            r, m = self._run_linear(sub, date_col, value_col, periods, freq)
            m.update({"model": "linear", "routing_reason": "Insufficient data length; forced baseline linear model."})
            return r, m

        eval_metric = _select_eval_metric(sub[value_col].values)
        scores = self._run_tournament(sub, date_col, value_col, freq, eval_metric)

        allowed_models = ["prophet", "ets", "arima", "linear"]

        if stats["has_seasonality"]:
            allowed_models = ["prophet", "ets"]
            reason = "Strong seasonality detected; restricted to seasonal models."
        else:
            if n < 24:
                allowed_models = ["ets", "arima", "linear"]
                reason = "No seasonality detected; Prophet excluded to prevent small-data overfitting."
            else:
                reason = "Standard algorithmic tournament."

        if stats["needs_log"] and "linear" in allowed_models:
            allowed_models.remove("linear")

        filtered_scores = {k: v for k, v in scores.items() if k in allowed_models}
        if not filtered_scores: filtered_scores = scores 

        best_model = min(filtered_scores, key=filtered_scores.get) if filtered_scores else "ets"
        validation_score = filtered_scores.get(best_model)

        run_fn = model_map.get(best_model, self._run_linear)
        r, m = run_fn(sub, date_col, value_col, periods, freq)

        m.update({"model": best_model, "validation_score": validation_score, "eval_metric": eval_metric.upper(), "routing_reason": reason})
        return r, m

    def _run_tournament(self, sub, date_col, value_col, freq, eval_metric):
        val_size = max(min(int(len(sub) * 0.2), 6), 3) 
        train = sub.iloc[:-val_size].copy()
        val_actuals = sub.iloc[-val_size:][value_col].values
        
        scores = {}
        try:
            p_res, _ = self._run_prophet(train, date_col, value_col, val_size, freq)
            scores['prophet'] = _evaluate_forecast(val_actuals, p_res[p_res['is_forecast']]['predicted'].values, eval_metric)
        except: pass
        try:
            e_res, _ = self._run_ets(train, date_col, value_col, val_size, freq)
            scores['ets'] = _evaluate_forecast(val_actuals, e_res[e_res['is_forecast']]['predicted'].values, eval_metric)
        except: pass
        try:
            a_res, _ = self._run_arima(train, date_col, value_col, val_size, freq)
            scores['arima'] = _evaluate_forecast(val_actuals, a_res[a_res['is_forecast']]['predicted'].values, eval_metric)
        except: pass
        try:
            l_res, _ = self._run_linear(train, date_col, value_col, val_size, freq)
            scores['linear'] = _evaluate_forecast(val_actuals, l_res[l_res['is_forecast']]['predicted'].values, eval_metric)
        except: pass

        return scores

    # ── Full model implementations ───────────────────────────────────────────

    def _run_linear(self, sub, date_col, value_col, periods, freq):
        x      = np.arange(len(sub))
        y      = sub[value_col].values
        coeffs = np.polyfit(x, y, 1)

        resid_std = np.std(y - np.polyval(coeffs, x)) + 1e-9
        z         = 1.645 

        hist_df = sub.rename(columns={value_col: "actual"}).copy()
        hist_df["predicted"]   = np.polyval(coeffs, x).round(2)
        hist_df["lower_bound"] = (hist_df["predicted"] - z * resid_std).round(2)
        hist_df["upper_bound"] = (hist_df["predicted"] + z * resid_std).round(2)
        hist_df["is_forecast"] = False

        last_date    = sub[date_col].iloc[-1]
        future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]
        future_x = np.arange(len(sub), len(sub) + periods)
        future_y = np.polyval(coeffs, future_x)

        fore_df = pd.DataFrame({
            date_col:      future_dates.strftime("%Y-%m-%d"),
            "actual":      np.nan,
            "predicted":   future_y.round(2),
            "lower_bound": (future_y - z * resid_std).round(2),
            "upper_bound": (future_y + z * resid_std).round(2),
            "is_forecast": True,
        })

        hist_df[date_col] = hist_df[date_col].dt.strftime("%Y-%m-%d")
        return pd.concat([hist_df, fore_df], ignore_index=True), {"trend_strength": None, "seasonality_detected": False}

    def _run_ets(self, sub, date_col, value_col, periods, freq):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        series = sub.set_index(date_col)[value_col]
        sp     = {"D":7,"W":52,"MS":12,"QS":4,"YS":1}.get(freq,12)
        
        # EXCEL OVERRIDE: Excel only requires 1 cycle (sp + 2). Python strictly wants 2 (sp * 2).
        use_s  = len(series) >= sp + 2 

        try:
            # Attempt 1: Standard Academic Optimization 
            model = ExponentialSmoothing(
                series, trend="add", seasonal="add" if use_s else None, seasonal_periods=sp if use_s else None, initialization_method="estimated"
            ).fit(optimized=True)
        except Exception:
            # Attempt 2: The "Excel Mode" Aggressive Fallback
            # If the strict Python method fails on short data, we manually inject the first 12 months as the seasonal baseline.
            if use_s and len(series) < (sp * 2):
                init_level = series.iloc[0]
                s_init = series.iloc[:sp].values - init_level 
                try:
                    model = ExponentialSmoothing(
                        series, trend="add", seasonal="add", seasonal_periods=sp, 
                        initialization_method="known", initial_level=init_level, 
                        initial_trend=0, initial_seasonal=s_init
                    ).fit(optimized=True)
                except:
                    # Absolute safety fallback
                    model = ExponentialSmoothing(series, trend="add", initialization_method="heuristic").fit(optimized=True)
                    use_s = False
            else:
                model = ExponentialSmoothing(series, trend="add", seasonal="add" if use_s else None, seasonal_periods=sp if use_s else None, initialization_method="heuristic").fit(optimized=True)

        fc        = model.forecast(periods)
        resid_std = model.resid.std() + 1e-9
        z         = 1.645

        hist_df                   = sub.rename(columns={value_col: "actual"}).copy()
        hist_df["predicted"]      = model.fittedvalues.values.round(2)
        hist_df["lower_bound"]    = (hist_df["predicted"] - z * resid_std).round(2)
        hist_df["upper_bound"]    = (hist_df["predicted"] + z * resid_std).round(2)
        hist_df["is_forecast"]    = False

        last_date    = series.index[-1]
        future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]

        fore_df = pd.DataFrame({
            date_col:      future_dates.strftime("%Y-%m-%d"),
            "actual":      np.nan,
            "predicted":   fc.values.round(2),
            "lower_bound": (fc.values - z * resid_std).round(2),
            "upper_bound": (fc.values + z * resid_std).round(2),
            "is_forecast": True,
        })

        hist_df = hist_df.reset_index(drop=True)
        hist_df[date_col] = hist_df[date_col].dt.strftime("%Y-%m-%d")

        return pd.concat([hist_df, fore_df], ignore_index=True), {"trend_strength": None, "seasonality_detected": use_s}

    def _run_prophet(self, sub, date_col, value_col, periods, freq):
        from prophet import Prophet
        import logging
        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

        df_p = sub.rename(columns={date_col: "ds", value_col: "y"}).dropna(subset=["y"])

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=(freq in ("D", "W")),
            daily_seasonality=False,
            interval_width=0.90,
            changepoint_prior_scale=0.05,
        )
        model.fit(df_p[["ds", "y"]])

        future   = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)

        result_df = (
            forecast[["ds","yhat","yhat_lower","yhat_upper"]]
            .merge(df_p[["ds","y"]], on="ds", how="left")
            .rename(columns={"ds": date_col, "yhat": "predicted", "yhat_lower": "lower_bound", "yhat_upper": "upper_bound", "y": "actual"})
        )
        result_df["is_forecast"] = result_df[date_col] > df_p["ds"].max()

        for col in ["predicted", "lower_bound", "upper_bound"]:
            result_df[col] = result_df[col].round(2)

        result_df[date_col] = result_df[date_col].dt.strftime("%Y-%m-%d")

        ts = round(abs(forecast["trend"].diff().mean()) / (forecast["yhat"].std() + 1e-9), 4)
        return result_df, {"trend_strength": ts, "seasonality_detected": True}

    def _run_arima(self, sub, date_col, value_col, periods, freq):
        import pmdarima as pm

        series = sub.set_index(date_col)[value_col]

        model = pm.auto_arima(
            series,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True, 
            error_action="ignore",
            max_p=3, max_q=3
        )

        fc, conf_int = model.predict(n_periods=periods, return_conf_int=True, alpha=0.10)
        
        fitted = model.predict_in_sample()
        resid_std = np.std(series.values - fitted) + 1e-9

        hist_df                   = sub.rename(columns={value_col: "actual"}).copy()
        hist_df["predicted"]      = np.round(fitted, 2)
        hist_df["lower_bound"]    = np.round(hist_df["predicted"] - 1.645 * resid_std, 2)
        hist_df["upper_bound"]    = np.round(hist_df["predicted"] + 1.645 * resid_std, 2)
        hist_df["is_forecast"]    = False

        last_date    = series.index[-1]
        future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]

        fore_df = pd.DataFrame({
            date_col:      future_dates.strftime("%Y-%m-%d"),
            "actual":      np.nan,
            "predicted":   np.round(fc.values, 2),
            "lower_bound": np.round(conf_int[:, 0], 2),
            "upper_bound": np.round(conf_int[:, 1], 2),
            "is_forecast": True,
        })

        hist_df = hist_df.reset_index(drop=True)
        hist_df[date_col] = hist_df[date_col].dt.strftime("%Y-%m-%d")

        return pd.concat([hist_df, fore_df], ignore_index=True), {"trend_strength": None, "seasonality_detected": False}

    def _add_rolling(self, df, col):
        c                = df[col]
        df["rolling_3"]  = c.rolling(3,  min_periods=1).mean().round(2)
        df["rolling_6"]  = c.rolling(6,  min_periods=1).mean().round(2)
        df["mom_pct"]    = c.pct_change(1).mul(100).round(2)
        return df

    def _narrative(self, meta) -> str:
        model   = str(meta.get("model", "forecast")).upper()
        score   = meta.get("validation_score")
        metric  = meta.get("eval_metric", "SMAPE")
        reason  = meta.get("routing_reason", "")
        parts   = []

        if reason:
            parts.append(f"**{model}** was prioritized: {reason}")
            if score is not None:
                score_display = f"{score:.1f}%" if metric == "SMAPE" else f"{score:.2f}"
                parts.append(f"(Evaluated using {metric}: {score_display}).")
        else:
            parts.append(f"**{model}** was used due to standard dataset length limitations.")

        if meta.get("direction") and meta.get("avg_mom") is not None:
            parts.append(f"Historical values trended {meta['direction']} at {meta['avg_mom']:+.1f}% avg MoM change.")

        parts.append("Shaded band = 90% confidence interval.")
        return " ".join(parts)

# ── Trend Engine ───────────────────────────────────────────────────────────────

class TrendEngine:

    def run(self, df, date_col, value_col, group_col=None, window=3):
        if date_col not in df.columns or value_col not in df.columns:
            return {"error": f"Columns '{date_col}' or '{value_col}' not found."}

        df            = df.copy()
        df[date_col]  = pd.to_datetime(df[date_col], errors="coerce")
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df            = df.dropna(subset=[date_col, value_col])
        agg_func      = _get_agg_func(value_col)

        def _compute(sub):
            freq = _infer_freq(sub[date_col])
            sub  = (sub.groupby(pd.Grouper(key=date_col, freq=freq))[value_col].agg(agg_func).reset_index().dropna().sort_values(date_col))

            vals, n = sub[value_col], len(sub)

            sub["rolling_3"]  = vals.rolling(3,  min_periods=1).mean().round(2)
            sub["rolling_6"]  = vals.rolling(6,  min_periods=1).mean().round(2)
            sub["rolling_12"] = vals.rolling(12, min_periods=1).mean().round(2)
            sub["mom_pct"]    = vals.pct_change(1).mul(100).round(2)
            sub["cumulative"] = vals.cumsum().round(2)
            sub["momentum"]   = sub["mom_pct"].diff().round(2)

            stl_meta = {}
            if n >= 24:
                try:
                    from statsmodels.tsa.seasonal import STL
                    res          = STL(sub.set_index(date_col)[value_col], period=12, robust=True).fit()
                    sub["trend_component"]    = res.trend.values
                    sub["seasonal_component"] = res.seasonal.values
                    sub["residual_component"] = res.resid.values
                    ts = max(0, round(1 - np.var(res.resid) / (np.var(res.trend + res.resid) + 1e-9), 3))
                    stl_meta = {"trend_strength": ts, "seasonality_detected": True}
                except Exception:
                    stl_meta = {"trend_strength": None, "seasonality_detected": False}
            elif n >= 2:
                x = np.arange(n)
                slope, intercept = np.polyfit(x, vals.fillna(vals.mean()), 1)
                sub["trend_component"] = np.polyval([slope, intercept], x).round(2)
                r2 = max(0, 1 - np.var(vals - sub["trend_component"]) / (np.var(vals) + 1e-9))
                stl_meta = {"trend_strength": round(r2, 3), "seasonality_detected": False}

            rm = vals.rolling(6, min_periods=2).mean()
            rs = vals.rolling(6, min_periods=2).std().replace(0, np.nan)
            sub["spike_zscore"] = ((vals - rm) / rs).round(3)
            sub["is_spike"]     = sub["spike_zscore"].abs() > 2.0

            direction = "upward" if vals.iloc[-1] > vals.iloc[0] else "downward"
            sub[date_col] = sub[date_col].dt.strftime("%Y-%m-%d")

            meta = {
                **stl_meta,
                "cagr":         _cagr(vals.iloc[0], vals.iloc[-1], n, freq),
                "avg_mom_pct":  round(sub["mom_pct"].mean(), 2) if n > 1 else None,
                "direction":    direction,
                "granularity":  _freq_label(freq),
                "periods":      n,
                "spike_count":  int(sub["is_spike"].sum()),
                "narrative":    self._narrative(direction, round(sub["mom_pct"].mean(), 2) if n > 1 else None, _cagr(vals.iloc[0], vals.iloc[-1], n, freq), stl_meta.get("trend_strength"), n),
            }
            return sub, meta

        if group_col and group_col in df.columns:
            parts, metas = [], {}
            for grp, gdf in df.groupby(group_col):
                r, m         = _compute(gdf)
                r[group_col] = grp
                parts.append(r)
                metas[str(grp)] = m
            return {"data": pd.concat(parts, ignore_index=True), "window": window, "meta": metas}

        result_df, meta = _compute(df)
        return {"data": result_df, "window": window, "meta": meta}

    def _narrative(self, direction, avg_mom, cagr, trend_strength, n) -> str:
        parts = []
        if avg_mom is not None:
            strength = "consistently" if (trend_strength or 0) > 0.6 else "moderately"
            parts.append(f"Values have {strength} trended {direction} by an average of {abs(avg_mom):.1f}% per period over {n} periods.")
        if cagr is not None: parts.append(f"CAGR: {cagr*100:.1f}%.")
        if trend_strength is not None:
            label = "very strong" if trend_strength > 0.8 else "strong" if trend_strength > 0.6 else "moderate" if trend_strength > 0.3 else "weak"
            parts.append(f"Trend strength: {trend_strength:.2f}/1.0 ({label}).")
        return " ".join(parts) if parts else "Insufficient data for trend narrative."

# ── Regression Engine ──────────────────────────────────────────────────────────

class RegressionEngine:
    def run(self, df, target_col, feature_cols, model_type="gradient_boosting"):
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_absolute_error
        from sklearn.preprocessing import LabelEncoder

        missing = [c for c in [target_col] + feature_cols if c not in df.columns]
        if missing: return {"error": f"Columns not found: {missing}"}

        sub = df[[target_col] + feature_cols].copy().dropna()
        if len(sub) < 20: return {"error": "Need at least 20 rows for regression."}

        for col in feature_cols:
            if sub[col].dtype == object: sub[col] = LabelEncoder().fit_transform(sub[col].astype(str))

        for col in feature_cols: sub[col] = pd.to_numeric(sub[col], errors="coerce")
        sub = sub.dropna()

        X = sub[feature_cols].values
        y = sub[target_col].values

        if len(sub) > 5000:
            idx = np.random.choice(len(sub), 5000, replace=False)
            X, y = X[idx], y[idx]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42).fit(X_train, y_train)

        preds = model.predict(X_test)
        r2    = round(r2_score(y_test, preds), 4)
        mae   = round(mean_absolute_error(y_test, preds), 4)
        mape  = round(float(np.mean(np.abs((y_test - preds) / (np.abs(y_test) + 1e-9)) * 100)), 2)

        fi_df = pd.DataFrame(sorted(zip(feature_cols, model.feature_importances_), key=lambda x: x[1], reverse=True), columns=["feature", "importance"])
        fi_df["importance"] = fi_df["importance"].round(4)

        comparison = pd.DataFrame({
            "actual":    y_test.round(2),
            "predicted": preds.round(2),
            "residual":  (y_test - preds).round(2),
            "pct_error": np.abs((y_test - preds) / (np.abs(y_test) + 1e-9) * 100).round(1),
        })

        return {"data": comparison, "metrics": {"R2": r2, "MAE": mae, "MAPE%": mape, "samples": len(sub)}, "feature_importance": fi_df, "target": target_col}

# ── Anomaly Engine ─────────────────────────────────────────────────────────────

class AnomalyEngine:
    def run(self, df, value_col, date_col=None, group_col=None, z_threshold=2.5, method="auto", **kwargs):
        if value_col not in df.columns: return {"error": f"Column '{value_col}' not found."}

        sub            = df.copy()
        sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
        sub            = sub.dropna(subset=[value_col])

        unique_dates = 0
        if date_col and date_col in sub.columns:
            sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce")
            unique_dates  = sub[date_col].nunique()

        if method == "auto":
            method = "isolation_forest" if unique_dates < 10 else "rolling_z"

        if method == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            iso    = IsolationForest(contamination=0.05, random_state=42)
            preds  = iso.fit_predict(sub[[value_col]])
            scores = iso.decision_function(sub[[value_col]])
            sub["anomaly_score"] = -scores
            rng = (sub["anomaly_score"].max() - sub["anomaly_score"].min() + 1e-9)
            sub["anomaly_score"] = ((sub["anomaly_score"] - sub["anomaly_score"].min()) / rng * 5).round(3)
            sub["is_anomaly"] = preds == -1
            method_used = "Isolation Forest"

            if date_col in sub.columns:
                sub = sub.sort_values(date_col)
                sub[date_col] = sub[date_col].dt.strftime("%Y-%m-%d")
        else:
            freq     = _infer_freq(sub[date_col])
            agg_func = _get_agg_func(value_col)
            sub = (sub.groupby(pd.Grouper(key=date_col, freq=freq))[value_col].agg(agg_func).reset_index().dropna().sort_values(date_col))
            sub          = self._rolling_zscore(sub, value_col, z_threshold)
            sub[date_col] = sub[date_col].dt.strftime("%Y-%m-%d")
            method_used  = "Rolling Z-Score"

        if len(sub) == 0: return {"error": "No valid data for anomaly detection."}

        sub["severity"] = sub["anomaly_score"].apply(lambda s: "Critical" if s > 4 else "High" if s > 3 else "Medium" if s > 2 else "Low")

        anomalies = sub[sub["is_anomaly"]].copy()
        narr = f"{len(anomalies)} anomalies detected using {method_used}."

        return {"data": sub, "anomalies": anomalies, "anomaly_count": int(len(anomalies)), "mean": round(sub[value_col].mean(), 2), "std": round(sub[value_col].std(), 2), "method": method_used, "meta": {"high_severity": int(anomalies["severity"].isin(["High", "Critical"]).sum()), "narrative": narr}}

    def _rolling_zscore(self, df, value_col, threshold):
        vals = df[value_col]
        rm   = vals.rolling(5, min_periods=2).mean().fillna(vals.mean())
        rs   = (vals.rolling(5, min_periods=2).std().fillna(vals.std()).replace(0, np.nan).fillna(vals.std() + 1e-9))
        df["anomaly_score"] = ((vals - rm) / rs).abs().round(3)
        df["is_anomaly"]    = df["anomaly_score"] > threshold
        return df