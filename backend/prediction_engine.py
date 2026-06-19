"""
prediction_engine.py — AutoML Forecast Engine v20 (Enterprise Edition: HDBSCAN + UMAP)

Fixes & Upgrades:
────────────────────────────────────────────────────────────────────
FORECAST ENGINE
  [BUG]  ETS seasonal guard was `len >= sp` (too loose). Changed to
         `len >= sp * 2` in BOTH _forecast_ets and _fit_ets.
  [BUG]  _forecast_ets CV function used the same loose guard. Fixed.
  [NEW]  Short-series detection and analyst warning.
  [NEW]  Short-series model preference (Theta/Linear prioritized).
  [NEW]  Volatility warning in narrative when CV > 0.25.
  [NEW]  MAE-as-%-of-mean added to meta as `mae_pct_mean`.
  [NEW]  Auto groupby suggestion for frontend.

CLUSTER ENGINE (V20 Upgrade)
  [NEW]  Replaced K-Means with HDBSCAN for density-based clustering.
  [NEW]  Replaced PCA with UMAP for superior non-linear 2D projection.
  [NEW]  Native noise isolation (removed redundant Isolation Forest).
  [NEW]  Graceful fallbacks: Reverts to K-Means/PCA if libraries are missing or data is too uniform.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List
import logging
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)


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

def _periods_per_year(freq: str) -> int:
    return {"D": 365, "W": 52, "MS": 12, "QS": 4, "YS": 1}.get(freq, 12)

def _cagr(start_val, end_val, n_periods, freq="MS") -> Optional[float]:
    try:
        years = n_periods / _periods_per_year(freq)
        if start_val <= 0 or end_val <= 0 or years <= 0: return None
        return round((end_val / start_val) ** (1 / years) - 1, 4)
    except Exception: return None

def _get_agg_func(col_name: str) -> str:
    col = str(col_name).lower()
    if any(w in col for w in ["%", "percent", "utilization", "rate", "avg", "average", "ratio", "score", "compliance", "margin"]):
        return "mean"
    return "sum"


# ── Evaluation metrics ─────────────────────────────────────────────────────────

def _smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    a, p  = np.array(actual, dtype=float), np.array(predicted, dtype=float)
    denom = (np.abs(a) + np.abs(p)) / 2 + 1e-9
    return float(np.mean(np.abs(a - p) / denom) * 100)

def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(np.array(actual, dtype=float) - np.array(predicted, dtype=float))))

def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.array(actual, dtype=float) - np.array(predicted, dtype=float)) ** 2)))

def _all_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict:
    return {
        "MAE":   round(_mae(actual, predicted),   4),
        "RMSE":  round(_rmse(actual, predicted),  4),
        "SMAPE": round(_smape(actual, predicted), 2),
    }

def _select_primary_metric(series: np.ndarray) -> str:
    if len(series) == 0: return "MAE"
    zero_ratio = np.sum(series <= 0.01) / len(series)
    if zero_ratio > 0.10:
        return "MAE"
    if series.min() >= 0 and series.max() <= 100 and np.mean(series) < 100:
        return "MAE"
    cv = np.std(series) / (np.mean(series) + 1e-9)
    if cv > 0.8: return "RMSE"
    return "SMAPE"


# ── Shared CI helper ───────────────────────────────────────────────────────────

def _compute_ci(
    resid_std: float,
    horizon: int,
    z: float = 1.645,
    growth_factor: float = 0.13,
) -> np.ndarray:
    return np.array([
        z * resid_std * np.sqrt(1 + i * growth_factor)
        for i in range(1, horizon + 1)
    ])


# ── Series profiling & Outlier Smoothing ───────────────────────────────────────

def _smooth_outliers(series: pd.Series) -> pd.Series:
    vals = series.copy()
    rm   = vals.rolling(5, min_periods=2, center=True).mean().fillna(vals.mean())
    rs   = (
        vals.rolling(5, min_periods=2, center=True)
            .std()
            .fillna(vals.std())
            .replace(0, np.nan)
            .fillna(vals.std() + 1e-9)
    )
    z_scores = ((vals - rm) / rs).abs()
    vals[z_scores > 3.0] = rm[z_scores > 3.0]
    return vals

def _profile_series(series: pd.Series, freq: str) -> Dict:
    n      = len(series)
    y      = series.fillna(series.mean()).values
    result = {"n": n, "has_trend": False, "has_seasonality": False,
              "trend_strength": 0.0, "cv": 0.0, "needs_log": False}
    if n < 3: return result

    mean = np.mean(y) + 1e-9
    result["cv"] = round(np.std(y) / abs(mean), 3)

    x          = np.arange(n)
    coeffs_lin = np.polyfit(x, y, 1)
    pred_lin   = np.polyval(coeffs_lin, x)
    r2_lin     = max(0, 1 - np.var(y - pred_lin) / (np.var(y) + 1e-9))
    result["trend_strength"] = round(r2_lin, 3)
    result["has_trend"]      = r2_lin > 0.15

    if y.min() > 0:
        coeffs_log = np.polyfit(x, np.log(y), 1)
        pred_log   = np.exp(np.polyval(coeffs_log, x))
        r2_log     = 1 - np.var(y - pred_log) / (np.var(y) + 1e-9)
        if r2_log > (r2_lin + 0.1) and r2_log > 0.5:
            result["needs_log"] = True

    sp = _periods_per_year(freq)
    if n >= sp * 2:
        try:
            acf = series.autocorr(lag=sp)
            if not np.isnan(acf) and abs(acf) > 0.25:
                result["has_seasonality"] = True
        except Exception:
            pass

    return result


# ── Walk-forward cross-validation ──────────────────────────────────────────────

def _walk_forward_cv(
    series: pd.Series,
    forecast_fn,
    n_folds: int,
    horizon: int,
    cv_val: float,
) -> Optional[Dict]:
    n         = len(series)
    min_train = n - (n_folds * horizon)
    if min_train < 4: return None

    fold_scores:    List[Dict]  = []
    coverage_hits:  List[bool]  = []

    for fold in range(n_folds):
        end_train = n - horizon * (n_folds - fold)
        train     = series.iloc[:end_train]
        actual    = series.iloc[end_train: end_train + horizon].values
        if len(actual) == 0: continue

        try:
            preds = forecast_fn(train, len(actual), cv_val)
            if preds is None or len(preds) < len(actual):
                continue

            p = preds[:len(actual)]
            fold_scores.append(_all_metrics(actual, p))

            resid_std = np.std(train.values - train.mean()) + 1e-9
            z         = 1.645
            lower     = p - z * resid_std
            upper     = p + z * resid_std
            hits      = np.mean((actual >= lower) & (actual <= upper))
            coverage_hits.append(hits)

        except Exception:
            continue

    if not fold_scores: return None

    result = {
        metric: round(float(np.mean([f[metric] for f in fold_scores])), 4)
        for metric in fold_scores[0]
    }
    result["empirical_coverage"] = round(float(np.mean(coverage_hits)) * 100, 1) if coverage_hits else None
    return result


# ── Individual model forecast functions ────────────────────────────────────────

def _forecast_linear(train, horizon, cv_val, freq="MS", needs_log=False):
    try:
        t  = np.log1p(train.values) if needs_log else train.values
        x  = np.arange(len(t))
        c  = np.polyfit(x, t, 1)
        p  = np.polyval(c, np.arange(len(t), len(t) + horizon))
        return np.expm1(p) if needs_log else p
    except Exception: return None

def _forecast_ets(train, horizon, cv_val, freq="MS", needs_log=False):
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        sp = _periods_per_year(freq)
        use_s  = len(train) >= sp * 2
        t_data = np.log1p(train) if needs_log else train
        try:
            model = ExponentialSmoothing(
                t_data, trend="add",
                seasonal="add" if use_s else None,
                seasonal_periods=sp if use_s else None,
                damped_trend=True, initialization_method="estimated"
            ).fit(optimized=True, use_brute=False)
        except Exception:
            model = ExponentialSmoothing(
                t_data, trend="add",
                seasonal="add" if use_s else None,
                seasonal_periods=sp if use_s else None,
                initialization_method="estimated"
            ).fit(optimized=True, use_brute=False)
        preds = np.asarray(model.forecast(horizon))
        return np.expm1(preds) if needs_log else preds
    except Exception: return None

def _forecast_sarima(train, horizon, cv_val, freq="MS", needs_log=False):
    try:
        import pmdarima as pm
        sp     = _periods_per_year(freq)
        use_s  = len(train) >= sp * 2
        t_data = np.log1p(train) if needs_log else train
        model  = pm.auto_arima(
            t_data, seasonal=use_s, m=sp if use_s else 1,
            stepwise=True, suppress_warnings=True, error_action="ignore",
            max_p=2, max_q=2, max_P=1, max_Q=1, max_d=1, max_D=1,
        )
        fc, _ = model.predict(n_periods=horizon, return_conf_int=True, alpha=0.10)
        return np.expm1(np.asarray(fc)) if needs_log else np.asarray(fc)
    except Exception: return None

def _forecast_prophet(train, horizon, cv_val, freq="MS", needs_log=False):
    try:
        from prophet import Prophet
        t_data = np.log1p(train) if needs_log else train
        df_p   = pd.DataFrame({"ds": train.index, "y": t_data.values}).dropna(subset=["y"])
        if len(df_p) < 5: return None
        sp     = _periods_per_year(freq)
        use_s  = len(train) >= sp
        s_mode = "multiplicative" if t_data.min() > 0 else "additive"
        cps    = 0.3 if cv_val > 0.5 else 0.15
        model  = Prophet(
            yearly_seasonality=use_s, weekly_seasonality=False,
            daily_seasonality=False, changepoint_prior_scale=cps,
            seasonality_mode=s_mode,
        )
        model.fit(df_p)
        future   = model.make_future_dataframe(periods=horizon, freq=freq)
        forecast = model.predict(future)
        preds    = np.asarray(forecast["yhat"].tail(horizon))
        return np.expm1(preds) if needs_log else preds
    except Exception: return None

def _forecast_theta(train, horizon, cv_val, freq="MS", needs_log=False):
    try:
        t_data = np.log1p(train.values.copy().astype(float)) if needs_log else train.values.copy().astype(float)
        n      = len(t_data)
        alpha, level, ses = 0.2, t_data[0], [t_data[0]]
        for v in t_data[1:]:
            level = alpha * v + (1 - alpha) * level
            ses.append(level)
        ses    = np.array(ses)
        coeffs = np.polyfit(np.arange(n), ses, 1)
        preds  = (np.full(horizon, ses[-1]) + np.polyval(coeffs, np.arange(n, n + horizon))) / 2
        return np.expm1(preds) if needs_log else preds
    except Exception: return None

def _forecast_naive_seasonal(train, horizon, cv_val, freq="MS", needs_log=False):
    try:
        sp = _periods_per_year(freq)
        if len(train) < sp: return np.full(horizon, train.iloc[-1])
        return np.array([train.iloc[-(sp - (h % sp))] for h in range(horizon)], dtype=float)
    except Exception: return None

_CV_FNS = {
    "ets":           _forecast_ets,
    "sarima":        _forecast_sarima,
    "prophet":       _forecast_prophet,
    "theta":         _forecast_theta,
    "linear":        _forecast_linear,
    "naive_seasonal":_forecast_naive_seasonal,
}


# ── Full model fit functions ───────────────────────────────────────────────────

def _fit_linear(sub, date_col, value_col, periods, freq, cv_val, needs_log, feature_cols=None):
    y      = np.log1p(sub[value_col].values) if needs_log else sub[value_col].values
    x      = np.arange(len(sub))
    coeffs = np.polyfit(x, y, 1)

    fitted_raw = np.polyval(coeffs, x)
    fitted     = np.expm1(fitted_raw) if needs_log else fitted_raw
    resid_std  = np.std(sub[value_col].values - fitted) + 1e-9

    hist_df = sub.rename(columns={value_col: "actual"}).copy()
    hist_df["predicted"]   = fitted.round(2)
    ci_hist                = _compute_ci(resid_std, len(sub))
    hist_df["lower_bound"] = (hist_df["predicted"] - ci_hist).round(2)
    hist_df["upper_bound"] = (hist_df["predicted"] + ci_hist).round(2)
    hist_df["is_forecast"] = False

    future_dates  = pd.date_range(start=sub[date_col].iloc[-1], periods=periods + 1, freq=freq)[1:]
    future_y_raw  = np.polyval(coeffs, np.arange(len(sub), len(sub) + periods))
    future_y      = np.expm1(future_y_raw) if needs_log else future_y_raw
    ci_fore       = _compute_ci(resid_std, periods)

    fore_df = pd.DataFrame({
        date_col:      future_dates.strftime("%Y-%m-%d"),
        "actual":      np.nan,
        "predicted":   future_y.round(2),
        "lower_bound": (future_y - ci_fore).round(2),
        "upper_bound": (future_y + ci_fore).round(2),
        "is_forecast": True,
    })
    hist_df[date_col] = hist_df[date_col].dt.strftime("%Y-%m-%d")
    return pd.concat([hist_df, fore_df], ignore_index=True), {"model_display": "Linear Trend", "seasonality_detected": False}, future_y

def _fit_ets(sub, date_col, value_col, periods, freq, cv_val, needs_log, feature_cols=None):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    sp     = _periods_per_year(freq)
    series = np.log1p(sub.set_index(date_col)[value_col]) if needs_log else sub.set_index(date_col)[value_col]
    use_s  = len(series) >= sp * 2

    try:
        model = ExponentialSmoothing(
            series, trend="add",
            seasonal="add" if use_s else None,
            seasonal_periods=sp if use_s else None,
            damped_trend=True, initialization_method="estimated",
        ).fit(optimized=True)
    except Exception:
        try:
            model = ExponentialSmoothing(
                series, trend="add",
                seasonal="add" if use_s else None,
                seasonal_periods=sp if use_s else None,
                initialization_method="estimated",
            ).fit(optimized=True)
        except Exception:
            model = ExponentialSmoothing(series, trend="add", initialization_method="heuristic").fit(optimized=True)

    fc_raw    = np.asarray(model.forecast(periods))
    fitted    = np.expm1(np.asarray(model.fittedvalues)) if needs_log else np.asarray(model.fittedvalues)
    fc        = np.expm1(fc_raw) if needs_log else fc_raw
    resid_std = np.std(sub[value_col].values - fitted) + 1e-9

    hist_df = sub.rename(columns={value_col: "actual"}).copy()
    hist_df["predicted"]   = fitted.round(2)
    ci_hist                = _compute_ci(resid_std, len(sub))
    hist_df["lower_bound"] = (hist_df["predicted"] - ci_hist).round(2)
    hist_df["upper_bound"] = (hist_df["predicted"] + ci_hist).round(2)
    hist_df["is_forecast"] = False

    future_dates = pd.date_range(start=sub[date_col].iloc[-1], periods=periods + 1, freq=freq)[1:]
    ci_fore      = _compute_ci(resid_std, periods)

    fore_df = pd.DataFrame({
        date_col:      future_dates.strftime("%Y-%m-%d"),
        "actual":      np.nan,
        "predicted":   np.round(fc, 2),
        "lower_bound": (fc - ci_fore).round(2),
        "upper_bound": (fc + ci_fore).round(2),
        "is_forecast": True,
    })
    hist_df[date_col] = hist_df[date_col].dt.strftime("%Y-%m-%d")
    return pd.concat([hist_df, fore_df], ignore_index=True), {"model_display": "Exponential Smoothing (ETS)", "seasonality_detected": use_s}, fc

def _fit_sarima(sub, date_col, value_col, periods, freq, cv_val, needs_log, feature_cols=None):
    import pmdarima as pm
    sp     = _periods_per_year(freq)
    series = np.log1p(sub.set_index(date_col)[value_col]) if needs_log else sub.set_index(date_col)[value_col]
    use_s  = len(series) >= sp * 2

    model          = pm.auto_arima(series, seasonal=use_s, m=sp if use_s else 1, stepwise=True,
                                   suppress_warnings=True, error_action="ignore",
                                   max_p=3, max_q=3, max_P=2, max_Q=2, max_d=2, max_D=1)
    fc_raw, _      = model.predict(n_periods=periods, return_conf_int=True, alpha=0.10)
    fitted         = np.expm1(np.asarray(model.predict_in_sample())) if needs_log else np.asarray(model.predict_in_sample())
    fc             = np.expm1(np.asarray(fc_raw)) if needs_log else np.asarray(fc_raw)
    resid_std      = np.std(sub[value_col].values - fitted) + 1e-9

    hist_df = sub.rename(columns={value_col: "actual"}).copy()
    hist_df["predicted"]   = np.round(fitted, 2)
    ci_hist                = _compute_ci(resid_std, len(sub))
    hist_df["lower_bound"] = (hist_df["predicted"] - ci_hist).round(2)
    hist_df["upper_bound"] = (hist_df["predicted"] + ci_hist).round(2)
    hist_df["is_forecast"] = False

    future_dates = pd.date_range(start=sub[date_col].iloc[-1], periods=periods + 1, freq=freq)[1:]
    ci_fore      = _compute_ci(resid_std, periods)

    fore_df = pd.DataFrame({
        date_col:      future_dates.strftime("%Y-%m-%d"),
        "actual":      np.nan,
        "predicted":   np.round(fc, 2),
        "lower_bound": (fc - ci_fore).round(2),
        "upper_bound": (fc + ci_fore).round(2),
        "is_forecast": True,
    })
    hist_df[date_col] = hist_df[date_col].dt.strftime("%Y-%m-%d")
    return pd.concat([hist_df, fore_df], ignore_index=True), {"model_display": "Auto-ARIMA", "seasonality_detected": use_s}, fc

def _fit_prophet(sub, date_col, value_col, periods, freq, cv_val, needs_log, feature_cols=None):
    from prophet import Prophet
    
    y_vals = np.log1p(sub[value_col]) if needs_log else sub[value_col]
    df_p   = pd.DataFrame({"ds": sub[date_col], "y": y_vals}).dropna(subset=["y"])
    
    if feature_cols:
        for f_col in feature_cols:
            if f_col in sub.columns:
                df_p[f_col] = sub[f_col].values
                
    s_mode = "multiplicative" if df_p["y"].min() > 0 else "additive"
    cps    = 0.3 if cv_val > 0.5 else 0.15
    use_s  = len(df_p) >= _periods_per_year(freq)

    model = Prophet(
        yearly_seasonality=use_s, weekly_seasonality=False, daily_seasonality=False,
        interval_width=0.90, changepoint_prior_scale=cps, seasonality_mode=s_mode,
    )
    if use_s: model.add_country_holidays(country_name="US")
    
    if feature_cols:
        for f_col in feature_cols:
            if f_col in df_p.columns:
                model.add_regressor(f_col)
                
    model.fit(df_p)

    future = model.make_future_dataframe(periods=periods, freq=freq)
    if feature_cols:
        for f_col in feature_cols:
            if f_col in df_p.columns:
                recent_avg = df_p[f_col].tail(3).mean()
                future[f_col] = list(df_p[f_col]) + [recent_avg] * periods
                
    forecast     = model.predict(future)
    fitted_raw   = np.asarray(forecast["yhat"].iloc[:len(sub)])
    fc_raw       = np.asarray(forecast["yhat"].tail(periods))
    fitted       = np.expm1(fitted_raw) if needs_log else fitted_raw
    fc           = np.expm1(fc_raw)    if needs_log else fc_raw
    resid_std    = np.std(sub[value_col].values - fitted) + 1e-9

    hist_df = sub.rename(columns={value_col: "actual"}).copy()
    hist_df["predicted"]   = np.round(fitted, 2)
    ci_hist                = _compute_ci(resid_std, len(sub))
    hist_df["lower_bound"] = (hist_df["predicted"] - ci_hist).round(2)
    hist_df["upper_bound"] = (hist_df["predicted"] + ci_hist).round(2)
    hist_df["is_forecast"] = False

    future_dates = pd.date_range(start=sub[date_col].iloc[-1], periods=periods + 1, freq=freq)[1:]
    ci_fore      = _compute_ci(resid_std, periods)

    fore_df = pd.DataFrame({
        date_col:      future_dates.strftime("%Y-%m-%d"),
        "actual":      np.nan,
        "predicted":   np.round(fc, 2),
        "lower_bound": (fc - ci_fore).round(2),
        "upper_bound": (fc + ci_fore).round(2),
        "is_forecast": True,
    })
    hist_df[date_col] = hist_df[date_col].dt.strftime("%Y-%m-%d")
    
    meta_desc = "Multivariate Prophet (with Regressors)" if feature_cols else "Prophet"
    return pd.concat([hist_df, fore_df], ignore_index=True), {"model_display": meta_desc, "seasonality_detected": use_s}, fc

def _fit_theta(sub, date_col, value_col, periods, freq, cv_val, needs_log, feature_cols=None):
    vals  = np.log1p(sub[value_col].values.astype(float)) if needs_log else sub[value_col].values.astype(float)
    n     = len(sub)
    alpha, level, ses = 0.2, vals[0], [vals[0]]
    for v in vals[1:]:
        level = alpha * v + (1 - alpha) * level
        ses.append(level)
    ses    = np.array(ses)
    coeffs = np.polyfit(np.arange(n), ses, 1)

    fitted_raw = (ses + np.polyval(coeffs, np.arange(n))) / 2
    fitted     = np.expm1(fitted_raw) if needs_log else fitted_raw
    resid_std  = np.std(sub[value_col].values - fitted) + 1e-9

    hist_df = sub.rename(columns={value_col: "actual"}).copy()
    hist_df["predicted"]   = fitted.round(2)
    ci_hist                = _compute_ci(resid_std, n)
    hist_df["lower_bound"] = (hist_df["predicted"] - ci_hist).round(2)
    hist_df["upper_bound"] = (hist_df["predicted"] + ci_hist).round(2)
    hist_df["is_forecast"] = False

    fc_raw       = (np.full(periods, ses[-1]) + np.polyval(coeffs, np.arange(n, n + periods))) / 2
    fc           = np.expm1(fc_raw) if needs_log else fc_raw
    ci_fore      = _compute_ci(resid_std, periods)
    future_dates = pd.date_range(start=sub[date_col].iloc[-1], periods=periods + 1, freq=freq)[1:]

    fore_df = pd.DataFrame({
        date_col:      future_dates.strftime("%Y-%m-%d"),
        "actual":      np.nan,
        "predicted":   fc.round(2),
        "lower_bound": (fc - ci_fore).round(2),
        "upper_bound": (fc + ci_fore).round(2),
        "is_forecast": True,
    })
    hist_df[date_col] = hist_df[date_col].dt.strftime("%Y-%m-%d")
    return pd.concat([hist_df, fore_df], ignore_index=True), {"model_display": "Theta Method", "seasonality_detected": False}, fc

def _fit_naive_seasonal(sub, date_col, value_col, periods, freq, cv_val, needs_log, feature_cols=None):
    sp, vals, n = _periods_per_year(freq), sub[value_col].values.astype(float), len(sub)
    fitted = np.full(n, np.nan)
    for i in range(sp, n): fitted[i] = vals[i - sp]
    fitted[:sp] = vals[:sp]
    resid_std   = np.nanstd(vals - fitted) + 1e-9

    hist_df = sub.rename(columns={value_col: "actual"}).copy()
    hist_df["predicted"]   = np.round(fitted, 2)
    ci_hist                = _compute_ci(resid_std, n)
    hist_df["lower_bound"] = (hist_df["predicted"] - ci_hist).round(2)
    hist_df["upper_bound"] = (hist_df["predicted"] + ci_hist).round(2)
    hist_df["is_forecast"] = False

    fc           = np.array([vals[-(sp - (h % sp))] for h in range(periods)])
    ci_fore      = _compute_ci(resid_std, periods)
    future_dates = pd.date_range(start=sub[date_col].iloc[-1], periods=periods + 1, freq=freq)[1:]

    fore_df = pd.DataFrame({
        date_col:      future_dates.strftime("%Y-%m-%d"),
        "actual":      np.nan,
        "predicted":   fc.round(2),
        "lower_bound": (fc - ci_fore).round(2),
        "upper_bound": (fc + ci_fore).round(2),
        "is_forecast": True,
    })
    hist_df[date_col] = hist_df[date_col].dt.strftime("%Y-%m-%d")
    return pd.concat([hist_df, fore_df], ignore_index=True), {"model_display": "Seasonal Naïve", "seasonality_detected": True}, fc

_FIT_FNS = {
    "ets":           _fit_ets,
    "sarima":        _fit_sarima,
    "prophet":       _fit_prophet,
    "theta":         _fit_theta,
    "linear":        _fit_linear,
    "naive_seasonal":_fit_naive_seasonal,
}


# ── ForecastEngine ─────────────────────────────────────────────────────────────

class ForecastEngine:

    def run(self, df, date_col, value_col, periods=6, group_col=None, feature_cols=None):
        try: periods = max(int(periods), 2)
        except Exception: periods = 6

        if date_col not in df.columns or value_col not in df.columns:
            return {"error": f"Columns '{date_col}' or '{value_col}' not found."}

        f_cols = feature_cols if feature_cols else []
        cols = [date_col, value_col] + ([group_col] if group_col and group_col in df.columns else []) + f_cols
        df   = df[cols].copy()
        df[date_col]  = pd.to_datetime(df[date_col],  errors="coerce")
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df            = df.dropna(subset=[date_col, value_col])

        freq     = _infer_freq(df[date_col])
        agg_func = _get_agg_func(value_col)
        groups   = df[group_col].unique() if group_col and group_col in df.columns else [None]

        all_cols = df.columns.tolist()
        cat_cols = [
            c for c in all_cols
            if c not in [date_col, value_col, group_col]
            and str(df[c].dtype) in ("object", "category")
        ]

        results  = {}

        for group in groups:
            sub = (
                df[df[group_col] == group].copy()
                if group is not None
                else df.copy()
            )
            
            agg_dict = {value_col: agg_func}
            for f in f_cols:
                if f in sub.columns:
                    agg_dict[f] = "mean"
                    
            sub      = sub.groupby(pd.Grouper(key=date_col, freq=freq)).agg(agg_dict).reset_index()
            full_idx = pd.date_range(sub[date_col].min(), sub[date_col].max(), freq=freq)
            sub      = sub.set_index(date_col).reindex(full_idx).rename_axis(date_col).reset_index()

            if agg_func == "sum":
                sub[value_col] = sub[value_col].fillna(0)
            else:
                sub[value_col] = sub[value_col].interpolate(method="linear").bfill().ffill()
                
            for f in f_cols:
                if f in sub.columns:
                    sub[f] = sub[f].interpolate(method="linear").bfill().ffill()

            n     = len(sub)
            label = str(group) if group is not None else "forecast"

            if n < 3:
                results[label] = {"error": f"Need at least 3 time periods (got {n})."}
                continue

            try:
                result_df, meta = self._run_for_group(sub, date_col, value_col, periods, freq, n, f_cols)

                if sub[value_col].min() >= 0:
                    for col in ["predicted", "lower_bound", "upper_bound"]:
                        if col in result_df.columns:
                            result_df[col] = result_df[col].clip(lower=0)

                result_df = self._add_rolling(result_df, "actual")
                actual_vals = result_df.loc[~result_df["is_forecast"], "actual"].dropna()

                if len(actual_vals) >= 2:
                    meta["cagr"]      = _cagr(actual_vals.iloc[0], actual_vals.iloc[-1], len(actual_vals), freq)
                    meta["avg_mom"]   = round(actual_vals.pct_change().mean() * 100, 2)
                    meta["direction"] = "upward" if actual_vals.iloc[-1] > actual_vals.iloc[0] else "downward"

                    hist_mean = float(actual_vals.mean())
                    mae_raw   = meta.get("winning_scores", {}).get("MAE", None)
                    if mae_raw is not None and hist_mean > 0:
                        meta["mae_pct_mean"] = round(mae_raw / hist_mean * 100, 1)
                    else:
                        meta["mae_pct_mean"] = None

                meta["granularity"] = _freq_label(freq)
                meta["periods"]     = periods

                if cat_cols:
                    meta["suggested_groupby"] = cat_cols[:3]
                    meta["groupby_note"] = (
                        f"Tip: Try grouping by '{cat_cols[0]}' to see segment-level forecasts."
                    )

                meta["narrative"]   = self._narrative(meta)
                results[label]      = {"data": result_df, "meta": meta}

            except Exception as e:
                results[label] = {"error": str(e)}

        if not results:
            return {"error": "Not enough data to forecast."}

        if len(results) == 1:
            val = list(results.values())[0]
            if "error" in val:
                return val
            return {
                "data":    val["data"],
                "meta":    val["meta"],
                "groups":  None,
                "periods": periods,
            }

        return {
            "data":    {k: v["data"] for k, v in results.items() if "data" in v},
            "meta":    {k: v.get("meta", {}) for k, v in results.items()},
            "groups":  list(results.keys()),
            "periods": periods,
        }

    def _run_for_group(self, sub, date_col, value_col, periods, freq, n, f_cols) -> Tuple[pd.DataFrame, Dict]:
        raw_vals       = sub[value_col].values
        primary_metric = _select_primary_metric(raw_vals)

        raw_series = sub.set_index(date_col)[value_col]
        profile    = _profile_series(raw_series, freq)
        eligible   = self._get_eligible_models(n, freq)

        if n < 24:
            short_preferred = ["theta", "linear", "ets", "naive_seasonal", "sarima", "prophet"]
            eligible = [m for m in short_preferred if m in eligible] + \
                       [m for m in eligible if m not in short_preferred]

        sp         = _periods_per_year(freq)
        min_train  = max(sp, n // 2)
        max_horizon = max(1, (n - min_train) // 2)
        cv_horizon  = min(periods, max_horizon)
        if cv_horizon == 0: cv_horizon = 1

        cv_folds = min(10, max(3, n // (sp * 2)))
        while cv_folds > 1 and (n - cv_folds * cv_horizon) < 4:
            cv_folds -= 1

        cv_results: Dict[str, Dict] = {}
        for model_name in eligible:
            cv_fn = _CV_FNS[model_name]
            def make_fn(fn, f=freq, cv_v=profile["cv"], nl=profile["needs_log"]):
                return lambda s, h, _cv: fn(s, h, _cv, f, nl)
            scores = _walk_forward_cv(
                raw_series,
                make_fn(cv_fn),
                n_folds=cv_folds,
                horizon=cv_horizon,
                cv_val=profile["cv"],
            )
            if scores is not None:
                cv_results[model_name] = scores

        all_coverage = [
            v["empirical_coverage"]
            for v in cv_results.values()
            if v.get("empirical_coverage") is not None
        ]
        avg_coverage  = float(np.mean(all_coverage)) if all_coverage else 90.0
        z_multiplier  = 1.645 if avg_coverage >= 75.0 else 2.0

        data_warning: Optional[str] = None
        if n < 18:
            data_warning = (
                f"⚠️ Only {n} time periods available — forecasts are less reliable below 18 periods. "
                "Consider collecting more history or switching to a quarterly view for stabler estimates."
            )

        if not cv_results:
            sub_smooth = sub.copy()
            sub_smooth[value_col] = _smooth_outliers(sub_smooth[value_col])

            fit_fn    = _FIT_FNS["ets"]
            result_df, meta, _ = fit_fn(sub_smooth, date_col, value_col, periods, freq, profile["cv"], profile["needs_log"], feature_cols=f_cols)

            actuals = sub[value_col].values
            fitted  = result_df.loc[~result_df["is_forecast"], "predicted"].values
            best_scores = _all_metrics(actuals, fitted) if len(actuals) == len(fitted) else {"MAE": 0, "RMSE": 0, "SMAPE": 0}

            meta["model"]              = "ets"
            meta["winner"]             = meta["model"]
            meta["primary_metric"]     = primary_metric
            meta["cv_folds"]           = "In-Sample (Data too short for CV)"
            meta["winning_scores"]     = {k: round(v, 4) for k, v in best_scores.items()}
            meta["empirical_coverage"] = None
            meta["series_profile"]     = profile
            if data_warning:
                meta["data_warning"]   = data_warning
            return result_df, meta

        sorted_models = sorted(
            cv_results.keys(),
            key=lambda m: cv_results[m].get(primary_metric, float("inf")),
        )
        best_model  = sorted_models[0]
        best_scores = cv_results[best_model]

        is_ensemble = False
        w1 = w2 = total_w = None

        if len(sorted_models) > 1:
            runner_up  = sorted_models[1]
            score_best = cv_results[best_model].get(primary_metric, 0)
            score_ru   = cv_results[runner_up].get(primary_metric, 0)
            score_diff = abs(score_best - score_ru)

            relative_threshold = max(score_best * 0.05, 0.5)
            if score_diff < relative_threshold:
                is_ensemble = True
                w1      = 1.0 / (score_best + 1e-9)
                w2      = 1.0 / (score_ru   + 1e-9)
                total_w = w1 + w2

        sub_smooth = sub.copy()
        sub_smooth[value_col] = _smooth_outliers(sub_smooth[value_col])

        if is_ensemble:
            m1_name = sorted_models[0]
            m2_name = sorted_models[1]

            df1, m1, fc1 = _FIT_FNS[m1_name](sub_smooth, date_col, value_col, periods, freq, profile["cv"], profile["needs_log"], feature_cols=f_cols)
            df2, m2, fc2 = _FIT_FNS[m2_name](sub_smooth, date_col, value_col, periods, freq, profile["cv"], profile["needs_log"], feature_cols=f_cols)

            result_df     = df1.copy()
            blended_preds = (fc1 * (w1 / total_w)) + (fc2 * (w2 / total_w))

            mask = result_df["is_forecast"] == True

            hist_fitted1 = df1.loc[~mask, "predicted"].values
            hist_fitted2 = df2.loc[~mask, "predicted"].values
            blended_hist = (hist_fitted1 * (w1 / total_w)) + (hist_fitted2 * (w2 / total_w))

            result_df.loc[mask,  "predicted"] = blended_preds.round(2)
            result_df.loc[~mask, "predicted"] = blended_hist.round(2)

            blended_resid_std = np.std(sub[value_col].values - blended_hist) + 1e-9
            ci_fore = _compute_ci(blended_resid_std, periods, z=z_multiplier)

            result_df.loc[mask, "lower_bound"] = (blended_preds - ci_fore).round(2)
            result_df.loc[mask, "upper_bound"] = (blended_preds + ci_fore).round(2)

            ensemble_cv_scores: List[Dict] = []
            for fold in range(cv_folds):
                end_train = n - cv_horizon * (cv_folds - fold)
                actual_f  = raw_series.iloc[end_train: end_train + cv_horizon].values
                if len(actual_f) == 0: continue
                try:
                    p1 = _FIT_FNS[m1_name](
                        sub.iloc[:end_train].copy(), date_col, value_col,
                        cv_horizon, freq, profile["cv"], profile["needs_log"]
                    )[2]
                    p2 = _FIT_FNS[m2_name](
                        sub.iloc[:end_train].copy(), date_col, value_col,
                        cv_horizon, freq, profile["cv"], profile["needs_log"]
                    )[2]
                    p_blend = (p1 * (w1 / total_w)) + (p2 * (w2 / total_w))
                    ensemble_cv_scores.append(_all_metrics(actual_f, p_blend[:len(actual_f)]))
                except Exception:
                    continue

            if ensemble_cv_scores:
                best_scores = {
                    metric: round(float(np.mean([f[metric] for f in ensemble_cv_scores])), 4)
                    for metric in ensemble_cv_scores[0]
                }

            meta = {
                "model_display":        f"Ensemble ({m1['model_display']} & {m2['model_display']})",
                "seasonality_detected": m1.get("seasonality_detected", False) or m2.get("seasonality_detected", False),
            }

        else:
            fit_fn = _FIT_FNS.get(best_model, _fit_ets)
            result_df, meta, _ = fit_fn(sub_smooth, date_col, value_col, periods, freq, profile["cv"], profile["needs_log"], feature_cols=f_cols)

            if z_multiplier > 1.645:
                mask      = result_df["is_forecast"] == True
                pred      = result_df.loc[mask, "predicted"].values
                hist_vals = result_df.loc[~mask, "predicted"].values
                r_std     = np.std(sub[value_col].values - hist_vals) + 1e-9
                ci_fore   = _compute_ci(r_std, periods, z=z_multiplier)
                result_df.loc[mask, "lower_bound"] = (pred - ci_fore).round(2)
                result_df.loc[mask, "upper_bound"] = (pred + ci_fore).round(2)

        meta["model"]              = "Ensemble" if is_ensemble else best_model
        meta["winner"]             = meta["model"]
        meta["primary_metric"]     = primary_metric
        meta["cv_folds"]           = cv_folds
        meta["empirical_coverage"] = round(avg_coverage, 1)
        meta["z_multiplier"]       = round(z_multiplier, 3)
        meta["series_profile"]     = profile

        if best_scores:
            meta["winning_scores"] = {k: round(v, 4) for k, v in best_scores.items()}

        if data_warning:
            meta["data_warning"] = data_warning

        return result_df, meta

    def _get_eligible_models(self, n: int, freq: str) -> list:
        eligible = ["linear"]
        if n >= 6:  eligible.extend(["theta", "ets"])
        if n >= _periods_per_year(freq): eligible.append("naive_seasonal")
        if n >= 8:
            try: import pmdarima; eligible.append("sarima")
            except ImportError: pass
        if n >= 10:
            try: import prophet; eligible.append("prophet")
            except ImportError: pass
        return eligible

    def _add_rolling(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        c = df[col]
        df["rolling_3"] = c.rolling(3, min_periods=1).mean().round(2)
        df["rolling_6"] = c.rolling(6, min_periods=1).mean().round(2)
        df["mom_pct"]   = c.pct_change(1).mul(100).round(2)
        return df

    def _narrative(self, meta: Dict) -> str:
        display = meta.get("model_display", "Forecast")
        metric  = meta.get("primary_metric", "SMAPE")
        scores  = meta.get("winning_scores", {})
        folds   = meta.get("cv_folds", 3)
        cov     = meta.get("empirical_coverage")
        z_mult  = meta.get("z_multiplier", 1.645)
        profile = meta.get("series_profile", {})
        parts   = []

        if meta.get("data_warning"):
            parts.append(meta["data_warning"])

        if metric == "MAE":
            reason = "the data ranges from 0-100 or contains zeros, making percentage error metrics mathematically unsafe"
        elif metric == "RMSE":
            reason = "high volatility was detected, and RMSE heavily penalizes large outlier misses"
        else:
            reason = "it is the standard balanced metric for continuous business data"

        if scores and metric in scores:
            val  = scores[metric]
            unit = "%" if metric in ("SMAPE", "MAPE") else ""
            parts.append(
                f"🏆 **{display}** was automatically selected as the most accurate model. "
                f"We evaluated the tournament using **{metric}** because {reason}."
            )
            mae_raw   = scores.get("MAE", 0)
            rmse_val  = scores.get("RMSE", 0)
            smape_val = scores.get("SMAPE", 0)
            mae_str   = f"MAE: {mae_raw:,.2f}"
            rmse_str  = f"RMSE: {rmse_val:,.2f}"
            smape_str = f"SMAPE: {smape_val:.2f}%"
            fold_str  = folds if isinstance(folds, str) else f"{folds}-fold walk-forward cross-validation"
            parts.append(
                f"Validated via {fold_str}, the winning error scores were: "
                f"**{metric}: {val:,.2f}{unit}** ({mae_str} | {rmse_str} | {smape_str})."
            )
            mae_pct = meta.get("mae_pct_mean")
            if mae_pct is not None:
                quality = (
                    "excellent" if mae_pct < 10 else
                    "good"      if mae_pct < 20 else
                    "moderate"  if mae_pct < 35 else
                    "poor"
                )
                parts.append(
                    f"In plain terms, the model's average error is **{mae_pct:.1f}% of the historical mean** "
                    f"({quality} forecast accuracy)."
                )
        else:
            parts.append(f"🏆 **{display}** was selected using **{metric}** (chosen because {reason}).")

        if cov is not None:
            ci_pct = 95 if z_mult >= 2.0 else 90
            parts.append(
                f"Empirical CI coverage across CV folds: **{cov:.1f}%** "
                f"(band auto-calibrated to {ci_pct}% target)."
            )

        cv_val = profile.get("cv", 0)
        if cv_val > 0.25:
            parts.append(
                f"⚠️ High series volatility detected (CV={cv_val:.2f}). "
                "Point forecasts carry significant uncertainty — use the confidence band range, not just the centre line."
            )

        if meta.get("seasonality_detected"):
            parts.append("Seasonal patterns and/or holiday events were mathematically detected and incorporated.")
        if meta.get("direction") and meta.get("avg_mom") is not None:
            parts.append(f"Historical values trended {meta['direction']} at {meta['avg_mom']:+.1f}% avg MoM change.")

        groupby_note = meta.get("groupby_note")
        if groupby_note:
            parts.append(f"💡 {groupby_note}")

        parts.append("Shaded band represents the confidence interval.")
        return " ".join(parts)


# ── Trend Engine ───────────────────────────────────────────────────────────────

class TrendEngine:
    def run(self, df, date_col, value_col, group_col=None, window=3):
        if date_col not in df.columns or value_col not in df.columns:
            return {"error": f"Columns '{date_col}' or '{value_col}' not found."}
        df             = df.copy()
        df[date_col]   = pd.to_datetime(df[date_col],  errors="coerce")
        df[value_col]  = pd.to_numeric(df[value_col], errors="coerce")
        df             = df.dropna(subset=[date_col, value_col])
        agg_func       = _get_agg_func(value_col)

        def _compute(sub):
            freq = _infer_freq(sub[date_col])
            sub  = (
                sub.groupby(pd.Grouper(key=date_col, freq=freq))[value_col]
                   .agg(agg_func).reset_index().dropna().sort_values(date_col)
            )
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
                    res = STL(sub.set_index(date_col)[value_col], period=12, robust=True).fit()
                    sub["trend_component"]    = res.trend.values
                    sub["seasonal_component"] = res.seasonal.values
                    sub["residual_component"] = res.resid.values
                    stl_meta = {
                        "trend_strength":      max(0, round(1 - np.var(res.resid) / (np.var(res.trend + res.resid) + 1e-9), 3)),
                        "seasonality_detected": True,
                    }
                except Exception:
                    stl_meta = {"trend_strength": None, "seasonality_detected": False}
            elif n >= 2:
                slope, intercept = np.polyfit(np.arange(n), vals.fillna(vals.mean()), 1)
                sub["trend_component"] = np.polyval([slope, intercept], np.arange(n)).round(2)
                stl_meta = {
                    "trend_strength":      round(max(0, 1 - np.var(vals - sub["trend_component"]) / (np.var(vals) + 1e-9)), 3),
                    "seasonality_detected": False,
                }

            rm  = vals.rolling(6, min_periods=2).mean()
            rs  = vals.rolling(6, min_periods=2).std().replace(0, np.nan)
            sub["spike_zscore"] = ((vals - rm) / rs).round(3)
            sub["is_spike"]     = ((vals - rm) / rs).abs() > 2.0
            direction           = "upward" if vals.iloc[-1] > vals.iloc[0] else "downward"
            sub[date_col]       = sub[date_col].dt.strftime("%Y-%m-%d")

            meta = {
                **stl_meta,
                "cagr":         _cagr(vals.iloc[0], vals.iloc[-1], n, freq),
                "avg_mom_pct":  round(sub["mom_pct"].mean(), 2) if n > 1 else None,
                "direction":    direction,
                "granularity":  _freq_label(freq),
                "periods":      n,
                "spike_count":  int(sub["is_spike"].sum()),
                "narrative":    self._narrative(
                    direction,
                    round(sub["mom_pct"].mean(), 2) if n > 1 else None,
                    _cagr(vals.iloc[0], vals.iloc[-1], n, freq),
                    stl_meta.get("trend_strength"),
                    n,
                ),
            }
            return sub, meta

        if group_col and group_col in df.columns:
            parts, metas = [], {}
            for grp, gdf in df.groupby(group_col):
                r, m = _compute(gdf)
                r[group_col]  = grp
                metas[str(grp)] = m
                parts.append(r)
            return {"data": pd.concat(parts, ignore_index=True), "window": window, "meta": metas}

        result_df, meta = _compute(df)
        return {"data": result_df, "window": window, "meta": meta}

    def _narrative(self, direction, avg_mom, cagr, trend_strength, n) -> str:
        parts = []
        if avg_mom is not None:
            parts.append(
                f"Values have {'consistently' if (trend_strength or 0) > 0.6 else 'moderately'} "
                f"trended {direction} by an average of {abs(avg_mom):.1f}% per period over {n} periods."
            )
        if cagr is not None:
            parts.append(f"CAGR: {cagr*100:.1f}%.")
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
        if missing:
            return {"error": f"Columns not found: {missing}"}

        sub = df[[target_col] + feature_cols].copy().dropna()
        if len(sub) < 20:
            return {"error": "Need at least 20 rows for regression."}

        for col in feature_cols:
            if sub[col].dtype == object:
                sub[col] = LabelEncoder().fit_transform(sub[col].astype(str))
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
        sub = sub.dropna()

        X, y = sub[feature_cols].values, sub[target_col].values
        if len(sub) > 5000:
            idx  = np.random.choice(len(sub), 5000, replace=False)
            X, y = X[idx], y[idx]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42).fit(X_train, y_train)
        preds = model.predict(X_test)

        fi_df = pd.DataFrame(
            sorted(zip(feature_cols, model.feature_importances_), key=lambda x: x[1], reverse=True),
            columns=["feature", "importance"],
        )
        fi_df["importance"] = fi_df["importance"].round(4)

        comparison = pd.DataFrame({
            "actual":    y_test.round(2),
            "predicted": preds.round(2),
            "residual":  (y_test - preds).round(2),
            "pct_error": np.abs((y_test - preds) / (np.abs(y_test) + 1e-9) * 100).round(1),
        })
        return {
            "data":               comparison,
            "metrics":            {
                "R2":      round(r2_score(y_test, preds), 4),
                "MAE":     round(mean_absolute_error(y_test, preds), 4),
                "MAPE%":   round(float(np.mean(np.abs((y_test - preds) / (np.abs(y_test) + 1e-9)) * 100)), 2),
                "samples": len(sub),
            },
            "feature_importance": fi_df,
            "target":             target_col,
        }


# ── Anomaly Engine ─────────────────────────────────────────────────────────────

class AnomalyEngine:
    def run(self, df, value_col, date_col=None, group_col=None,
            z_threshold=2.5, method="auto", contamination="auto", **kwargs):

        if value_col not in df.columns:
            return {"error": f"Column '{value_col}' not found."}

        sub             = df.copy()
        sub[value_col]  = pd.to_numeric(sub[value_col], errors="coerce")
        sub             = sub.dropna(subset=[value_col])

        unique_dates = 0
        if date_col and date_col in sub.columns:
            sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce")
            unique_dates  = sub[date_col].nunique()

        method = "isolation_forest" if (method == "auto" and unique_dates < 10) else "rolling_z" if method == "auto" else method

        if contamination == "auto":
            q1, q3 = np.percentile(sub[value_col].dropna(), [25, 75])
            iqr    = q3 - q1
            fence_lo, fence_hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out  = ((sub[value_col] < fence_lo) | (sub[value_col] > fence_hi)).sum()
            contamination = float(np.clip(n_out / len(sub), 0.01, 0.15))

        if method == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            iso    = IsolationForest(contamination=contamination, random_state=42)
            preds  = iso.fit_predict(sub[[value_col]])
            scores = iso.decision_function(sub[[value_col]])
            sub["anomaly_score"] = -scores
            sub["anomaly_score"] = (
                (sub["anomaly_score"] - sub["anomaly_score"].min())
                / (sub["anomaly_score"].max() - sub["anomaly_score"].min() + 1e-9)
                * 5
            ).round(3)
            sub["is_anomaly"] = preds == -1
            sub["anomaly_type"] = "point"

            if date_col in sub.columns:
                sub = sub.sort_values(date_col)
                sub[date_col] = sub[date_col].dt.strftime("%Y-%m-%d")

        else:
            freq     = _infer_freq(sub[date_col])
            agg_func = _get_agg_func(value_col)
            sub      = (
                sub.groupby(pd.Grouper(key=date_col, freq=freq))[value_col]
                   .agg(agg_func).reset_index().dropna().sort_values(date_col)
            )

            sp              = _periods_per_year(freq)
            rolling_window  = max(5, sp)
            sub             = self._rolling_zscore(sub, value_col, z_threshold, window=rolling_window)
            sub["anomaly_type"] = "point"

            if len(sub) >= sp * 2:
                sub = self._contextual_anomaly(sub, value_col, date_col, sp, z_threshold)

            sub[date_col] = sub[date_col].dt.strftime("%Y-%m-%d")

        if len(sub) == 0:
            return {"error": "No valid data for anomaly detection."}

        sub["severity"] = sub["anomaly_score"].apply(
            lambda s: "Critical" if s > 4 else "High" if s > 3 else "Medium" if s > 2 else "Low"
        )
        anomalies = sub[sub["is_anomaly"]].copy()

        return {
            "data":          sub,
            "anomalies":     anomalies,
            "anomaly_count": int(len(anomalies)),
            "mean":          round(sub[value_col].mean(), 2),
            "std":           round(sub[value_col].std(), 2),
            "method":        "Isolation Forest" if method == "isolation_forest" else "Rolling Z-Score + STL",
            "contamination": round(contamination, 4),
            "meta": {
                "high_severity": int(anomalies["severity"].isin(["High", "Critical"]).sum()),
                "contextual":    int((anomalies.get("anomaly_type", pd.Series()) == "contextual").sum()),
                "narrative":     (
                    f"{len(anomalies)} anomalies detected "
                    f"({'Isolation Forest' if method == 'isolation_forest' else 'Rolling Z-Score + STL contextual'}). "
                    f"Estimated contamination: {contamination*100:.1f}%."
                ),
            },
        }

    def _rolling_zscore(self, df, value_col, threshold, window=5):
        vals = df[value_col]
        rm   = vals.rolling(window, min_periods=2).mean().fillna(vals.mean())
        rs   = (
            vals.rolling(window, min_periods=2).std()
                .fillna(vals.std()).replace(0, np.nan)
                .fillna(vals.std() + 1e-9)
        )
        df["anomaly_score"] = ((vals - rm) / rs).abs().round(3)
        df["is_anomaly"]    = df["anomaly_score"] > threshold
        return df

    def _contextual_anomaly(self, df, value_col, date_col, sp, z_threshold):
        try:
            from statsmodels.tsa.seasonal import STL
            series   = df.set_index(date_col)[value_col]
            stl      = STL(series, period=sp, robust=True).fit()
            resid    = pd.Series(stl.resid, index=series.index)
            r_std    = resid.std() + 1e-9
            ctx_z    = (resid.abs() / r_std)

            ctx_mask = ctx_z > z_threshold
            for idx in df.index:
                ts = df.loc[idx, date_col] if not isinstance(df.index, pd.DatetimeIndex) else idx
                if ts in ctx_z.index and ctx_mask.loc[ts]:
                    current_score = df.loc[idx, "anomaly_score"]
                    ctx_score     = min(5.0, float(ctx_z.loc[ts]))
                    df.loc[idx, "anomaly_score"] = max(current_score, ctx_score)
                    if not df.loc[idx, "is_anomaly"]:
                        df.loc[idx, "is_anomaly"]   = True
                        df.loc[idx, "anomaly_type"] = "contextual"
        except Exception:
            pass
        return df


# ── Cluster Engine (V20 - HDBSCAN + UMAP) ──────────────────────────────────────

class ClusterEngine:
    """
    Advanced non-linear clustering:
    - HDBSCAN for density-based grouping and native noise identification.
    - UMAP for 2D topological mapping.
    - Fallbacks to KMeans / PCA if libraries are missing or data is constrained.
    """

    def __init__(self):
        self._groq_client = None

    def _get_groq_client(self):
        if self._groq_client is not None:
            return self._groq_client
        try:
            import os
            from groq import Groq
            key = os.getenv("GROQ_API_KEY")
            if not key: return None
            self._groq_client = Groq(api_key=key)
            return self._groq_client
        except Exception:
            return None

    def _generate_labels(self, profiles: dict) -> dict:
        import json
        client = self._get_groq_client()
        if client is None: return {}

        prompt = (
            "You are an expert business strategist and data analyst.\n"
            "I have segmented my data into distinct clusters. Here are the average feature values:\n"
            f"{json.dumps(profiles, indent=2)}\n\n"
            "Give each cluster a highly descriptive, 2-3 word 'Persona Name' based on these averages.\n"
            "(e.g., 'High-Value Savers', 'At-Risk Churners', 'Premium Winners').\n\n"
            "Return ONLY a valid JSON object where keys are the exact original cluster names "
            "(e.g., 'Cluster 0') and values are the 2-3 word persona strings. "
            "No markdown, no explanations."
        )
        try:
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content.strip())
        except Exception as e:
            print(f"[ClusterEngine] Auto-labeling failed: {e}")
            return {}

    def run(self, df, feature_cols):
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import silhouette_score

        # Attempt to load advanced topological libraries
        try:
            from sklearn.cluster import HDBSCAN
            use_hdbscan = True
        except ImportError:
            try:
                from hdbscan import HDBSCAN
                use_hdbscan = True
            except ImportError:
                from sklearn.cluster import KMeans
                use_hdbscan = False

        try:
            from umap import UMAP
            use_umap = True
        except ImportError:
            from sklearn.decomposition import PCA
            use_umap = False

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            return {"error": f"Columns not found: {missing}"}

        sub = df[feature_cols].copy().dropna()
        if len(sub) < 15:
            return {"error": "Need at least 15 rows for clustering."}

        encoded_df   = pd.get_dummies(sub, drop_first=True)
        encoded_cols = encoded_df.columns.tolist()
        if len(encoded_cols) < 2:
            return {"error": "Clustering requires at least 2 distinct features."}

        scaler      = StandardScaler()
        scaled_data = scaler.fit_transform(encoded_df)

        # ── 1. HDBSCAN Core Clustering ──
        if use_hdbscan:
            # HDBSCAN auto-tunes k based on density. min_cluster_size dictates group strictness.
            min_sz = max(5, len(scaled_data) // 15)
            hdb    = HDBSCAN(min_cluster_size=min_sz)
            labels = hdb.fit_predict(scaled_data)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            # If HDBSCAN decides everything is just 1 giant blob or everything is noise, fallback to KMeans to force a split
            if n_clusters < 2:
                use_hdbscan = False

        # ── Fallback: KMeans + Isolation Forest ──
        if not use_hdbscan:
            from sklearn.ensemble import IsolationForest
            from sklearn.cluster import KMeans
            iso = IsolationForest(contamination=0.05, random_state=42)
            is_inlier = iso.fit_predict(scaled_data) == 1
            if is_inlier.sum() < 10:
                is_inlier = np.ones(len(sub), dtype=bool)

            scaled_normal = scaled_data[is_inlier]

            max_k = min(12, max(2, len(scaled_normal) // 15))
            best_k, best_score = 2, -1
            for k in range(2, max_k + 1):
                if k >= len(scaled_normal): break
                km   = KMeans(n_clusters=k, random_state=42, n_init="auto")
                lbls = km.fit_predict(scaled_normal)
                if len(np.unique(lbls)) < 2: continue
                score = silhouette_score(scaled_normal, lbls)
                if score > best_score:
                    best_k, best_score = k, score

            km = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
            normal_labels = km.fit_predict(scaled_normal)

            labels = np.full(len(sub), -1)
            labels[is_inlier] = normal_labels
            n_clusters = best_k

        # ── Labeling & Outlier tracking ──
        sub["Cluster"]  = [f"Cluster {x}" if x != -1 else "Outlier (Noise)" for x in labels]
        is_inlier_final = labels != -1
        n_outliers      = int((~is_inlier_final).sum())

        # Generate a silhouette score purely for the UI KPI metric (excluding noise)
        sil_score = -1.0
        if is_inlier_final.sum() > 2 and n_clusters >= 2:
            try:
                sil_score = silhouette_score(scaled_data[is_inlier_final], labels[is_inlier_final])
            except Exception:
                pass

        # ── Explainable AI (Random Forest) ──
        fi_df = None
        top_driver = "Unknown"
        if is_inlier_final.sum() > 10 and n_clusters >= 2:
            rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            rf.fit(scaled_data[is_inlier_final], labels[is_inlier_final])
            fi_df = pd.DataFrame({
                "feature":    encoded_cols,
                "importance": rf.feature_importances_.round(4),
            }).sort_values(by="importance", ascending=False).head(10)
            top_driver = fi_df.iloc[0]["feature"]

        # ── 2. UMAP Topological Projection ──
        if use_umap:
            # Compresses multidimensional data to 2D while preserving neighborhood relationships
            reducer = UMAP(n_components=2, random_state=42)
            pca_coords = reducer.fit_transform(scaled_data)
            algo_name = "UMAP"
        else:
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            pca_coords = reducer.fit_transform(scaled_data)
            algo_name = "PCA"

        # UI Chart builder expects "PCA_X" and "PCA_Y" strictly
        sub["PCA_X"] = pca_coords[:, 0].round(3)
        sub["PCA_Y"] = pca_coords[:, 1].round(3)

        # ── LLM Profiling ──
        valid_clusters = sub[sub["Cluster"] != "Outlier (Noise)"]
        numeric_orig   = valid_clusters.select_dtypes(include=np.number).drop(columns=["PCA_X", "PCA_Y"], errors="ignore")
        profiles = {}
        if not numeric_orig.empty:
            profiles = valid_clusters.groupby("Cluster")[numeric_orig.columns].mean().round(2).to_dict("index")

        new_labels = self._generate_labels(profiles)
        if new_labels:
            sub["Cluster"] = sub["Cluster"].map(lambda x: new_labels.get(x, x))
            profiles       = {new_labels.get(k, k): v for k, v in profiles.items()}

        clust_algo = "HDBSCAN" if use_hdbscan else "K-Means"
        narrative  = (
            f"Segmented into {n_clusters} distinct personas using {clust_algo} and {algo_name}. "
            f"{n_outliers} noise/outliers were natively isolated. "
            f"#1 driving factor: '{top_driver}' (standardized). "
        )
        for persona_name, prof in profiles.items():
            top_feat  = ", ".join([f"{k}={v}" for k, v in list(prof.items())[:3]])
            narrative += f"**{persona_name}**: {top_feat}. "

        meta = {
            "k":             n_clusters,
            "silhouette":    round(sil_score, 4),
            "features":      feature_cols,
            "top_driver":    top_driver,
            "profiles":      profiles,
            "narrative":     narrative.strip(),
            "outlier_count": n_outliers,
            "algorithm":     clust_algo,
            "reduction":     algo_name,
        }
        
        if not use_umap:
            meta["pca_variance"] = [round(v, 4) for v in reducer.explained_variance_ratio_.tolist()]
        else:
            meta["pca_variance"] = [0.0, 0.0]

        return {"data": sub, "meta": meta, "feature_importance": fi_df}