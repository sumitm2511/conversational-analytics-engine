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

import re
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


# ── Shared Classifier Tournament ────────────────────────────────────────────────
# Same philosophy as ForecastEngine's model tournament above: try several
# candidate algorithms, score them honestly via cross-validation, pick a
# winner, and report the full leaderboard rather than silently committing to
# one hardcoded model. Used by both ClusterEngine (to pick the model that
# powers predict_point on discovered clusters) and SupervisedClassifierEngine
# (to pick the model that predicts a real label column).

def _run_classifier_tournament(X, y, min_folds: int = 2, max_folds: int = 5, random_state: int = 42) -> Dict:
    """
    Compares Random Forest, Gradient Boosting, Extra Trees, Logistic Regression,
    and K-Nearest Neighbors via stratified k-fold CV, selecting the winner by
    weighted F1 (robust to class imbalance) with accuracy and log-loss reported
    alongside for transparency. Log-loss matters specifically here because these
    models' predict_proba output is what gets shown to users as "confidence %" —
    a poorly-calibrated model can report high confidence while being wrong far
    more often than that number implies.

    Returns:
      {
        "winner":         <model name str>,
        "model":          <fitted sklearn estimator, refit on ALL of X/y>,
        "leaderboard":    [ {"model": name, "f1_weighted": .., "accuracy": .., "log_loss": .., "selected": "✓"/""} , ... ],
        "cv_folds":       int or None,
        "skipped_reason": str or None,   # set when CV wasn't possible (too few samples per class)
      }
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import StratifiedKFold, cross_validate

    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    min_class_count = int(counts.min()) if len(counts) else 0

    candidates = {
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=8, random_state=random_state, class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=random_state
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=150, max_depth=8, random_state=random_state, class_weight="balanced"
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=random_state
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=min(15, max(3, min_class_count))
        ),
    }

    # Not enough samples in the smallest class to cross-validate at all —
    # mirrors ForecastEngine's short-series "In-Sample" fallback: don't error,
    # just fall back to a single safe default and say so honestly.
    if min_class_count < 2 or len(classes) < 2:
        model = candidates["Random Forest"]
        model.fit(X, y)
        return {
            "winner": "Random Forest",
            "model": model,
            "leaderboard": [],
            "cv_folds": None,
            "skipped_reason": (
                f"Only {min_class_count} sample(s) in the smallest class — need at least 2 "
                "for cross-validated model comparison, so a single default model was used instead."
            ),
        }

    n_folds = max(min_folds, min(max_folds, min_class_count))
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    leaderboard = []
    fitted_candidates = {}

    for name, model in candidates.items():
        try:
            scores = cross_validate(
                model, X, y, cv=cv,
                scoring=["f1_weighted", "accuracy", "neg_log_loss"],
                error_score=np.nan,
            )
            f1  = float(np.nanmean(scores["test_f1_weighted"]))
            acc = float(np.nanmean(scores["test_accuracy"]))
            ll_raw = np.nanmean(scores["test_neg_log_loss"])
            ll  = float(-ll_raw) if not np.isnan(ll_raw) else None
            if np.isnan(f1):
                continue
            leaderboard.append({
                "model": name,
                "f1_weighted": round(f1, 4),
                "accuracy": round(acc, 4),
                "log_loss": round(ll, 4) if ll is not None else None,
            })
            fitted_candidates[name] = model
        except Exception:
            continue

    if not leaderboard:
        model = candidates["Random Forest"]
        model.fit(X, y)
        return {
            "winner": "Random Forest",
            "model": model,
            "leaderboard": [],
            "cv_folds": n_folds,
            "skipped_reason": "All tournament candidates failed cross-validation; used Random Forest as a safe default.",
        }

    leaderboard.sort(key=lambda r: r["f1_weighted"], reverse=True)
    winner_name  = leaderboard[0]["model"]
    winner_model = fitted_candidates[winner_name]
    winner_model.fit(X, y)   # refit the winner on ALL provided data for production use

    for row in leaderboard:
        row["selected"] = "✓" if row["model"] == winner_name else ""

    return {
        "winner": winner_name,
        "model": winner_model,
        "leaderboard": leaderboard,
        "cv_folds": n_folds,
        "skipped_reason": None,
    }


# ── Shared SHAP Explainability ──────────────────────────────────────────────────
# Per-prediction explanations for predict_point() on both ClusterEngine and
# SupervisedClassifierEngine. Only fast, EXACT SHAP explainers are used live —
# TreeExplainer for tree models (with a known exception below) and
# LinearExplainer for Logistic Regression. A model-agnostic fallback exists in
# the shap library but was measured at 2-7 seconds for a single prediction
# during development — too slow for a live chat response — so it is
# deliberately NOT used here. When no fast explainer is available (multi-class
# Gradient Boosting — a known SHAP TreeExplainer limitation — or KNN, which has
# no fast native explainer at all), predict_point() falls back to the model's
# GLOBAL feature importance instead, clearly labeled as such rather than
# presented as an exact per-point explanation.

def _build_shap_explainer(clf, background_sample) -> Tuple[Optional[object], Optional[str]]:
    try:
        import shap
    except ImportError:
        return None, None

    model_type = type(clf).__name__
    n_classes = len(clf.classes_) if hasattr(clf, "classes_") else 2

    try:
        if model_type in ("RandomForestClassifier", "ExtraTreesClassifier"):
            return shap.TreeExplainer(clf), "tree"
        if model_type == "GradientBoostingClassifier" and n_classes == 2:
            # TreeExplainer only supports GradientBoostingClassifier for binary
            # classification — multi-class raises InvalidModelError outright.
            return shap.TreeExplainer(clf), "tree"
        if model_type == "LogisticRegression" and background_sample is not None and len(background_sample) > 0:
            return shap.LinearExplainer(clf, background_sample), "linear"
    except Exception:
        pass

    return None, None  # e.g. multi-class GradientBoosting, KNN — no fast explainer available


def _explain_point(explainer, explainer_type, clf, scaled_pt, predicted_class, encoded_cols,
                    fallback_importance: Optional[List[Dict]] = None, top_n: int = 5) -> Tuple[Optional[List[Dict]], Optional[str]]:
    """
    Returns (explanation, explanation_type). explanation_type is "per_prediction"
    for a true SHAP explanation of this specific point, or "global_fallback" when
    falling back to the model's overall feature importance (same for every
    prediction, not specific to this point) because no fast explainer applied.
    """
    if explainer is not None:
        try:
            sv = explainer.shap_values(scaled_pt)
            class_idx = list(clf.classes_).index(predicted_class)
            if isinstance(sv, list):
                point_sv = np.asarray(sv[class_idx])[0]
            elif np.asarray(sv).ndim == 3:
                point_sv = np.asarray(sv)[0, :, class_idx]
            else:
                point_sv = np.asarray(sv)[0]

            pairs = sorted(zip(encoded_cols, point_sv.tolist()), key=lambda x: abs(x[1]), reverse=True)[:top_n]
            explanation = [
                {"feature": f, "contribution": round(v, 4), "direction": "increased" if v > 0 else "decreased"}
                for f, v in pairs
            ]
            return explanation, "per_prediction"
        except Exception:
            pass

    if fallback_importance:
        explanation = [
            {"feature": row["feature"], "contribution": row["importance"], "direction": None}
            for row in fallback_importance[:top_n]
        ]
        return explanation, "global_fallback"

    return None, None


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
        import pandas.api.types as ptypes

        missing = [c for c in [target_col] + feature_cols if c not in df.columns]
        if missing:
            return {"error": f"Columns not found: {missing}"}

        sub = df[[target_col] + feature_cols].copy().dropna()
        if len(sub) < 20:
            return {"error": "Need at least 20 rows for regression."}

        for col in feature_cols:
            # NOTE: was `if sub[col].dtype == object:` — that check silently
            # fails to catch pandas' newer native "string"/"str" dtype (seen
            # from pandas >= 2.x onward when reading Excel/CSV columns), so
            # a categorical column like "region" would never get LabelEncoded
            # and would instead go straight into pd.to_numeric(), which fails
            # on every row of real text and wipes the whole column to NaN —
            # collapsing the entire dataset to 0 rows on the final dropna().
            # Fix: explicitly check "is this NOT already numeric" rather than
            # relying on the exact legacy `object` dtype tag.
            if not ptypes.is_numeric_dtype(sub[col]):
                sub[col] = LabelEncoder().fit_transform(sub[col].astype(str))
            else:
                sub[col] = pd.to_numeric(sub[col], errors="coerce")
        sub = sub.dropna()

        if len(sub) < 20:
            return {"error": f"Only {len(sub)} valid rows remained after encoding features — need at least 20."}

        X, y = sub[feature_cols].values, sub[target_col].values
        if len(sub) > 5000:
            rng  = np.random.RandomState(42)
            idx  = rng.choice(len(sub), 5000, replace=False)
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


# ── Cluster Engine (V20.1 - Predictive HDBSCAN + UMAP) ─────────────────────────

class ClusterEngine:
    """
    Advanced non-linear clustering with predictive mapping:
    - HDBSCAN for density-based grouping and native noise identification.
    - UMAP for 2D topological mapping.
    - Post-hoc Random Forest Classifier for new-point classification.
    - Consistent categorical encoding via stored category levels (bug fix).
    """

    def __init__(self):
        self._groq_client  = None
        # ── State persisted across run() → predict_point() calls ──
        self.scaler          = None
        self.encoded_cols    = None   # column order the model was trained on
        self.feature_cols    = None   # original feature columns requested
        self.cat_mappings    = {}     # {col: [sorted list of all categories]}
        self.clf             = None   # production classifier
        self.cluster_profiles = None
        self.feature_defaults = {}    # {col: median (numeric) or mode (categorical)} — used to
                                       # impute missing fields in predict_point() rather than hard-fail
        self.explainer      = None    # SHAP explainer for the winning point-classifier, if fast one applies
        self.explainer_type = None    # "tree" | "linear" | None
        self._background_sample = None   # small sample of scaled training data, needed by LinearExplainer
        self.feature_importance_global = None   # fallback explanation when no fast SHAP explainer applies

    # ── Groq lazy init (unchanged) ─────────────────────────────────────────────

    def _get_groq_client(self):
        if self._groq_client is not None:
            return self._groq_client
        try:
            import os
            from groq import Groq
            key = os.getenv("GROQ_API_KEY")
            if not key:
                return None
            self._groq_client = Groq(api_key=key)
            return self._groq_client
        except Exception:
            return None

    def _compute_categorical_distinguishing_features(self, valid_clusters: "pd.DataFrame", cat_cols: list, top_n: int = 2) -> dict:
        """
        For each cluster, finds which category value is most OVER-REPRESENTED
        compared to its overall share in the dataset — e.g. "Region is 78% APAC
        vs 33% overall dataset-wide". This is the categorical counterpart to
        _compute_distinguishing_features() (which only covers numeric columns).
        Without this, a cluster discovered mostly by categorical structure
        (the common case for mixed data like Competence/Region) gets named
        using ONLY its numeric averages, missing the actual reason the cluster
        exists in the first place — this is why naming can feel generic/
        disconnected from the real driving factors on categorical-heavy data.
        """
        if valid_clusters.empty or not cat_cols:
            return {}

        overall_shares = {
            col: valid_clusters[col].value_counts(normalize=True)
            for col in cat_cols
        }

        result = {}
        for cluster_name, group in valid_clusters.groupby("Cluster"):
            findings = []
            for col in cat_cols:
                cluster_share = group[col].value_counts(normalize=True)
                overall = overall_shares[col]
                # Biggest positive gap between in-cluster share and overall share
                diffs = [(cat, cluster_share.get(cat, 0) - overall.get(cat, 0), cluster_share.get(cat, 0))
                         for cat in cluster_share.index]
                if not diffs:
                    continue
                diffs.sort(key=lambda x: x[1], reverse=True)
                top_cat, gap, cluster_pct = diffs[0]
                if gap > 0.05:  # only report a meaningful over-representation, not noise
                    overall_pct = overall.get(top_cat, 0)
                    findings.append(
                        f"{col} is {cluster_pct*100:.0f}% '{top_cat}' vs {overall_pct*100:.0f}% "
                        f"overall dataset-wide"
                    )
            result[cluster_name] = findings[:top_n]
        return result

    def _compute_distinguishing_features(self, valid_clusters: "pd.DataFrame", numeric_cols: list, top_n: int = 3) -> dict:
        """
        For each cluster, finds which numeric features deviate most (in % terms)
        from the OVERALL dataset average. This is the fix for vague naming: raw
        cluster averages (e.g. "fte=0.9") don't tell an LLM what's actually
        DIFFERENT about a cluster, so it falls back to generic words like
        "Medium Users". Framing it as "fte is 42% above the overall average"
        gives the naming step something genuinely distinguishing to describe.
        """
        import pandas as pd
        if valid_clusters.empty or not len(numeric_cols):
            return {}

        overall_means = valid_clusters[numeric_cols].mean()
        result = {}

        for cluster_name, group in valid_clusters.groupby("Cluster"):
            cluster_means = group[numeric_cols].mean()
            deltas = []
            for col in numeric_cols:
                overall = overall_means[col]
                cluster_val = cluster_means[col]
                if abs(overall) < 1e-9:
                    continue
                pct_diff = (cluster_val - overall) / abs(overall) * 100
                deltas.append((col, pct_diff, cluster_val))

            deltas.sort(key=lambda x: abs(x[1]), reverse=True)

            descriptions = []
            for col, pct_diff, cluster_val in deltas[:top_n]:
                direction = "above" if pct_diff > 0 else "below"
                descriptions.append(
                    f"{col} is {abs(pct_diff):.0f}% {direction} the overall dataset average "
                    f"(cluster avg: {cluster_val:.2f})"
                )
            result[cluster_name] = descriptions

        return result

    def _generate_labels(self, profiles: dict, distinguishing: dict = None, domain_context: str = None) -> dict:
        import json
        client = self._get_groq_client()
        if client is None:
            return {}

        distinguishing = distinguishing or {}
        context_line = f"\nDomain context provided by the user: {domain_context}\n" if domain_context else ""

        prompt = (
            "You are an expert business strategist naming customer/resource segments.\n"
            f"{context_line}"
            "Below is what makes each cluster DIFFERENT from the overall dataset average — use these "
            "distinguishing factors as the PRIMARY basis for naming, not raw averages alone:\n\n"
            f"{json.dumps(distinguishing, indent=2)}\n\n"
            "For additional grounding, here are the raw average feature values per cluster:\n"
            f"{json.dumps(profiles, indent=2)}\n\n"
            "Naming rules:\n"
            "- Give each cluster a sharp, specific, 2-4 word 'Persona Name' that reflects its most "
            "distinguishing factor(s) — not a generic catch-all label.\n"
            "- AVOID vague words as the primary descriptor: 'Users', 'Group', 'Segment', 'Medium', "
            "'Average', 'Standard', 'Regular' should not carry the name on their own.\n"
            "- Prefer names that reference the actual driving factor AND its direction, e.g. "
            "'High-Utilization Specialists', 'Low-FTE Contributors', 'Premium Cost Drivers' — "
            "not 'Medium Users' or 'Group A'.\n"
            "- Every persona name you return MUST be unique across all clusters — even if two clusters "
            "look statistically similar, choose distinguishing wording rather than reusing a name.\n\n"
            "Return ONLY a valid JSON object where keys are the exact original cluster names "
            "(e.g., 'Cluster 0') and values are the persona name strings. No markdown, no explanations."
        )
        try:
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            labels = json.loads(resp.choices[0].message.content.strip())
            return self._dedupe_labels(labels)
        except Exception as e:
            print(f"[ClusterEngine] Auto-labeling failed: {e}")
            return {}

    def _dedupe_labels(self, labels: dict) -> dict:
        """
        Guards against the LLM reusing the same persona name for multiple
        distinct clusters (e.g. giving 3 different clusters the name "Medium
        Users" because their profiles look superficially similar). If that
        happens, downstream code that colors/groups by the "Cluster" column
        silently merges those distinct clusters into one visual group, even
        though the underlying cluster assignments (and the reported k) are
        still correct. Here we detect collisions and disambiguate by
        appending a numeric suffix, so every original cluster keeps a
        distinct label.
        """
        seen: dict = {}
        deduped = {}
        for original_cluster, name in labels.items():
            if name not in seen:
                seen[name] = 1
                deduped[original_cluster] = name
            else:
                seen[name] += 1
                deduped[original_cluster] = f"{name} ({seen[name]})"
        return deduped

    # ── Consistent encoding helper (THE CORE FIX) ──────────────────────────────

    def _encode_consistent(self, df_in: "pd.DataFrame") -> "pd.DataFrame":
        """
        Encode a DataFrame using stored category levels from training.

        Why this matters:
          pd.get_dummies on a single-row DataFrame only sees the ONE value
          present in that row. With drop_first=True it then drops the only
          dummy it created, so every category value collapses to all-zeros.

          By using pd.Categorical with the FULL list of categories seen
          during training, pandas always emits one column per level —
          even when only a single row (or a subset of levels) is present.
          drop_first=False ensures no information is silently discarded.

        After encoding we align to self.encoded_cols so the feature matrix
        always has the exact same shape and column order the model expects.
        """
        import pandas as pd
        df_copy = df_in.copy()

        # Apply stored categories to each known categorical column
        for col, cats in self.cat_mappings.items():
            if col in df_copy.columns:
                # Cast to Categorical with all training-time levels
                df_copy[col] = pd.Categorical(df_copy[col], categories=cats)

        # drop_first=False: every level keeps its column
        encoded = pd.get_dummies(df_copy, drop_first=False)

        # Align to the training column order (fill any unseen cols with 0)
        if self.encoded_cols is not None:
            final = pd.DataFrame(0, index=df_copy.index, columns=self.encoded_cols)
            for col in self.encoded_cols:
                if col in encoded.columns:
                    final[col] = encoded[col].values
            return final

        return encoded

    # ── run() ──────────────────────────────────────────────────────────────────

    # ── Context-aware clustering algorithm selection ────────────────────────────

    def _select_clustering_algorithm(self, sub: "pd.DataFrame", feature_cols: list,
                                       scaled_data, encoded_df: "pd.DataFrame") -> Optional[Dict]:
        """
        Profiles the data (row count, categorical-vs-numeric mix, post-encoding
        dimensionality) and runs a shortlist of ELIGIBLE clustering algorithms
        for that shape of data, scoring each with a composite of three
        unsupervised quality metrics — Silhouette, Davies-Bouldin, and
        Calinski-Harabasz — rather than a single score, since silhouette alone
        can be misleading for density-based methods that produce noise points,
        or when comparing fundamentally different algorithms against each other.

        All candidates are scored on the SAME standardized one-hot feature
        space (scaled_data) for a fair apples-to-apples comparison — even
        K-Prototypes, which actually PERFORMS its clustering on the raw
        mixed-type data (treating categories natively via Hamming distance
        instead of exploding them into sparse dummy columns), has its
        resulting assignment evaluated on scaled_data just like every other
        candidate. This is what makes K-Prototypes eligible in the first
        place: one-hot encoding a lot of categorical columns dilutes numeric
        signal, so when a meaningful share of the requested features are
        categorical, K-Prototypes gets a real chance to compete instead of
        being silently unavailable.

        Returns None if no candidate produced anything usable (caller falls
        back to the older fixed HDBSCAN → K-Means logic).
        """
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

        n_rows = len(scaled_data)
        cat_ratio = (len(self.cat_mappings) / len(feature_cols)) if feature_cols else 0

        candidates = {}   # name -> (labels array, extra_meta dict)

        # ── HDBSCAN — always eligible; strong default for numeric-heavy data,
        # native noise detection. Searches a few min_cluster_size fractions
        # rather than one fixed guess (n_rows // 15 alone can produce 100%
        # noise on tightly-grouped data — confirmed on real test data during
        # development), same spirit as the k-search every other candidate does. ──
        try:
            try:
                from sklearn.cluster import HDBSCAN
            except ImportError:
                from hdbscan import HDBSCAN

            best_score, best_labels, best_min_sz = -1, None, None
            for frac in (0.01, 0.02, 0.03, 0.05, 1.0 / 15):
                min_sz = max(5, int(n_rows * frac))
                try:
                    lbls = HDBSCAN(min_cluster_size=min_sz).fit_predict(scaled_data)
                except Exception:
                    continue
                mask = lbls != -1
                if mask.sum() < 3 or len(np.unique(lbls[mask])) < 2:
                    continue
                # A candidate that's mostly noise isn't usable even if its
                # silhouette on the small inlier remainder looks decent.
                if mask.sum() / n_rows < 0.5:
                    continue
                # Same "usable for a human audience" cap as the leaderboard
                # filter below — a very small min_cluster_size can carve
                # categorical-heavy data into dozens of tight micro-clusters
                # that score well statistically but aren't a usable persona set.
                if len(np.unique(lbls[mask])) > 15:
                    continue
                score = silhouette_score(scaled_data[mask], lbls[mask])
                if score > best_score:
                    best_score, best_labels, best_min_sz = score, lbls, min_sz

            if best_labels is not None:
                candidates["HDBSCAN"] = (best_labels, {"handles_noise": True, "min_cluster_size": best_min_sz})
        except Exception:
            pass

        # ── K-Means — always eligible; fast, good baseline for roughly
        # spherical clusters. ──
        try:
            from sklearn.cluster import KMeans
            best_k, best_score, best_labels = None, -1, None
            max_k = min(12, max(2, n_rows // 15))
            for k in range(2, max_k + 1):
                if k >= n_rows: break
                km = KMeans(n_clusters=k, random_state=42, n_init="auto")
                lbls = km.fit_predict(scaled_data)
                if len(np.unique(lbls)) < 2: continue
                score = silhouette_score(scaled_data, lbls)
                if score > best_score:
                    best_k, best_score, best_labels = k, score, lbls
            if best_labels is not None:
                candidates["K-Means"] = (best_labels, {"handles_noise": False, "k": best_k})
        except Exception:
            pass

        # ── Gaussian Mixture — eligible for moderate dataset sizes; captures
        # elliptical/overlapping clusters and gives soft membership natively. ──
        try:
            if n_rows <= 50000:
                from sklearn.mixture import GaussianMixture
                best_k, best_score, best_labels = None, -1, None
                max_k = min(10, max(2, n_rows // 20))
                for k in range(2, max_k + 1):
                    gmm = GaussianMixture(n_components=k, random_state=42, n_init=1)
                    lbls = gmm.fit_predict(scaled_data)
                    if len(np.unique(lbls)) < 2: continue
                    score = silhouette_score(scaled_data, lbls)
                    if score > best_score:
                        best_k, best_score, best_labels = k, score, lbls
                if best_labels is not None:
                    candidates["Gaussian Mixture"] = (best_labels, {"handles_noise": False, "k": best_k})
        except Exception:
            pass

        # ── K-Prototypes — ONLY eligible when the data is genuinely mixed/
        # categorical-heavy (cat_ratio >= 0.3). This is the "context aware"
        # part of the selection. ──
        kprototypes_note = None
        if cat_ratio >= 0.3 and self.cat_mappings:
            try:
                from kmodes.kprototypes import KPrototypes
                from sklearn.preprocessing import StandardScaler as _SS

                raw = sub[feature_cols].copy()
                cat_idx = [i for i, c in enumerate(feature_cols) if c in self.cat_mappings]
                num_cols = [c for c in feature_cols if c not in self.cat_mappings]
                if num_cols:
                    raw[num_cols] = _SS().fit_transform(raw[num_cols])
                for c in self.cat_mappings:
                    raw[c] = raw[c].astype(str)

                best_k, best_score, best_labels = None, -1, None
                max_k = min(10, max(2, n_rows // 20))
                per_k_errors = []
                for k in range(2, max_k + 1):
                    try:
                        kp = KPrototypes(n_clusters=k, init="Huang", random_state=42, n_init=2, verbose=0)
                        lbls = kp.fit_predict(raw.values, categorical=cat_idx)
                        if len(np.unique(lbls)) < 2: continue
                        # Scored on the shared scaled_data, same as every other candidate.
                        score = silhouette_score(scaled_data, lbls)
                        if score > best_score:
                            best_k, best_score, best_labels = k, score, lbls
                    except Exception as e:
                        per_k_errors.append(str(e))
                        continue

                if best_labels is not None:
                    candidates["K-Prototypes"] = (
                        best_labels,
                        {"handles_noise": False, "k": best_k, "native_categorical": True},
                    )
                    kprototypes_note = "included in tournament"
                else:
                    # kmodes imported fine, but every k value it tried failed or
                    # produced a degenerate (<2 cluster) result — surface why,
                    # since a silent "just absent" leaves no way to diagnose this.
                    reason = per_k_errors[-1] if per_k_errors else "no k value produced >=2 clusters"
                    kprototypes_note = f"eligible but unusable: {reason}"
                    print(f"[ClusterEngine] K-Prototypes eligible but produced no usable clustering: {reason}")
            except ImportError as e:
                kprototypes_note = f"not installed ({e}) — run: pip install kmodes"
                print(f"[ClusterEngine] K-Prototypes unavailable — kmodes not installed: {e}")
            except Exception as e:
                # Anything else (bad dtype, sklearn/kmodes version mismatch, etc.)
                # — catch broadly here so a K-Prototypes-specific failure can
                # never take down the whole clustering request, but log the
                # real reason instead of silently pretending it just wasn't eligible.
                kprototypes_note = f"failed unexpectedly: {e}"
                print(f"[ClusterEngine] K-Prototypes failed unexpectedly: {e}")
        else:
            kprototypes_note = f"not attempted — categorical ratio {cat_ratio:.2f} below the 0.3 eligibility threshold"

        if not candidates:
            return None

        # ── Score every candidate on 3 unsupervised metrics, normalize, combine ──
        # Cap at a "usable for a human audience" cluster count — a result can be
        # statistically tighter with many small clusters (confirmed during testing:
        # HDBSCAN's finest min_cluster_size fraction found 36 clusters on
        # categorical-heavy data, technically outscoring a 6-cluster result), but
        # 36 auto-named personas isn't a useful output for a business audience.
        # None of the unsupervised quality metrics know or care about that — this
        # cap is what keeps the tournament's answer actually usable downstream.
        MAX_USABLE_CLUSTERS = 15

        leaderboard = []
        for name, (labels, extra) in candidates.items():
            mask = labels != -1   # exclude HDBSCAN noise from quality scoring
            if mask.sum() < 3 or len(np.unique(labels[mask])) < 2:
                continue
            if len(np.unique(labels[mask])) > MAX_USABLE_CLUSTERS:
                continue
            try:
                sil = silhouette_score(scaled_data[mask], labels[mask])
                db  = davies_bouldin_score(scaled_data[mask], labels[mask])
                ch  = calinski_harabasz_score(scaled_data[mask], labels[mask])
            except Exception:
                continue
            leaderboard.append({
                "algorithm": name, "labels": labels, "extra": extra,
                "silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch,
                "n_clusters": len(np.unique(labels[mask])),
                "noise_count": int((~mask).sum()),
            })

        if not leaderboard:
            return None

        sils = [r["silhouette"] for r in leaderboard]
        dbs  = [r["davies_bouldin"] for r in leaderboard]
        chs  = [r["calinski_harabasz"] for r in leaderboard]

        def _norm(vals, higher_better=True):
            lo, hi = min(vals), max(vals)
            if hi - lo < 1e-9:
                return [0.5] * len(vals)
            normed = [(v - lo) / (hi - lo) for v in vals]
            return normed if higher_better else [1 - v for v in normed]

        n_sil = _norm(sils, True)
        n_db  = _norm(dbs, False)
        n_ch  = _norm(chs, True)

        # Weighting: silhouette weighted highest since it's the most widely
        # understood/interpretable clustering quality metric; Davies-Bouldin
        # and Calinski-Harabasz split the remainder as corroborating signals.
        for i, row in enumerate(leaderboard):
            row["composite_score"] = round(0.4 * n_sil[i] + 0.3 * n_db[i] + 0.3 * n_ch[i], 4)

        leaderboard.sort(key=lambda r: r["composite_score"], reverse=True)
        for i, row in enumerate(leaderboard):
            row["selected"] = "✓" if i == 0 else ""

        winner = leaderboard[0]
        return {
            "winner_name":   winner["algorithm"],
            "winner_labels": winner["labels"],
            "winner_extra":  winner["extra"],
            "kprototypes_note": kprototypes_note,
            "leaderboard": [
                {
                    "algorithm":         r["algorithm"],
                    "silhouette":        round(r["silhouette"], 4),
                    "davies_bouldin":    round(r["davies_bouldin"], 4),
                    "calinski_harabasz": round(r["calinski_harabasz"], 2),
                    "n_clusters":        r["n_clusters"],
                    "noise_count":       r["noise_count"],
                    "composite_score":   r["composite_score"],
                    "selected":          r["selected"],
                }
                for r in leaderboard
            ],
        }

    def run(self, df, feature_cols, domain_context: str = None):
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import silhouette_score

        self.feature_cols = feature_cols

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

        # ── 1. Store category levels BEFORE encoding ───────────────────────────
        # This is what makes predict_point() work correctly later.
        self.cat_mappings = {}
        for col in sub.columns:
            import pandas.api.types as _pat
            if _pat.is_string_dtype(sub[col]) or _pat.is_object_dtype(sub[col]) or str(sub[col].dtype) in ("category", "string"):
                cats = sorted(sub[col].dropna().unique().tolist())
                if len(cats) >= 2:
                    self.cat_mappings[col] = cats

        # ── 1b. Store per-feature defaults for graceful partial predictions ────
        # If a later predict_point() call is missing a field, we impute it
        # from these training-time defaults (median for numeric, mode for
        # categorical) rather than hard-failing — see predict_point() for how
        # this trades off against prediction reliability.
        self.feature_defaults = {}
        for col in feature_cols:
            if col in self.cat_mappings:
                mode_vals = sub[col].mode(dropna=True)
                self.feature_defaults[col] = mode_vals.iloc[0] if not mode_vals.empty else None
            else:
                try:
                    self.feature_defaults[col] = float(sub[col].median())
                except Exception:
                    self.feature_defaults[col] = None

        # ── 2. Encode training data with consistent helper ─────────────────────
        # self.encoded_cols is None here, so _encode_consistent falls through
        # to a plain get_dummies. We capture the column order afterwards.
        sub_for_enc = sub.copy()
        for col, cats in self.cat_mappings.items():
            sub_for_enc[col] = pd.Categorical(sub_for_enc[col], categories=cats)
        encoded_df = pd.get_dummies(sub_for_enc, drop_first=False)

        self.encoded_cols = encoded_df.columns.tolist()

        if len(self.encoded_cols) < 2:
            return {"error": "Clustering requires at least 2 distinct features."}

        self.scaler  = StandardScaler()
        scaled_data  = self.scaler.fit_transform(encoded_df)

        # ── 3. Clustering — context-aware algorithm tournament ─────────────────
        # Profiles the data and runs a shortlist of eligible algorithms (HDBSCAN,
        # K-Means, Gaussian Mixture, and K-Prototypes when the data is
        # categorical-heavy), scored on a composite of 3 unsupervised quality
        # metrics. Falls back to the old fixed HDBSCAN→K-Means logic only if the
        # tournament produced nothing usable at all (defensive, should be rare).
        tournament = self._select_clustering_algorithm(sub, feature_cols, scaled_data, encoded_df)

        if tournament is not None:
            labels = tournament["winner_labels"]
            clust_algo = tournament["winner_name"]
            n_clusters = len([c for c in set(labels) if c != -1])
            clustering_leaderboard = tournament["leaderboard"]
            kprototypes_note = tournament.get("kprototypes_note")
        else:
            if use_hdbscan:
                min_sz = max(5, len(scaled_data) // 15)
                hdb    = HDBSCAN(min_cluster_size=min_sz)
                labels = hdb.fit_predict(scaled_data)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters < 2:
                    use_hdbscan = False  # fall through to KMeans
                else:
                    clust_algo = "HDBSCAN"

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
                clust_algo = "K-Means"

            clustering_leaderboard = []
            kprototypes_note = "tournament unavailable — fell back to legacy HDBSCAN/K-Means logic"

        sub["Cluster"]      = [f"Cluster {x}" if x != -1 else "Outlier (Noise)" for x in labels]
        is_inlier_final     = labels != -1
        n_outliers          = int((~is_inlier_final).sum())

        sil_score = -1.0
        if is_inlier_final.sum() > 2 and n_clusters >= 2:
            try:
                sil_score = silhouette_score(
                    scaled_data[is_inlier_final], labels[is_inlier_final]
                )
            except Exception:
                pass

        # ── 4. Feature importance ──────────────────────────────────────────────
        fi_df      = None
        top_driver = "Unknown"
        if is_inlier_final.sum() > 10 and n_clusters >= 2:
            rf_explainer = RandomForestClassifier(
                n_estimators=100, max_depth=5, random_state=42
            )
            rf_explainer.fit(scaled_data[is_inlier_final], labels[is_inlier_final])
            fi_df = pd.DataFrame({
                "feature":    self.encoded_cols,
                "importance": rf_explainer.feature_importances_.round(4),
            }).sort_values(by="importance", ascending=False).head(10)
            top_driver = fi_df.iloc[0]["feature"]

        # ── 5. 2D projection ───────────────────────────────────────────────────
        if use_umap:
            from umap import UMAP
            reducer    = UMAP(n_components=2, random_state=42)
            coords     = reducer.fit_transform(scaled_data)
            algo_name  = "UMAP"
            pca_var    = [0.0, 0.0]
        else:
            from sklearn.decomposition import PCA
            reducer    = PCA(n_components=2)
            reducer.fit(scaled_data[is_inlier_final])   # fit on inliers only
            coords     = reducer.transform(scaled_data)  # transform all
            algo_name  = "PCA"
            pca_var    = [round(v, 4) for v in reducer.explained_variance_ratio_.tolist()]

        sub["PCA_X"] = coords[:, 0].round(3)
        sub["PCA_Y"] = coords[:, 1].round(3)

        # ── 6. Cluster profiles & LLM labeling ────────────────────────────────
        valid_clusters = sub[sub["Cluster"] != "Outlier (Noise)"]
        numeric_orig   = valid_clusters.select_dtypes(include=np.number).drop(
            columns=["PCA_X", "PCA_Y"], errors="ignore"
        )
        profiles = {}
        if not numeric_orig.empty:
            profiles = (
                valid_clusters.groupby("Cluster")[numeric_orig.columns]
                .mean().round(2).to_dict("index")
            )

        # Two-step naming: compute what's actually DISTINCTIVE about each
        # cluster (vs. the overall dataset average) before asking Groq to
        # name it — raw averages alone tend to produce vague names like
        # "Medium Users". _generate_labels() also still guarantees unique
        # persona names via _dedupe_labels().
        distinguishing = self._compute_distinguishing_features(valid_clusters, numeric_orig.columns.tolist())

        # Merge in categorical distinguishing features (e.g. "Region is 78% APAC")
        # — see _compute_categorical_distinguishing_features() docstring for why
        # this matters specifically for mixed categorical/numeric clustering.
        cat_cols_used = [c for c in feature_cols if c in self.cat_mappings]
        if cat_cols_used:
            cat_distinguishing = self._compute_categorical_distinguishing_features(valid_clusters, cat_cols_used)
            for cluster_name, cat_findings in cat_distinguishing.items():
                distinguishing[cluster_name] = cat_findings + distinguishing.get(cluster_name, [])

        new_labels = self._generate_labels(profiles, distinguishing, domain_context=domain_context)
        if new_labels:
            sub["Cluster"] = sub["Cluster"].map(lambda x: new_labels.get(x, x))
            profiles       = {new_labels.get(k, k): v for k, v in profiles.items()}

        self.cluster_profiles = profiles
        # clust_algo already set above by the tournament (or its fallback) — no overwrite here.

        # ── 7. Production classifier (trained on ALL points incl. noise) ───────
        # Same tournament philosophy as the clustering algorithm choice above —
        # compares several classifier types instead of hardcoding Random Forest.
        clf_tournament = _run_classifier_tournament(scaled_data, sub["Cluster"].values)
        self.clf = clf_tournament["model"]
        classifier_winner = clf_tournament["winner"]
        classifier_leaderboard = clf_tournament["leaderboard"]
        classifier_skipped_reason = clf_tournament.get("skipped_reason")

        # SHAP explainability for predict_point() — background sample needed
        # by LinearExplainer; fi_df (already computed above via rf_explainer)
        # doubles as the global-importance fallback for models with no fast
        # per-point explainer (multi-class Gradient Boosting, KNN).
        self._background_sample = scaled_data[np.random.RandomState(42).choice(
            len(scaled_data), min(100, len(scaled_data)), replace=False
        )]
        self.feature_importance_global = (
            fi_df.to_dict("records") if fi_df is not None else None
        )
        self.explainer, self.explainer_type = _build_shap_explainer(self.clf, self._background_sample)

        narrative = (
            f"Segmented into {n_clusters} distinct personas using {clust_algo} + {algo_name} "
            f"(chosen from {len(clustering_leaderboard) or 1} candidate clustering algorithm(s) "
            f"by composite quality score). "
            f"{n_outliers} noise/outliers natively isolated. "
            f"#1 driving factor: '{top_driver}' (standardized importance). "
            f"Point-prediction powered by {classifier_winner} "
            f"(chosen from {len(classifier_leaderboard) or 1} candidate model(s) by weighted F1). "
        )
        for persona_name, prof in profiles.items():
            top_feat   = ", ".join([f"{k}={v}" for k, v in list(prof.items())[:3]])
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
            "pca_variance":  pca_var,
            "fi_note":       "Importance scores are on the standardized (z-score) feature scale.",
            "cat_mappings":  {k: v for k, v in self.cat_mappings.items()},  # expose for debugging
            "distinguishing_features": distinguishing,  # what made each persona name what it is
            "clustering_leaderboard":  clustering_leaderboard,   # algorithm tournament results
            "kprototypes_note":        kprototypes_note,   # diagnostic: why K-Prototypes was/wasn't in the leaderboard
            "classifier_model":       classifier_winner,
            "classifier_leaderboard": classifier_leaderboard,    # point-classifier tournament results
            "classifier_skipped_reason": classifier_skipped_reason,  # why the tournament fell back to a single model, if it did
        }

        return {"data": sub, "meta": meta, "feature_importance": fi_df}

    # ── predict_point() ────────────────────────────────────────────────────────

    def predict_point(self, point_dict: dict, allow_partial: bool = True) -> dict:
        """
        Classify a new data point into one of the established personas.

        Uses _encode_consistent() which applies the stored category levels
        so every categorical column is encoded identically to training time.
        This is what fixes the rural/urban collision bug.

        If `allow_partial` is True (default) and the caller omits some
        trained features, those are imputed from training-time defaults
        (median for numeric, mode for categorical) rather than hard-failing.
        This trades prediction reliability for convenience — the response
        includes a `defaulted_fields` list and `warning` string whenever this
        happens, so callers can decide how much to trust the result. Set
        `allow_partial=False` to restore the old strict "error on any missing
        field" behavior.
        """
        import pandas as pd

        if self.clf is None or self.scaler is None:
            return {
                "error": (
                    "The clustering engine must be run on a base dataset "
                    "before classifying individual points."
                )
            }

        missing = [c for c in self.feature_cols if c not in point_dict]
        defaulted_fields = []

        if missing:
            if not allow_partial:
                return {"error": f"Missing required fields: {missing}"}

            point_dict = dict(point_dict)  # don't mutate the caller's dict
            still_missing = []
            for col in missing:
                default_val = self.feature_defaults.get(col)
                if default_val is None:
                    still_missing.append(col)
                else:
                    point_dict[col] = default_val
                    defaulted_fields.append(col)

            if still_missing:
                return {"error": f"Missing required fields with no available default: {still_missing}"}

        # Build a single-row DataFrame, then encode using stored categories
        input_df     = pd.DataFrame([point_dict])
        encoded_pt   = self._encode_consistent(input_df)   # handles drop_first=False + alignment
        scaled_pt    = self.scaler.transform(encoded_pt)

        predicted_persona = self.clf.predict(scaled_pt)[0]
        probabilities     = self.clf.predict_proba(scaled_pt)[0]
        classes           = self.clf.classes_

        confidence_scores  = {
            cls: round(float(prob), 4)
            for cls, prob in zip(classes, probabilities)
        }
        match_confidence = round(float(max(probabilities)), 4)

        result = {
            "assigned_persona":  predicted_persona,
            "match_confidence":  match_confidence,
            "confidence_matrix": confidence_scores,
        }

        explanation, explanation_type = _explain_point(
            self.explainer, self.explainer_type, self.clf, scaled_pt,
            predicted_persona, self.encoded_cols, self.feature_importance_global,
        )
        if explanation:
            result["explanation"] = explanation
            result["explanation_type"] = explanation_type

        if defaulted_fields:
            result["defaulted_fields"] = defaulted_fields
            result["warning"] = (
                f"These fields weren't specified and were defaulted to typical training values: "
                f"{', '.join(defaulted_fields)}. Treat this prediction with lower confidence, since "
                f"it's based on incomplete information."
            )

        return result

    # ── predict_point_from_text() ──────────────────────────────────────────────

    def predict_point_from_text(self, question: str) -> dict:
        """Extract features from natural language, then classify."""
        if self.clf is None or self.scaler is None:
            return {
                "error": (
                    "I need to cluster your data first before I can classify "
                    "individual users. Try asking me to segment the data first!"
                )
            }

        import json
        client = self._get_groq_client()
        if client is None:
            return {
                "error": "Groq API key required for natural language feature extraction."
            }

        # Tell the LLM what types each feature expects
        feature_type_hints = []
        for col in self.feature_cols:
            if col in self.cat_mappings:
                opts = ", ".join(self.cat_mappings[col])
                feature_type_hints.append(f"'{col}' (categorical, one of: {opts})")
            else:
                feature_type_hints.append(f"'{col}' (numeric)")

        prompt = (
            "You are a strict data extraction tool.\n"
            f"Extract the attributes from this query: '{question}'.\n"
            f"Expected features: {', '.join(feature_type_hints)}.\n"
            "Rules:\n"
            "- For categorical features, return the EXACT string value (e.g. 'rural', 'urban').\n"
            "- For numeric features, return a number.\n"
            "- If a feature is missing from the query, return null.\n"
            "Return ONLY a valid JSON object with the feature names as keys. "
            "No markdown, no explanation."
        )

        try:
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            point_dict = json.loads(resp.choices[0].message.content.strip())

            # Remove null values — predict_point handles missing feature check
            point_dict = {k: v for k, v in point_dict.items() if v is not None}

            prediction = self.predict_point(point_dict)
            if "error" not in prediction:
                prediction["extracted_features"] = point_dict
            return prediction

        except Exception as e:
            return {"error": f"Failed to extract features from text: {str(e)}"}


# ── Supervised Classifier Engine ────────────────────────────────────────────────

class SupervisedClassifierEngine:
    """
    Trains a classifier against a REAL label column already present in the
    user's data (e.g. "Risk_Tier", "Churn_Status", "Approval_Outcome") — as
    opposed to ClusterEngine, which DISCOVERS groups with no ground truth.

    Use this when the user has labeled historical data and wants to predict
    that same label for new, unlabeled records. Use ClusterEngine when there
    is no existing label and the goal is to find natural groupings.

    Mirrors ClusterEngine's session-scoped state pattern (trained model lives
    on the instance, so each session/label-column needs its own instance —
    see main.py's per-session `classifiers` dict).
    """

    def __init__(self):
        self._groq_client       = None
        self.scaler             = None
        self.encoded_cols       = None   # column order the model was trained on
        self.feature_cols       = None   # original feature columns requested
        self.label_col          = None   # the real label column being predicted
        self.cat_mappings       = {}     # {col: [sorted list of all categories]}
        self.clf                = None   # trained classifier
        self.classes_           = None
        self.label_distribution = None
        self.feature_defaults   = {}   # {col: median (numeric) or mode (categorical)} — used to
                                        # impute missing fields in predict_point() rather than hard-fail
        self.explainer      = None
        self.explainer_type = None
        self._background_sample = None
        self.feature_importance_global = None

    # ── Groq lazy init (same pattern as ClusterEngine) ──────────────────────────

    def _get_groq_client(self):
        if self._groq_client is not None:
            return self._groq_client
        try:
            import os
            from groq import Groq
            key = os.getenv("GROQ_API_KEY")
            if not key:
                return None
            self._groq_client = Groq(api_key=key)
            return self._groq_client
        except Exception:
            return None

    # ── Consistent encoding helper (same mechanism as ClusterEngine) ────────────

    def _encode_consistent(self, df_in: "pd.DataFrame") -> "pd.DataFrame":
        import pandas as pd
        df_copy = df_in.copy()

        for col, cats in self.cat_mappings.items():
            if col in df_copy.columns:
                df_copy[col] = pd.Categorical(df_copy[col].astype(str), categories=cats)

        encoded = pd.get_dummies(df_copy, drop_first=False)

        if self.encoded_cols is not None:
            final = pd.DataFrame(0, index=df_copy.index, columns=self.encoded_cols)
            for col in self.encoded_cols:
                if col in encoded.columns:
                    final[col] = encoded[col].values
            return final

        return encoded

    # ── run() — train against a real label column ───────────────────────────────

    def run(self, df, feature_cols, label_col):
        import numpy as np
        import pandas as pd
        import pandas.api.types as _pat
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

        missing = [c for c in feature_cols + [label_col] if c not in df.columns]
        if missing:
            return {"error": f"Columns not found: {missing}"}

        self.feature_cols = feature_cols
        self.label_col    = label_col

        sub = df[feature_cols + [label_col]].copy().dropna()
        if len(sub) < 20:
            return {"error": "Need at least 20 rows to train a supervised classifier."}

        n_classes = sub[label_col].nunique()
        if n_classes < 2:
            return {"error": f"Label column '{label_col}' has only {n_classes} distinct value — need at least 2 classes to classify."}
        if n_classes > 20:
            return {
                "error": (
                    f"Label column '{label_col}' has {n_classes} distinct values — unusually high for "
                    "classification. If this is a continuous/numeric target, use Regression instead; "
                    "if it's meant to be categorical, consider grouping it into fewer categories first."
                )
            }

        y_raw = sub[label_col].astype(str)
        self.label_distribution = y_raw.value_counts().to_dict()

        X_df = sub[feature_cols].copy()

        # Store category levels for feature columns — same mechanism as
        # ClusterEngine, needed so a single-row prediction later encodes
        # identically to how training data was encoded.
        self.cat_mappings = {}
        for col in X_df.columns:
            if not _pat.is_numeric_dtype(X_df[col]):
                X_df[col] = X_df[col].astype(str)
                cats = sorted(X_df[col].dropna().unique().tolist())
                if len(cats) >= 2:
                    self.cat_mappings[col] = cats

        # Store per-feature defaults for graceful partial predictions — see
        # ClusterEngine.predict_point() for the identical rationale/mechanism.
        self.feature_defaults = {}
        for col in feature_cols:
            if col in self.cat_mappings:
                mode_vals = X_df[col].mode(dropna=True)
                self.feature_defaults[col] = mode_vals.iloc[0] if not mode_vals.empty else None
            else:
                try:
                    self.feature_defaults[col] = float(X_df[col].median())
                except Exception:
                    self.feature_defaults[col] = None

        X_for_enc = X_df.copy()
        for col, cats in self.cat_mappings.items():
            X_for_enc[col] = pd.Categorical(X_for_enc[col], categories=cats)
        encoded_df = pd.get_dummies(X_for_enc, drop_first=False)
        self.encoded_cols = encoded_df.columns.tolist()

        if len(self.encoded_cols) < 1:
            return {"error": "No usable feature columns after encoding."}

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(encoded_df)

        # Stratify when feasible (every class needs enough rows for a test split)
        min_class_count = y_raw.value_counts().min()
        stratify = y_raw if min_class_count >= 2 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_raw, test_size=0.2, random_state=42, stratify=stratify
        )

        # Model tournament — compares Random Forest, Gradient Boosting, Extra
        # Trees, Logistic Regression, and KNN via CV on the TRAIN split only,
        # so the held-out X_test/y_test below stays untouched for an honest
        # final accuracy/F1 report. Same helper ClusterEngine's point-classifier
        # uses — see _run_classifier_tournament for the full rationale.
        train_tournament = _run_classifier_tournament(X_train, y_train)
        eval_model = train_tournament["model"]   # fit on X_train only
        classifier_leaderboard = train_tournament["leaderboard"]
        classifier_winner = train_tournament["winner"]
        classifier_skipped_reason = train_tournament.get("skipped_reason")

        preds    = eval_model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        f1       = f1_score(y_test, preds, average="weighted")
        classes_sorted = sorted(y_raw.unique().tolist())
        cm       = confusion_matrix(y_test, preds, labels=classes_sorted)

        # Feature importance: only tree-based winners expose feature_importances_
        # directly; Logistic Regression exposes coefficients instead (used as an
        # importance proxy via mean absolute magnitude); KNN exposes neither, so
        # feature importance is simply omitted rather than fabricated for it.
        if hasattr(eval_model, "feature_importances_"):
            importances = eval_model.feature_importances_
        elif hasattr(eval_model, "coef_"):
            importances = np.abs(eval_model.coef_).mean(axis=0)
        else:
            importances = None

        if importances is not None:
            total = importances.sum()
            fi_df = pd.DataFrame({
                "feature":    self.encoded_cols,
                "importance": np.round(importances / total, 4) if total > 0 else np.round(importances, 4),
            }).sort_values("importance", ascending=False).head(10)
        else:
            fi_df = pd.DataFrame(columns=["feature", "importance"])

        # Refit the SAME winning algorithm type on ALL available data (train+test
        # combined) for the production model used by predict_point() — re-running
        # the tournament on the full dataset rather than just refitting the
        # train-split winner, since the best-performing algorithm CAN differ once
        # more data is available; the held-out metrics above remain the honest
        # report regardless of what the full-data tournament picks.
        full_tournament = _run_classifier_tournament(X_scaled, y_raw)
        self.clf = full_tournament["model"]
        self.classes_ = sorted(self.clf.classes_.tolist())

        # SHAP explainability for predict_point() — background sample needed by
        # LinearExplainer; global-importance fallback computed fresh from the
        # PRODUCTION model (not the held-out eval_model above), since that's
        # the model actually making predictions.
        self._background_sample = X_scaled[np.random.RandomState(42).choice(
            len(X_scaled), min(100, len(X_scaled)), replace=False
        )]
        if hasattr(self.clf, "feature_importances_"):
            prod_importances = self.clf.feature_importances_
        elif hasattr(self.clf, "coef_"):
            prod_importances = np.abs(self.clf.coef_).mean(axis=0)
        else:
            prod_importances = None
        if prod_importances is not None:
            total = prod_importances.sum()
            normed = prod_importances / total if total > 0 else prod_importances
            self.feature_importance_global = [
                {"feature": f, "importance": round(float(v), 4)}
                for f, v in sorted(zip(self.encoded_cols, normed.tolist()), key=lambda x: x[1], reverse=True)
            ]
        else:
            self.feature_importance_global = None
        self.explainer, self.explainer_type = _build_shap_explainer(self.clf, self._background_sample)

        comparison = pd.DataFrame({
            "actual":    y_test.values,
            "predicted": preds,
            "correct":   (y_test.values == preds),
        })

        top_feature = fi_df.iloc[0]["feature"] if not fi_df.empty else "Unknown"
        narrative = (
            f"Compared {len(classifier_leaderboard) or 1} classifier model(s) to predict **{label_col}** "
            f"from {len(feature_cols)} feature(s) across {n_classes} classes — **{classifier_winner}** won "
            f"on weighted F1. Held-out accuracy: **{accuracy*100:.1f}%**, "
            f"weighted F1: **{f1:.3f}**. Top predictive feature: **{top_feature}**."
        )

        return {
            "data": comparison,
            "meta": {
                "label_col":              label_col,
                "feature_cols":           feature_cols,
                "classes":                classes_sorted,
                "n_classes":              n_classes,
                "accuracy":               round(float(accuracy), 4),
                "f1_weighted":            round(float(f1), 4),
                "confusion_matrix":       cm.tolist(),
                "label_distribution":     self.label_distribution,
                "narrative":              narrative,
                "classifier_model":       classifier_winner,
                "classifier_leaderboard": classifier_leaderboard,
                "classifier_skipped_reason": classifier_skipped_reason,
            },
            "feature_importance": fi_df,
        }


    # ── predict_point() ────────────────────────────────────────────────────────

    def predict_point(self, point_dict: dict, allow_partial: bool = True) -> dict:
        """
        Same imputation mechanism as ClusterEngine.predict_point() — see that
        method's docstring for the full rationale. Missing trained features
        are filled from training-time defaults (median/mode) by default
        rather than hard-failing, with a `defaulted_fields`/`warning` in the
        response when this happens. Pass allow_partial=False to restore the
        strict "error on any missing field" behavior.
        """
        import pandas as pd

        if self.clf is None or self.scaler is None:
            return {
                "error": (
                    f"No trained classifier available for label '{self.label_col}'. "
                    "Train one first by asking a classification question against labeled data."
                )
            }

        missing = [c for c in self.feature_cols if c not in point_dict]
        defaulted_fields = []

        if missing:
            if not allow_partial:
                return {"error": f"Missing required fields: {missing}"}

            point_dict = dict(point_dict)
            still_missing = []
            for col in missing:
                default_val = self.feature_defaults.get(col)
                if default_val is None:
                    still_missing.append(col)
                else:
                    point_dict[col] = default_val
                    defaulted_fields.append(col)

            if still_missing:
                return {"error": f"Missing required fields with no available default: {still_missing}"}

        input_df   = pd.DataFrame([point_dict])
        encoded_pt = self._encode_consistent(input_df)
        scaled_pt  = self.scaler.transform(encoded_pt)

        predicted_label = self.clf.predict(scaled_pt)[0]
        probabilities   = self.clf.predict_proba(scaled_pt)[0]
        classes         = self.clf.classes_

        confidence_scores = {
            cls: round(float(prob), 4)
            for cls, prob in zip(classes, probabilities)
        }
        match_confidence = round(float(max(probabilities)), 4)

        result = {
            "predicted_label":   predicted_label,
            "label_col":         self.label_col,
            "match_confidence":  match_confidence,
            "confidence_matrix": confidence_scores,
        }

        explanation, explanation_type = _explain_point(
            self.explainer, self.explainer_type, self.clf, scaled_pt,
            predicted_label, self.encoded_cols, self.feature_importance_global,
        )
        if explanation:
            result["explanation"] = explanation
            result["explanation_type"] = explanation_type

        if defaulted_fields:
            result["defaulted_fields"] = defaulted_fields
            result["warning"] = (
                f"These fields weren't specified and were defaulted to typical training values: "
                f"{', '.join(defaulted_fields)}. Treat this prediction with lower confidence, since "
                f"it's based on incomplete information."
            )

        return result

    # ── predict_point_from_text() ──────────────────────────────────────────────

    def predict_point_from_text(self, question: str) -> dict:
        """Extract features from natural language, then classify against the
        trained label. Mirrors ClusterEngine.predict_point_from_text()."""
        if self.clf is None or self.scaler is None:
            return {
                "error": (
                    f"I need to train a classifier for '{self.label_col}' on labeled data first. "
                    "Try asking a classification question before predicting individual records!"
                )
            }

        import json
        client = self._get_groq_client()
        if client is None:
            return {"error": "Groq API key required for natural language feature extraction."}

        feature_type_hints = []
        for col in self.feature_cols:
            if col in self.cat_mappings:
                opts = ", ".join(self.cat_mappings[col])
                feature_type_hints.append(f"'{col}' (categorical, one of: {opts})")
            else:
                feature_type_hints.append(f"'{col}' (numeric)")

        prompt = (
            "You are a strict data extraction tool.\n"
            f"Extract the attributes from this query: '{question}'.\n"
            f"Expected features: {', '.join(feature_type_hints)}.\n"
            "Rules:\n"
            "- For categorical features, return the EXACT string value.\n"
            "- For numeric features, return a number.\n"
            "- If a feature is missing from the query, return null.\n"
            "Return ONLY a valid JSON object with the feature names as keys. "
            "No markdown, no explanation."
        )

        try:
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            point_dict = json.loads(resp.choices[0].message.content.strip())
            point_dict = {k: v for k, v in point_dict.items() if v is not None}

            prediction = self.predict_point(point_dict)
            if "error" not in prediction:
                prediction["extracted_features"] = point_dict
            return prediction

        except Exception as e:
            return {"error": f"Failed to extract features from text: {str(e)}"}


# ── Investigation Engine ─────────────────────────────────────────────────────
# Chains TrendEngine + AnomalyEngine + a binary "problem-period vs baseline"
# classifier tournament to answer compound "why did X happen" questions in one
# pass. Reuses _run_classifier_tournament — the same tournament that already
# picks the best forecast/cluster/classification model — treating "is this
# row from the flagged period" as the label, so its feature importances
# become a model-backed signal for what actually differs between the two
# periods, not just naive before/after averages.

_MONTH_NAMES = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}


class InvestigationEngine:
    """
    Answers compound "why did X happen" questions by chaining existing engines
    rather than requiring the analyst to run trend, then anomaly, then manually
    eyeball what else changed, as three separate queries.
    """

    def run(self, df, date_col, value_col, question, feature_cols=None, group_col=None):
        if date_col not in df.columns or value_col not in df.columns:
            return {"error": f"Columns '{date_col}' or '{value_col}' not found."}

        trend_result = TrendEngine().run(df, date_col, value_col, group_col)
        if "error" in trend_result:
            return {"error": trend_result["error"]}
        trend_meta = trend_result.get("meta", {})
        trend_df   = trend_result["data"]
        if group_col:  # TrendEngine returns per-group meta dicts when grouped — flatten to first group for now
            first_key = next(iter(trend_meta)) if trend_meta else None
            trend_meta = trend_meta.get(first_key, {}) if first_key else {}

        anomaly_result = AnomalyEngine().run(df, value_col, date_col, group_col, contamination="auto")
        anomaly_meta = anomaly_result.get("meta", {}) if "error" not in anomaly_result else {}
        anomalies_df = anomaly_result.get("anomalies") if "error" not in anomaly_result else pd.DataFrame()

        period_info = self._identify_period_of_interest(question, trend_df, date_col)
        if period_info is None:
            return {
                "error": (
                    "Could not identify a specific period to investigate — mention a month "
                    "(e.g. 'in March') or ask about the biggest change, and I'll find the "
                    "largest swing in the data automatically."
                )
            }

        what_changed = None
        if feature_cols:
            what_changed = self._what_changed(df, date_col, feature_cols, period_info)

        return {
            "trend_meta":    trend_meta,
            "anomaly_meta":  anomaly_meta,
            "period_info":   period_info,
            "what_changed":  what_changed,
            "trend_data":    trend_df,
            "anomalies_data": anomalies_df,
        }

    def _identify_period_of_interest(self, question: str, trend_df: "pd.DataFrame", date_col: str) -> Optional[Dict]:
        q_lower = question.lower()

        # 1. Explicit month mention in the question, matched against dates actually present
        dates = pd.to_datetime(trend_df[date_col], errors="coerce")
        for name, month_num in _MONTH_NAMES.items():
            if re.search(rf"\b{name}\b", q_lower):
                year_match = re.search(r"\b(20\d{2})\b", question)
                candidate_years = dates[dates.dt.month == month_num].dt.year.unique().tolist()
                if not candidate_years:
                    continue
                year = int(year_match.group(1)) if year_match and int(year_match.group(1)) in candidate_years else max(candidate_years)
                start = pd.Timestamp(year=year, month=month_num, day=1)
                end   = start + pd.offsets.MonthEnd(1)
                return {
                    "start": str(start.date()), "end": str(end.date()),
                    "label": f"{name.title()} {year}", "source": "explicit",
                }

        # 2. No explicit period mentioned — auto-detect the biggest swing (by
        # momentum, i.e. change in MoM% — an acceleration/deceleration, not
        # just the largest single value) or the most severe flagged spike.
        if "is_spike" in trend_df.columns and trend_df["is_spike"].any():
            idx = trend_df.loc[trend_df["is_spike"], "spike_zscore"].abs().idxmax()
        elif "momentum" in trend_df.columns and trend_df["momentum"].notna().any():
            idx = trend_df["momentum"].abs().idxmax()
        elif "mom_pct" in trend_df.columns and trend_df["mom_pct"].notna().any():
            idx = trend_df["mom_pct"].abs().idxmax()
        else:
            return None

        target_date = pd.to_datetime(trend_df.loc[idx, date_col])
        start = target_date.replace(day=1)
        end   = start + pd.offsets.MonthEnd(1)
        return {
            "start": str(start.date()), "end": str(end.date()),
            "label": target_date.strftime("%B %Y"), "source": "auto-detected (largest swing)",
        }

    def _what_changed(self, df: "pd.DataFrame", date_col: str, feature_cols: list, period_info: Dict) -> Optional[Dict]:
        d = df.copy()
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        start, end = pd.to_datetime(period_info["start"]), pd.to_datetime(period_info["end"])
        d["_is_period"] = (d[date_col] >= start) & (d[date_col] <= end)

        usable_feature_cols = [c for c in feature_cols if c in d.columns and c != date_col]
        if not usable_feature_cols:
            return None

        sub = d[usable_feature_cols + ["_is_period"]].dropna()
        if sub["_is_period"].sum() < 5 or (~sub["_is_period"]).sum() < 5:
            return None  # not enough rows on one side of the split to compare meaningfully

        numeric_cols = [c for c in usable_feature_cols if pd.api.types.is_numeric_dtype(sub[c])]
        cat_cols     = [c for c in usable_feature_cols if c not in numeric_cols]

        baseline = sub[~sub["_is_period"]]
        period   = sub[sub["_is_period"]]

        comparisons = []
        for col in numeric_cols:
            b_mean, p_mean = baseline[col].mean(), period[col].mean()
            if abs(b_mean) > 1e-9:
                pct_diff = (p_mean - b_mean) / abs(b_mean) * 100
                comparisons.append({
                    "feature": col, "type": "numeric",
                    "baseline_avg": round(float(b_mean), 2), "period_avg": round(float(p_mean), 2),
                    "pct_diff": round(float(pct_diff), 1),
                })

        for col in cat_cols:
            b_dist = baseline[col].value_counts(normalize=True)
            p_dist = period[col].value_counts(normalize=True)
            all_cats = set(b_dist.index) | set(p_dist.index)
            shifts = [(cat, p_dist.get(cat, 0) - b_dist.get(cat, 0)) for cat in all_cats]
            shifts.sort(key=lambda x: abs(x[1]), reverse=True)
            if shifts:
                top_cat, shift = shifts[0]
                comparisons.append({
                    "feature": col, "type": "categorical", "shifted_category": str(top_cat),
                    "baseline_share": round(float(b_dist.get(top_cat, 0)) * 100, 1),
                    "period_share":   round(float(p_dist.get(top_cat, 0)) * 100, 1),
                })

        comparisons.sort(
            key=lambda c: abs(c.get("pct_diff", c.get("period_share", 0) - c.get("baseline_share", 0))),
            reverse=True,
        )

        # Model-backed signal: reuse the classifier tournament, treating
        # "is this row in the flagged period" as the label.
        model_importance = None
        try:
            X_df = sub[usable_feature_cols].copy()
            for col in cat_cols:
                X_df[col] = X_df[col].astype(str)
            X_encoded = pd.get_dummies(X_df, columns=cat_cols, drop_first=False) if cat_cols else X_df
            from sklearn.preprocessing import StandardScaler
            X_scaled = StandardScaler().fit_transform(X_encoded)
            y = sub["_is_period"].values

            tournament = _run_classifier_tournament(X_scaled, y)
            model = tournament["model"]
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                coef = model.coef_
                importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
            else:
                importances = None

            if importances is not None:
                pairs = sorted(zip(X_encoded.columns.tolist(), importances.tolist()), key=lambda x: x[1], reverse=True)
                model_importance = [{"feature": f, "importance": round(v, 4)} for f, v in pairs[:5]]
        except Exception:
            pass

        return {
            "comparisons":     comparisons[:6],
            "model_importance": model_importance,
            "period_rows":     int(period["_is_period"].shape[0] if hasattr(period["_is_period"], "shape") else len(period)),
            "baseline_rows":   int(len(baseline)),
        }