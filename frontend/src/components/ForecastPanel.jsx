import React from "react"

export default function ForecastPanel({ result }) {
  if (!result || result.type === "sql") return null

  const isForecast   = result.type === "forecast"
  const isRegression = result.type === "regression"
  const isTrend      = result.type === "trend"

  return (
    <div className="forecast-panel">
      {/* Model info strip */}
      <div className="forecast-info-row">
        {isForecast && (
          <>
            <div className="forecast-info-pill">
              <span className="fip-label">Model</span>
              <span className="fip-value">
                {result.summary?.includes("Prophet") ? "Prophet" : "Linear Trend"}
              </span>
            </div>
            <div className="forecast-info-pill">
              <span className="fip-label">Horizon</span>
              <span className="fip-value">
                {result.rows?.filter(r =>
                  r[result.columns?.indexOf("is_forecast")] === true
                ).length || "—"} periods
              </span>
            </div>
            <div className="forecast-info-pill">
              <span className="fip-label">Confidence</span>
              <span className="fip-value">95%</span>
            </div>
          </>
        )}

        {isRegression && result.metrics && (
          <>
            <div className="forecast-info-pill">
              <span className="fip-label">Model</span>
              <span className="fip-value">Gradient Boosting</span>
            </div>
            <div className="forecast-info-pill">
              <span className="fip-label">R² Score</span>
              <span className={`fip-value ${
                result.metrics.r2_score >= 0.8 ? "good"
                : result.metrics.r2_score >= 0.6 ? "warn" : "bad"
              }`}>
                {result.metrics.r2_score}
              </span>
            </div>
            <div className="forecast-info-pill">
              <span className="fip-label">MAE</span>
              <span className="fip-value">{result.metrics.mae}</span>
            </div>
          </>
        )}

        {isTrend && (
          <div className="forecast-info-pill">
            <span className="fip-label">Analysis</span>
            <span className="fip-value">
              3-month & 6-month rolling average
            </span>
          </div>
        )}
      </div>

      {/* Feature importance bar chart */}
      {isRegression && result.feature_importance && (
        <div className="fi-bars">
          <div className="fi-title">Feature Importance</div>
          {result.feature_importance.rows.slice(0, 8).map((row, i) => {
            const feature = row[0]
            const score   = parseFloat(row[1])
            const pct     = Math.round(score * 100)
            return (
              <div key={i} className="fi-bar-row">
                <span className="fi-bar-label">{feature}</span>
                <div className="fi-bar-track">
                  <div
                    className="fi-bar-fill"
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <span className="fi-bar-value">{pct}%</span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}