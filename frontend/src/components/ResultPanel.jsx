import React, { useState, useRef, useEffect } from "react"
import ChartPanel  from "./ChartPanel"
import ExportMenu  from "./ExportMenu"
import axios       from "axios"
import { API_BASE } from "../config"

const INTENT_ICONS = { sql: "⊟", forecast: "↗", regression: "⎇", trend: "∿", anomaly: "⚡", summary: "◈", clustering: "◍", classification: "🏷" }
const INTENT_COLORS = { sql: "#6366F1", forecast: "#F59E0B", regression: "#10B981", trend: "#06B6D4", anomaly: "#F43F5E", summary: "#8B5CF6", clustering: "#8B5CF6", classification: "#0EA5E9" }

const FOLLOWUPS = {
  sql:            ["Show trend over time", "Forecast next 3 months", "Break down by region", "Sort descending"],
  forecast:       ["Extend forecast to 24 months", "Compare by segment", "Show historical only", "Detect anomalies"],
  trend:          ["Forecast next 6 months", "Compare by group", "Detect anomalies in this data", "Show raw numbers"],
  anomaly:        ["Show trend excluding anomalies", "Which dates had the most anomalies?", "Compare by category"],
  regression:     ["Show top feature drivers", "Compare actual vs predicted", "Run forecast instead"],
  summary:        ["Show sales trend", "Which region performs best?", "Detect anomalies"],
  clustering:     ["Segment by different features", "Which cluster would a new record belong to?", "Show cluster breakdown"],
  classification: ["Show confusion matrix breakdown", "Which features matter most?", "Classify a new record", "Show class distribution"],
}

// Structural styles for the Insights tab. Deliberately inline rather than
// added to the stylesheet: the previous Insights tab used classNames
// ("insights-full", "insight-section", "insight-section-title") that were
// never defined anywhere in App.css, which is why it rendered as flat,
// unstyled text. Everything possible below instead reuses classes that
// already ARE styled elsewhere (insight-strip, anomaly-flags, followup-chip)
// so the tab visually matches the rest of the card; these few objects only
// cover the genuinely new structural pieces (section labels, numbered
// finding cards, the next-questions grid) that have no existing equivalent.
const INSIGHTS_SECTION_TITLE_STYLE = {
  fontSize: "12px",
  fontWeight: 700,
  color: "#64748B",
  textTransform: "uppercase",
  letterSpacing: "0.05em",
  marginBottom: "10px",
}

const FINDING_CARD_STYLE = {
  display: "flex",
  gap: "12px",
  alignItems: "flex-start",
  background: "#F8FAFC",
  border: "1px solid #F1F5F9",
  borderRadius: "10px",
  padding: "12px 14px",
}

export default function ResultPanel({ result, sessionId, onFollowUp }) {
  const [activeTab,    setActiveTab]    = useState("Summary")
  const [chartData,    setChartData]    = useState(result.chart || null)
  const [chartLoading, setChartLoading] = useState(false)
  
  // Collapse State for UI Cleanliness
  const [isCollapsed,  setIsCollapsed]  = useState(false)
  const chartRef = useRef()

  useEffect(() => {
    setChartData(result.chart || null)
    setActiveTab("Summary") 
    setIsCollapsed(false) // Always open new queries
  }, [result])

  if (!result) return null

  if (result.error) return (
    <div className="result-card error-card">
      <div className="error-header">
        <span className="intent-badge" style={{ background: "#FEE2E2", color: "#F43F5E" }}>⚠ Error</span>
        <span className="result-question">{result.question}</span>
      </div>
      <div className="error-msg"><strong>Query failed:</strong> {result.error}</div>
    </div>
  )

  const type   = result.type || "sql"
  const color  = INTENT_COLORS[type] || "#6366F1"

  const insights    = result.insights || {}
  const kpis        = insights.kpis || {}
  const hasKpis     = Object.keys(kpis).length > 0
  const hasMetrics  = result.metrics && Object.keys(result.metrics).length > 0

  // Prefer the genuinely personalized, LLM-generated follow-ups (grounded
  // in this specific question/data — the same ones shown in the Insights
  // tab) for the always-visible footer row too, instead of the static
  // per-intent-type list, which showed the exact same 4 questions on every
  // single query of a given type regardless of what was actually asked.
  // Falls back to the static list only when the backend didn't return any
  // (e.g. an errored/empty insights response).
  const personalizedChips = Array.isArray(insights.next_questions) && insights.next_questions.length > 0
    ? insights.next_questions
    : null
  const chips = personalizedChips || FOLLOWUPS[type] || FOLLOWUPS.sql

  const tabs = ["Summary"]
  if (chartData)     tabs.push("Chart")
  if (result.rows?.length > 0) tabs.push("Table")
  if (result.sql)    tabs.push("SQL")
  if (result.insights) tabs.push("Insights")

  const hasAnomalyFlags   = insights.anomalies?.filter(Boolean).length > 0
  const hasInsightsContent =
    !!insights.summary ||
    (insights.insights?.length > 0) ||
    hasAnomalyFlags ||
    (insights.next_questions?.length > 0)

  async function handleRequestChart(chartType) {
    if (!sessionId) return
    setChartLoading(true)
    setActiveTab("Chart")
    try {
      const { data } = await axios.post(`${API_BASE}/query/${sessionId}`, { question: result.question, chart_type: chartType })
      if (data.chart) setChartData(data.chart)
    } catch (e) { console.error(e) } finally { setChartLoading(false) }
  }

  return (
    <div className={`result-card ${isCollapsed ? 'collapsed' : ''}`}>
      
      {/* ── HEADER (Always visible) ── */}
      <div className="result-header">
        <div className="result-header-left">
          <span className="intent-badge" style={{ background: color + "18", color }}>{INTENT_ICONS[type]} {type}</span>
          <span className="result-question" style={{ fontSize: "15px", fontWeight: "600" }}>{result.question}</span>
        </div>
        <div className="result-header-right" style={{ display: "flex", gap: "8px" }}>
          {!isCollapsed && <ExportMenu data={result.rows} columns={result.columns} chartRef={chartData ? chartRef : null} question={result.question} />}
          
          {/* Collapse/Expand Button */}
          <button 
            className="card-icon-btn" 
            onClick={() => setIsCollapsed(!isCollapsed)}
            title={isCollapsed ? "Expand panel" : "Collapse panel"}
            style={{ fontWeight: "bold" }}
          >
            {isCollapsed ? "▼" : "▲"}
          </button>
        </div>
      </div>

      {/* ── BODY (Hidden when collapsed) ── */}
      {!isCollapsed && (
        <>
          {(hasKpis || hasMetrics) && (
            <div className="kpi-strip">
              {Object.entries({ ...result.metrics, ...kpis }).slice(0, 6).map(([k, v]) => (
                <div key={k} className="kpi-pill"><span className="kpi-val">{v}</span><span className="kpi-key">{k}</span></div>
              ))}
            </div>
          )}
      
          <div className="tab-bar">
            {tabs.map(tab => (
              <button key={tab} className={`tab-btn ${activeTab === tab ? "active" : ""}`} style={activeTab === tab ? { borderBottomColor: color, color } : {}} onClick={() => setActiveTab(tab)}>
                {tab}
                {tab === "Table" && result.total_rows > 0 && <span className="tab-count">{result.total_rows.toLocaleString()}</span>}
              </button>
            ))}
          </div>
      
          <div className="tab-content">
            {activeTab === "Summary" && (
              <div className="summary-tab">
                {insights.summary && <div className="insight-strip"><span className="insight-icon">✦</span><span className="insight-text" style={{ fontSize: "14px", lineHeight: "1.6" }}>{insights.summary}</span></div>}
                {result.stat_narrative && <div className="insight-strip secondary"><span className="insight-icon">∿</span><span className="insight-text" style={{ fontSize: "14px", lineHeight: "1.6" }}>{result.stat_narrative}</span></div>}

                {/* Per-prediction explanation (SHAP or global-importance fallback) */}
                {result.explanation?.length > 0 && (
                  <div className="forecast-leaderboard" style={{ marginTop: "16px", background: "#f8fafc", padding: "16px", borderRadius: "8px", border: "1px solid #e2e8f0" }}>
                    <h4 style={{ margin: "0 0 12px 0", color: "#0f172a", fontSize: "14px" }}>
                      🔍 {result.explanation_type === "per_prediction" ? "Why This Prediction" : "Typical Drivers (general, not specific to this record)"}
                    </h4>
                    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px", textAlign: "left" }}>
                      <thead>
                        <tr style={{ borderBottom: "2px solid #cbd5e1", color: "#64748b" }}>
                          <th style={{ padding: "8px" }}>Feature</th>
                          <th style={{ padding: "8px" }}>{result.explanation_type === "per_prediction" ? "Contribution" : "Importance"}</th>
                          {result.explanation_type === "per_prediction" && <th style={{ padding: "8px" }}>Direction</th>}
                        </tr>
                      </thead>
                      <tbody>
                        {result.explanation.map((row, i) => (
                          <tr key={i} style={{ borderBottom: "1px solid #e2e8f0" }}>
                            <td style={{ padding: "10px 8px", color: "#0f172a" }}>{row.feature}</td>
                            <td style={{ padding: "10px 8px" }}>{row.contribution}</td>
                            {result.explanation_type === "per_prediction" && (
                              <td style={{ padding: "10px 8px", color: row.direction === "increased" ? "#059669" : "#DC2626" }}>
                                {row.direction === "increased" ? "↑ increased" : "↓ decreased"}
                              </td>
                            )}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
                {insights.insights?.length > 0 && (
                  <div className="insight-bullets">
                    {insights.insights.map((ins, i) => <div key={i} className="insight-bullet"><span className="bullet-dot" style={{ background: color }} /><span style={{ fontSize: "14px" }}>{ins}</span></div>)}
                  </div>
                )}
                {insights.anomalies?.length > 0 && (
                  <div className="anomaly-flags">
                    <div className="anomaly-flags-title">⚡ Flags</div>
                    {insights.anomalies.map((a, i) => <div key={i} className="anomaly-flag">{a}</div>)}
                  </div>
                )}
                
                {/* 🔥 NEW: FORECAST ML TOURNAMENT LEADERBOARD 🔥 */}
                {result.type === "forecast" && result.forecast_leaderboard?.length > 0 && (
                  <div className="forecast-leaderboard" style={{ marginTop: "24px", background: "#f8fafc", padding: "16px", borderRadius: "8px", border: "1px solid #e2e8f0" }}>
                    <h4 style={{ margin: "0 0 12px 0", color: "#0f172a", fontSize: "14px" }}>
                      🏆 Model Tournament — {result.primary_metric} used for selection
                    </h4>
                    
                    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px", textAlign: "left" }}>
                      <thead>
                        <tr style={{ borderBottom: "2px solid #cbd5e1", color: "#64748b" }}>
                          <th style={{ padding: "8px" }}>Model</th>
                          <th style={{ padding: "8px" }}>{result.primary_metric}</th>
                          <th style={{ padding: "8px" }}>MAE</th>
                          <th style={{ padding: "8px" }}>RMSE</th>
                          <th style={{ padding: "8px" }}>Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.forecast_leaderboard.map((row, i) => (
                          <tr key={i} style={{ 
                            borderBottom: "1px solid #e2e8f0",
                            background: row.selected === "✓" ? "rgba(16, 185, 129, 0.1)" : "transparent",
                            fontWeight: row.selected === "✓" ? "600" : "400"
                          }}>
                            <td style={{ padding: "10px 8px", color: "#0f172a" }}>{row.model}</td>
                            <td style={{ padding: "10px 8px" }}>{row[result.primary_metric]}</td>
                            <td style={{ padding: "10px 8px" }}>{row.MAE}</td>
                            <td style={{ padding: "10px 8px" }}>{row.RMSE}</td>
                            <td style={{ padding: "10px 8px", color: row.selected === "✓" ? "#059669" : "#94a3b8" }}>
                              {row.selected === "✓" ? "Winner" : "Eliminated"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    
                    <p style={{ margin: "12px 0 0 0", fontSize: "11px", color: "#64748b", fontStyle: "italic" }}>
                      *Metric selected: {result.primary_metric} 
                      {result.primary_metric === "MAE" && " (Data contains zeros/near-zeros)"}
                      {result.primary_metric === "RMSE" && " (High volatility detected in historical data)"}
                      {result.primary_metric === "SMAPE" && " (Standard scale-independent metric)"}
                    </p>
                  </div>
                )}

                {/* Clustering algorithm tournament — which algorithm won and why */}
                {result.type === "clustering" && result.clustering_leaderboard?.length > 0 && (
                  <div className="forecast-leaderboard" style={{ marginTop: "24px", background: "#f8fafc", padding: "16px", borderRadius: "8px", border: "1px solid #e2e8f0" }}>
                    <h4 style={{ margin: "0 0 12px 0", color: "#0f172a", fontSize: "14px" }}>
                      🧪 Clustering Algorithm Tournament — composite quality score
                    </h4>
                    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px", textAlign: "left" }}>
                      <thead>
                        <tr style={{ borderBottom: "2px solid #cbd5e1", color: "#64748b" }}>
                          <th style={{ padding: "8px" }}>Algorithm</th>
                          <th style={{ padding: "8px" }}>Silhouette</th>
                          <th style={{ padding: "8px" }}>Davies-Bouldin</th>
                          <th style={{ padding: "8px" }}>Calinski-Harabasz</th>
                          <th style={{ padding: "8px" }}>Clusters</th>
                          <th style={{ padding: "8px" }}>Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.clustering_leaderboard.map((row, i) => (
                          <tr key={i} style={{
                            borderBottom: "1px solid #e2e8f0",
                            background: row.selected === "✓" ? "rgba(139, 92, 246, 0.1)" : "transparent",
                            fontWeight: row.selected === "✓" ? "600" : "400"
                          }}>
                            <td style={{ padding: "10px 8px", color: "#0f172a" }}>{row.algorithm}</td>
                            <td style={{ padding: "10px 8px" }}>{row.silhouette}</td>
                            <td style={{ padding: "10px 8px" }}>{row.davies_bouldin}</td>
                            <td style={{ padding: "10px 8px" }}>{row.calinski_harabasz}</td>
                            <td style={{ padding: "10px 8px" }}>{row.n_clusters}{row.noise_count > 0 ? ` (+${row.noise_count} noise)` : ""}</td>
                            <td style={{ padding: "10px 8px", color: row.selected === "✓" ? "#7C3AED" : "#94a3b8" }}>
                              {row.selected === "✓" ? "Winner" : "Considered"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <p style={{ margin: "12px 0 0 0", fontSize: "11px", color: "#64748b", fontStyle: "italic" }}>
                      Chosen by a composite of Silhouette (40%), Davies-Bouldin (30%), and Calinski-Harabasz (30%) — not a single metric alone.
                    </p>
                  </div>
                )}

                {/* Point-classifier tournament (powers predict_point for this cluster model) */}
                {result.type === "clustering" && result.classifier_leaderboard?.length > 0 && (
                  <div className="forecast-leaderboard" style={{ marginTop: "16px", background: "#f8fafc", padding: "16px", borderRadius: "8px", border: "1px solid #e2e8f0" }}>
                    <h4 style={{ margin: "0 0 12px 0", color: "#0f172a", fontSize: "14px" }}>
                      🎯 Point-Classifier Tournament — {result.classifier_model} selected
                    </h4>
                    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px", textAlign: "left" }}>
                      <thead>
                        <tr style={{ borderBottom: "2px solid #cbd5e1", color: "#64748b" }}>
                          <th style={{ padding: "8px" }}>Model</th>
                          <th style={{ padding: "8px" }}>F1 (weighted)</th>
                          <th style={{ padding: "8px" }}>Accuracy</th>
                          <th style={{ padding: "8px" }}>Log-Loss</th>
                          <th style={{ padding: "8px" }}>Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.classifier_leaderboard.map((row, i) => (
                          <tr key={i} style={{
                            borderBottom: "1px solid #e2e8f0",
                            background: row.selected === "✓" ? "rgba(14, 165, 233, 0.1)" : "transparent",
                            fontWeight: row.selected === "✓" ? "600" : "400"
                          }}>
                            <td style={{ padding: "10px 8px", color: "#0f172a" }}>{row.model}</td>
                            <td style={{ padding: "10px 8px" }}>{row.f1_weighted}</td>
                            <td style={{ padding: "10px 8px" }}>{row.accuracy}</td>
                            <td style={{ padding: "10px 8px" }}>{row.log_loss ?? "—"}</td>
                            <td style={{ padding: "10px 8px", color: row.selected === "✓" ? "#0369A1" : "#94a3b8" }}>
                              {row.selected === "✓" ? "Winner" : "Considered"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <p style={{ margin: "12px 0 0 0", fontSize: "11px", color: "#64748b", fontStyle: "italic" }}>
                      Lower log-loss means better-calibrated confidence scores — a model can have decent accuracy but still be overconfident or underconfident.
                    </p>
                  </div>
                )}

                {/* Point 10: when the tournament couldn't run (e.g. too few samples in
                    the smallest class/cluster) it falls back to a single default model
                    silently unless we explain why — this replaces what would otherwise
                    just be an empty leaderboard with no context. */}
                {result.type === "clustering" && !(result.classifier_leaderboard?.length > 0) && result.classifier_skipped_reason && (
                  <div className="forecast-leaderboard" style={{ marginTop: "16px", background: "#FFFBEB", padding: "14px 16px", borderRadius: "8px", border: "1px solid #FDE68A" }}>
                    <div style={{ fontSize: "13px", color: "#92400E", display: "flex", gap: "8px", alignItems: "flex-start" }}>
                      <span>⚠️</span>
                      <span>
                        <strong>Model comparison skipped for the point-classifier: </strong>
                        {result.classifier_skipped_reason} A single default model ({result.classifier_model || "Random Forest"}) was used instead.
                      </span>
                    </div>
                  </div>
                )}

                {/* Supervised classification model tournament */}
                {result.type === "classification" && result.classifier_leaderboard?.length > 0 && (
                  <div className="forecast-leaderboard" style={{ marginTop: "24px", background: "#f8fafc", padding: "16px", borderRadius: "8px", border: "1px solid #e2e8f0" }}>
                    <h4 style={{ margin: "0 0 12px 0", color: "#0f172a", fontSize: "14px" }}>
                      🎯 Classifier Tournament — {result.metrics?.Model} selected
                    </h4>
                    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px", textAlign: "left" }}>
                      <thead>
                        <tr style={{ borderBottom: "2px solid #cbd5e1", color: "#64748b" }}>
                          <th style={{ padding: "8px" }}>Model</th>
                          <th style={{ padding: "8px" }}>F1 (weighted)</th>
                          <th style={{ padding: "8px" }}>Accuracy</th>
                          <th style={{ padding: "8px" }}>Log-Loss</th>
                          <th style={{ padding: "8px" }}>Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.classifier_leaderboard.map((row, i) => (
                          <tr key={i} style={{
                            borderBottom: "1px solid #e2e8f0",
                            background: row.selected === "✓" ? "rgba(14, 165, 233, 0.1)" : "transparent",
                            fontWeight: row.selected === "✓" ? "600" : "400"
                          }}>
                            <td style={{ padding: "10px 8px", color: "#0f172a" }}>{row.model}</td>
                            <td style={{ padding: "10px 8px" }}>{row.f1_weighted}</td>
                            <td style={{ padding: "10px 8px" }}>{row.accuracy}</td>
                            <td style={{ padding: "10px 8px" }}>{row.log_loss ?? "—"}</td>
                            <td style={{ padding: "10px 8px", color: row.selected === "✓" ? "#0369A1" : "#94a3b8" }}>
                              {row.selected === "✓" ? "Winner" : "Considered"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <p style={{ margin: "12px 0 0 0", fontSize: "11px", color: "#64748b", fontStyle: "italic" }}>
                      Selected by weighted F1 on held-out cross-validation, not just raw accuracy — more robust to class imbalance.
                    </p>
                  </div>
                )}

                {result.type === "classification" && !(result.classifier_leaderboard?.length > 0) && result.classifier_skipped_reason && (
                  <div className="forecast-leaderboard" style={{ marginTop: "24px", background: "#FFFBEB", padding: "14px 16px", borderRadius: "8px", border: "1px solid #FDE68A" }}>
                    <div style={{ fontSize: "13px", color: "#92400E", display: "flex", gap: "8px", alignItems: "flex-start" }}>
                      <span>⚠️</span>
                      <span>
                        <strong>Model comparison skipped: </strong>
                        {result.classifier_skipped_reason} A single default model ({result.metrics?.Model || "Random Forest"}) was used instead —
                        treat this result's accuracy with extra caution given the limited data.
                      </span>
                    </div>
                  </div>
                )}

                {/* Classification: class distribution, shown alongside the confusion-matrix chart on the Chart tab */}
                {result.type === "classification" && result.label_distribution && (
                  <div className="forecast-leaderboard" style={{ marginTop: "24px", background: "#f8fafc", padding: "16px", borderRadius: "8px", border: "1px solid #e2e8f0" }}>
                    <h4 style={{ margin: "0 0 12px 0", color: "#0f172a", fontSize: "14px" }}>
                      🏷 Class Distribution — {result.metrics?.["Label Column"]}
                    </h4>
                    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px", textAlign: "left" }}>
                      <thead>
                        <tr style={{ borderBottom: "2px solid #cbd5e1", color: "#64748b" }}>
                          <th style={{ padding: "8px" }}>Class</th>
                          <th style={{ padding: "8px" }}>Row Count</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(result.label_distribution).map(([cls, count]) => (
                          <tr key={cls} style={{ borderBottom: "1px solid #e2e8f0" }}>
                            <td style={{ padding: "10px 8px", color: "#0f172a" }}>{cls}</td>
                            <td style={{ padding: "10px 8px" }}>{count.toLocaleString()}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
                
                {result.rows?.length > 0 && (
                  <>
                    <div className="preview-label" style={{ marginTop: "24px" }}>Data preview</div>
                    <div className="table-wrap preview-table">
                      <table className="data-table">
                        <thead><tr>{result.columns.map(c => <th key={c}>{c}</th>)}</tr></thead>
                        <tbody>
                          {result.rows.slice(0, 5).map((row, i) => <tr key={i}>{row.map((cell, j) => <td key={j}>{cell ?? "—"}</td>)}</tr>)}
                        </tbody>
                      </table>
                    </div>
                    {result.rows.length > 5 && (
                      <div className="table-note">
                        Showing 5 of {result.total_rows?.toLocaleString()} rows — <button className="link-btn" onClick={() => setActiveTab("Table")}>View all →</button>
                      </div>
                    )}
                  </>
                )}
              </div>
            )}
      
            {activeTab === "Chart" && <ChartPanel ref={chartRef} chart={chartData} loading={chartLoading} onRequestChart={handleRequestChart} />}
      
            {activeTab === "Table" && result.rows && (
              <div>
                <div className="table-wrap full-table">
                  <table className="data-table">
                    <thead><tr>{result.columns.map(c => <th key={c}>{c}</th>)}</tr></thead>
                    <tbody>
                      {result.rows.map((row, i) => <tr key={i}>{row.map((cell, j) => <td key={j}>{cell ?? "—"}</td>)}</tr>)}
                    </tbody>
                  </table>
                </div>
                {result.total_rows > 500 && <div className="table-note">Showing 500 of {result.total_rows.toLocaleString()} rows</div>}
              </div>
            )}
      
            {activeTab === "SQL" && result.sql && (
              <div className="sql-panel">
                <div className="sql-panel-header"><span>Generated SQL</span><button className="copy-btn" onClick={() => navigator.clipboard.writeText(result.sql)}>Copy</button></div>
                <pre className="sql-block">{result.sql}</pre>
              </div>
            )}
      
            {activeTab === "Insights" && (
              <div style={{ padding: "20px 18px", display: "flex", flexDirection: "column", gap: "22px" }}>

                {/* Executive Summary — reuses the already-styled purple
                    insight-strip, same as the Summary tab, instead of the
                    old unstyled "insight-section" wrapper. */}
                {insights.summary && (
                  <div>
                    <div style={INSIGHTS_SECTION_TITLE_STYLE}>Executive Summary</div>
                    <div className="insight-strip">
                      <span className="insight-icon">✦</span>
                      <span className="insight-text" style={{ fontSize: "14px", lineHeight: "1.6" }}>{insights.summary}</span>
                    </div>
                  </div>
                )}

                {/* Key Findings — each finding gets its own numbered card
                    instead of a flat bullet, so a list of 3-5 findings
                    reads as distinct, scannable items rather than one
                    unbroken block of text. */}
                {insights.insights?.length > 0 && (
                  <div>
                    <div style={INSIGHTS_SECTION_TITLE_STYLE}>🔍 Key Findings</div>
                    <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                      {insights.insights.map((ins, i) => (
                        <div key={i} style={FINDING_CARD_STYLE}>
                          <span style={{
                            flexShrink: 0, width: "22px", height: "22px", borderRadius: "50%",
                            background: color + "18", color, fontSize: "11px", fontWeight: 700,
                            display: "flex", alignItems: "center", justifyContent: "center",
                          }}>
                            {i + 1}
                          </span>
                          <span style={{ fontSize: "14px", color: "#334155", lineHeight: "1.55" }}>{ins}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Anomalies & Flags — reuses the already-styled red
                    anomaly-flags card, same as the Summary tab. */}
                {hasAnomalyFlags && (
                  <div className="anomaly-flags" style={{ margin: 0 }}>
                    <div className="anomaly-flags-title">⚡ Anomalies &amp; Flags</div>
                    {insights.anomalies.map((a, i) => <div key={i} className="anomaly-flag">{a}</div>)}
                  </div>
                )}

                {/* Recommended Next Queries — same followup-chip pill style
                    used elsewhere in the app, but arranged in a responsive
                    grid here instead of a single scrolling row, since this
                    tab has the vertical space for a proper grid layout. */}
                {insights.next_questions?.length > 0 && (
                  <div>
                    <div style={INSIGHTS_SECTION_TITLE_STYLE}>💡 Recommended Next Queries</div>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))", gap: "8px" }}>
                      {insights.next_questions.map((q, i) => (
                        <button
                          key={i}
                          className="followup-chip"
                          style={{
                            display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: "8px",
                            width: "100%", textAlign: "left",
                            whiteSpace: "normal", wordBreak: "break-word", lineHeight: "1.4",
                            padding: "10px 14px",
                          }}
                          onClick={() => onFollowUp && onFollowUp(q)}
                        >
                          <span>{q}</span>
                          <span style={{ color: "#94A3B8", flexShrink: 0 }}>→</span>
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {!hasInsightsContent && (
                  <div style={{ textAlign: "center", padding: "40px 20px", color: "#94A3B8", fontSize: "13px" }}>
                    No additional insights available for this query.
                  </div>
                )}
              </div>
            )}
          </div>
      
          <div className="followup-row">
            <span className="followup-label">Ask next →</span>
            {chips.map(c => <button key={c} className="followup-chip" onClick={() => onFollowUp && onFollowUp(c)}>{c}</button>)}
          </div>
        </>
      )}
    </div>
  )
}