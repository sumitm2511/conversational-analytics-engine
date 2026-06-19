import React, { useState, useRef, useEffect } from "react"
import ChartPanel  from "./ChartPanel"
import ExportMenu  from "./ExportMenu"
import axios       from "axios"
import { API_BASE } from "../config"

const INTENT_ICONS = { sql: "⊟", forecast: "↗", regression: "⎇", trend: "∿", anomaly: "⚡", summary: "◈" }
const INTENT_COLORS = { sql: "#6366F1", forecast: "#F59E0B", regression: "#10B981", trend: "#06B6D4", anomaly: "#F43F5E", summary: "#8B5CF6" }

const FOLLOWUPS = {
  sql:        ["Show trend over time", "Forecast next 3 months", "Break down by region", "Sort descending"],
  forecast:   ["Extend forecast to 24 months", "Compare by segment", "Show historical only", "Detect anomalies"],
  trend:      ["Forecast next 6 months", "Compare by group", "Detect anomalies in this data", "Show raw numbers"],
  anomaly:    ["Show trend excluding anomalies", "Which dates had the most anomalies?", "Compare by category"],
  regression: ["Show top feature drivers", "Compare actual vs predicted", "Run forecast instead"],
  summary:    ["Show sales trend", "Which region performs best?", "Detect anomalies"],
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
  const chips  = FOLLOWUPS[type] || FOLLOWUPS.sql
  const color  = INTENT_COLORS[type] || "#6366F1"

  const tabs = ["Summary"]
  if (chartData)     tabs.push("Chart")
  if (result.rows?.length > 0) tabs.push("Table")
  if (result.sql)    tabs.push("SQL")
  if (result.insights) tabs.push("Insights")

  const insights    = result.insights || {}
  const kpis        = insights.kpis || {}
  const hasKpis     = Object.keys(kpis).length > 0
  const hasMetrics  = result.metrics && Object.keys(result.metrics).length > 0

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
              <div className="insights-full">
                {insights.summary && <div className="insight-section"><div className="insight-section-title">Summary</div><p style={{ fontSize: "14px" }}>{insights.summary}</p></div>}
                {insights.insights?.length > 0 && <div className="insight-section"><div className="insight-section-title">Key Findings</div>{insights.insights.map((ins, i) => <div key={i} className="insight-bullet"><span className="bullet-dot" style={{ background: color }} /><span style={{ fontSize: "14px" }}>{ins}</span></div>)}</div>}
                {insights.anomalies?.filter(Boolean).length > 0 && <div className="insight-section"><div className="insight-section-title">⚡ Anomalies & Flags</div>{insights.anomalies.map((a, i) => <div key={i} className="anomaly-flag">{a}</div>)}</div>}
                {insights.next_questions?.length > 0 && <div className="insight-section"><div className="insight-section-title">Recommended Next Queries</div>{insights.next_questions.map((q, i) => <button key={i} className="followup-chip" onClick={() => onFollowUp && onFollowUp(q)}>{q}</button>)}</div>}
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