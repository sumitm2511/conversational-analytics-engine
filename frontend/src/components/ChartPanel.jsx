import React, { useState, useRef, useEffect, forwardRef, useImperativeHandle } from "react"
import Plot from "react-plotly.js"

const CHART_TYPES = [
  { value: "bar",       label: "📊 Bar" },
  { value: "line",      label: "📈 Line" },
  { value: "area",      label: "〰 Area" },
  { value: "pie",       label: "🥧 Pie" },
  { value: "scatter",   label: "⊙ Scatter" },
  { value: "histogram", label: "▦ Histogram" },
]

const ChartPanel = forwardRef(function ChartPanel({ chart, onRequestChart, loading }, ref) {
  const [fullscreen,    setFullscreen]    = useState(false)
  const [selectedType,  setSelectedType]  = useState("auto")
  const plotRef = useRef()

  useImperativeHandle(ref, () => ({
    get el() { return plotRef.current?.el; }
  }))

  useEffect(() => {
    function onKey(e) { if (e.key === "Escape") setFullscreen(false) }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [])

  if (!chart && !loading) {
    return (
      <div className="chart-empty">
        <div className="chart-empty-icon">📊</div>
        <p>No visualization generated yet</p>
        <div className="chart-type-row">
          <select className="chart-type-select" value={selectedType} onChange={e => setSelectedType(e.target.value)}>
            <option value="auto">Auto Detect</option>
            {CHART_TYPES.map(t => <option key={t.value} value={t.value}>{t.label}</option>)}
          </select>
          <button className="generate-chart-btn" onClick={() => onRequestChart(selectedType)}>✨ Generate Chart</button>
        </div>
      </div>
    )
  }

  if (loading) {
    return (
      <div className="chart-loading">
        <div className="spinner" />
        <span>Generating visualization…</span>
      </div>
    )
  }

  // 🔥 GUARANTEED SPACIOUS CHART LAYOUT
  const plotLayout = {
    ...chart.layout,
    autosize:      true,
    paper_bgcolor: "transparent",
    plot_bgcolor:  "#FAFBFC",
    font:          { family: "Inter", size: 12, color: "#64748B" },
    margin:        { l: 60, r: 30, t: 40, b: 120 }, // Extra large bottom margin for labels
    colorway: ["#6366F1","#8B5CF6","#06B6D4","#10B981","#F59E0B","#F43F5E"],
  }

  const plotConfig = {
    displayModeBar:  true,
    displaylogo:     false,
    modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"],
    responsive:      true,
    toImageButtonOptions: { format: "png", filename: "chart", width: 1200, height: 600 }
  }

  return (
    <>
      <div className="chart-panel-wrap">
        <div className="chart-toolbar">
          <div className="chart-type-row">
            {CHART_TYPES.map(t => (
              <button key={t.value} className={`chart-type-pill ${chart.suggested_type === t.value ? "active" : ""}`} onClick={() => onRequestChart(t.value)}>
                {t.label}
              </button>
            ))}
          </div>
          <button className="fullscreen-btn" onClick={() => setFullscreen(true)} title="Fullscreen">⛶</button>
        </div>
        {/* 🔥 TALLER CHART SO YOU DON'T HAVE TO ZOOM */}
        <Plot ref={plotRef} data={chart.data} layout={plotLayout} config={plotConfig} useResizeHandler style={{ width: "100%", height: "450px", minHeight: "450px" }} />
      </div>

      {fullscreen && (
        <div className="fullscreen-overlay" onClick={() => setFullscreen(false)}>
          <div className="fullscreen-modal" onClick={e => e.stopPropagation()}>
            <div className="fullscreen-header">
              <span className="fullscreen-title">Chart View</span>
              <button className="fullscreen-close" onClick={() => setFullscreen(false)}>✕</button>
            </div>
            <Plot data={chart.data} layout={{ ...plotLayout, height: undefined, autosize: true }} config={plotConfig} useResizeHandler style={{ width: "100%", height: "calc(90vh - 80px)" }} />
          </div>
        </div>
      )}
    </>
  )
})

export default ChartPanel