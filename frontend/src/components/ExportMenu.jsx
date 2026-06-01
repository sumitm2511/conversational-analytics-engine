import React, { useState, useRef, useEffect } from "react"

export default function ExportMenu({ data, columns, chartRef, question }) {
  const [open, setOpen] = useState(false)
  const ref = useRef()

  useEffect(() => {
    function handleClick(e) {
      if (ref.current && !ref.current.contains(e.target))
        setOpen(false)
    }
    document.addEventListener("mousedown", handleClick)
    return () => document.removeEventListener("mousedown", handleClick)
  }, [])

  function exportCSV() {
    if (!data || !columns) return
    const header = columns.join(",")
    const rows   = data.map(r =>
      r.map(c => `"${String(c ?? "").replace(/"/g, '""')}"`).join(",")
    )
    const csv  = [header, ...rows].join("\n")
    const blob = new Blob([csv], { type: "text/csv" })
    download(blob, `${question?.slice(0, 30) || "export"}.csv`)
    setOpen(false)
  }

  function exportJSON() {
    if (!data || !columns) return
    const json = JSON.stringify(
      data.map(r => Object.fromEntries(columns.map((c, i) => [c, r[i]]))),
      null, 2
    )
    const blob = new Blob([json], { type: "application/json" })
    download(blob, `${question?.slice(0, 30) || "export"}.json`)
    setOpen(false)
  }

  function exportExcel() {
    if (!data || !columns) return
    // Simple TSV that Excel opens natively
    const tsv  = [columns.join("\t"),
      ...data.map(r => r.map(c => c ?? "").join("\t"))
    ].join("\n")
    const blob = new Blob([tsv], { type: "text/tab-separated-values" })
    download(blob, `${question?.slice(0, 30) || "export"}.xls`)
    setOpen(false)
  }

  function exportChartPNG() {
    if (!chartRef?.current) return
    const plotlyDiv = chartRef.current.el
    if (!plotlyDiv) return
    import("plotly.js-dist").then(Plotly => {
      Plotly.downloadImage(plotlyDiv, {
        format: "png", width: 1200, height: 600,
        filename: question?.slice(0, 30) || "chart"
      })
    })
    setOpen(false)
  }

  function exportChartSVG() {
    if (!chartRef?.current) return
    const plotlyDiv = chartRef.current.el
    if (!plotlyDiv) return
    import("plotly.js-dist").then(Plotly => {
      Plotly.downloadImage(plotlyDiv, {
        format: "svg", width: 1200, height: 600,
        filename: question?.slice(0, 30) || "chart"
      })
    })
    setOpen(false)
  }

  function download(blob, filename) {
    const url = URL.createObjectURL(blob)
    const a   = document.createElement("a")
    a.href    = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="export-menu-wrap" ref={ref}>
      <button
        className="card-icon-btn"
        onClick={() => setOpen(p => !p)}
        title="Export"
      >
        ↓
      </button>

      {open && (
        <div className="export-dropdown">
          <div className="export-section-label">Table</div>
          <button className="export-item" onClick={exportCSV}>
            <span>📄</span> Download CSV
          </button>
          <button className="export-item" onClick={exportExcel}>
            <span>📊</span> Download Excel
          </button>
          <button className="export-item" onClick={exportJSON}>
            <span>{ }</span> Download JSON
          </button>

          {chartRef && (
            <>
              <div className="export-divider" />
              <div className="export-section-label">Chart</div>
              <button className="export-item" onClick={exportChartPNG}>
                <span>🖼</span> Download PNG
              </button>
              <button className="export-item" onClick={exportChartSVG}>
                <span>✏</span> Download SVG
              </button>
            </>
          )}
        </div>
      )}
    </div>
  )
}