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

  // Strips characters that are invalid in filenames on Windows (and cause
  // silent failures/mangled names there) — \ / : * ? " < > | — since
  // `question` is arbitrary natural-language text and commonly contains
  // several of these ("What's the top region?" has both ' and ?).
  function safeFilename(base, ext) {
    const cleaned = (base || "export")
      .replace(/[\\/:*?"<>|]/g, "")
      .trim()
      .slice(0, 60) || "export"
    return `${cleaned}.${ext}`
  }

  function exportCSV() {
    if (!data || !columns) return
    const header = columns.join(",")
    const rows   = data.map(r =>
      r.map(c => `"${String(c ?? "").replace(/"/g, '""')}"`).join(",")
    )
    const csv  = [header, ...rows].join("\n")
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" })
    download(blob, safeFilename(question, "csv"))
    setOpen(false)
  }

  function exportJSON() {
    if (!data || !columns) return
    const json = JSON.stringify(
      data.map(r => Object.fromEntries(columns.map((c, i) => [c, r[i]]))),
      null, 2
    )
    const blob = new Blob([json], { type: "application/json" })
    download(blob, safeFilename(question, "json"))
    setOpen(false)
  }

  function exportExcel() {
    if (!data || !columns) return
    // Simple TSV that Excel opens natively
    const tsv  = [columns.join("\t"),
      ...data.map(r => r.map(c => c ?? "").join("\t"))
    ].join("\n")
    const blob = new Blob([tsv], { type: "text/tab-separated-values" })
    download(blob, safeFilename(question, "xls"))
    setOpen(false)
  }

  function exportChartPNG() {
    if (!chartRef?.current) return
    const plotlyDiv = chartRef.current.el
    if (!plotlyDiv) return
    import("plotly.js-dist").then(Plotly => {
      Plotly.downloadImage(plotlyDiv, {
        format: "png", width: 1200, height: 600,
        filename: safeFilename(question, "").replace(/\.$/, "") || "chart"
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
        filename: safeFilename(question, "").replace(/\.$/, "") || "chart"
      })
    })
    setOpen(false)
  }

  function download(blob, filename) {
    const url = URL.createObjectURL(blob)
    const a   = document.createElement("a")
    a.href     = url
    a.download = filename
    // Must be attached to the DOM for .click() to reliably trigger a
    // download in every browser — a detached anchor's click can silently
    // no-op, which is the most likely reason "Export" looked broken.
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    // Revoke on a short delay rather than immediately — revoking the
    // object URL synchronously can race with the browser actually starting
    // the download in some cases.
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  const hasTable = data && columns

  return (
    <div className="export-menu-wrap" ref={ref} style={{ display: "flex", gap: "4px" }}>
      {/* Single click on this icon = instant CSV download of the table
          currently shown, no menu required. */}
      <button
        className="card-icon-btn"
        onClick={exportCSV}
        disabled={!hasTable}
        title="Download table as CSV"
      >
        ↓
      </button>

      {/* Small separate caret opens the dropdown for the other formats
          (Excel, JSON, chart image) — kept available, just not required
          for the common case of "just give me the table". */}
      <button
        className="card-icon-btn"
        onClick={() => setOpen(p => !p)}
        title="More export options"
        style={{ fontSize: "10px" }}
      >
        ▾
      </button>

      {open && (
        <div className="export-dropdown">
          <div className="export-section-label">Table</div>
          <button className="export-item" onClick={exportCSV} disabled={!hasTable}>
            <span>📄</span> Download CSV
          </button>
          <button className="export-item" onClick={exportExcel} disabled={!hasTable}>
            <span>📊</span> Download Excel
          </button>
          <button className="export-item" onClick={exportJSON} disabled={!hasTable}>
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