import React, { useEffect, useState } from "react"

function AnimatedNumber({ target }) {
  const [value, setValue] = useState(0)

  useEffect(() => {
    if (!target) return
    const num = parseFloat(target)
    if (isNaN(num)) { setValue(target); return; }

    let start = 0
    const duration = 800
    const step = 16
    const increment = num / (duration / step)

    const timer = setInterval(() => {
      start += increment
      if (start >= num) {
        setValue(target)
        clearInterval(timer)
      } else {
        setValue(Math.floor(start))
      }
    }, step)

    return () => clearInterval(timer)
  }, [target])

  return <>{value}</>
}

export default function KPIBar({ schema }) {
  const tableDetails = schema ? Object.values(schema.table_details || {}) : []

  const totalRows   = tableDetails.reduce((sum, t) => sum + (t.rows || 0), 0)
  const totalCols   = tableDetails.reduce((sum, t) => sum + (t.columns?.length || 0), 0)
  const totalTables = tableDetails.length

  // Real data health, computed backend-side per table (nulls + duplicates).
  // Multi-sheet uploads get averaged across tables rather than only
  // reflecting the first/active sheet.
  const healthScores = tableDetails
    .map(t => t.health?.score)
    .filter(s => s != null)

  const health = healthScores.length
    ? Math.round((healthScores.reduce((a, b) => a + b, 0) / healthScores.length) * 10) / 10
    : null   // unknown, not a fake 100 — backend hasn't sent health data

  const healthClass = health == null ? ""
    : health >= 90 ? "health-good"
    : health >= 70 ? "health-warn" : "health-bad"

  const allWarnings = tableDetails.flatMap(t => t.health?.warnings || [])

  // Dismissing hides the warnings panel without touching the backend or the
  // underlying health data — it's purely "stop taking up space right now."
  // Keyed to the ACTUAL warning content (not just a boolean) so a genuinely
  // new problem showing up later — a fresh upload with different issues, or
  // the same session's warnings changing after a combine/remove — un-hides
  // the panel again instead of staying permanently dismissed for the rest
  // of the session. Re-dismissing the exact same set of warnings keeps it
  // hidden, so switching tabs/tables doesn't make it reappear for no reason.
  const warningsKey = allWarnings.join("|")
  const [dismissedKey, setDismissedKey] = useState(null)
  const isDismissed = dismissedKey === warningsKey && warningsKey !== ""

  if (!schema) return null

  const kpis = [
    { icon: "☰", label: "Total Rows",  value: totalRows.toLocaleString(), raw: totalRows },
    { icon: "⊞", label: "Columns",     value: totalCols, raw: totalCols },
    { icon: "⊟", label: "Tables",      value: totalTables, raw: totalTables },
    { icon: "⬡", label: "Data Health",
      value: health == null ? "—" : `${health}%`,
      raw: health,
      cls: healthClass },
  ]

  return (
    <>
      <div className="kpi-bar">
        {kpis.map((k) => (
          <div key={k.label} className="kpi-card">
            <div className="kpi-top">
              <span className="kpi-icon">{k.icon}</span>
              <span className="kpi-label">{k.label}</span>
            </div>
            <div className={`kpi-value ${k.cls || ""}`}>
              {k.label === "Data Health"
                ? k.value
                : <AnimatedNumber target={k.raw} />
              }
            </div>
          </div>
        ))}
      </div>

      {allWarnings.length > 0 && !isDismissed && (
        <div className="health-warnings-container">
          <div className="health-warnings-header">
            <div className="health-warnings-title">Data Health Warnings</div>
            <button
              className="health-warnings-dismiss"
              onClick={() => setDismissedKey(warningsKey)}
              title="Dismiss"
              aria-label="Dismiss data health warnings"
            >
              ✕
            </button>
          </div>
          {allWarnings.map((w, i) => (
            <div key={i} className="health-warning-item">⚠ {w}</div>
          ))}
        </div>
      )}
    </>
  )
}