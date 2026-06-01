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
  if (!schema) return null

  const totalRows    = Object.values(schema.table_details || {})
    .reduce((sum, t) => sum + (t.rows || 0), 0)
  const totalCols    = Object.values(schema.table_details || {})
    .reduce((sum, t) => sum + (t.columns?.length || 0), 0)
  const totalTables  = Object.keys(schema.table_details || {}).length

  // Calculate data health based on dtypes presence
  const allCols = Object.values(schema.table_details || {})
    .flatMap(t => t.columns || [])
  const health = allCols.length > 0
    ? Math.round((allCols.filter(c => c).length / allCols.length) * 100 * 10) / 10
    : 100

  const healthClass = health >= 90 ? "health-good"
    : health >= 70 ? "health-warn" : "health-bad"

  const kpis = [
    { icon: "☰", label: "Total Rows",  value: totalRows.toLocaleString(), raw: totalRows },
    { icon: "⊞", label: "Columns",     value: totalCols, raw: totalCols },
    { icon: "⊟", label: "Tables",      value: totalTables, raw: totalTables },
    { icon: "⬡", label: "Data Health", value: `${health}%`, raw: health,
      cls: healthClass },
  ]

  return (
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
            {k.label === "Data Health" ? "" : ""}
          </div>
        </div>
      ))}
    </div>
  )
}