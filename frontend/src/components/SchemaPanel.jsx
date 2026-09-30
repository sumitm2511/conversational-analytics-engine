import React, { useState } from "react"
import axios from "axios"
import { API_BASE } from "../config"

// The `dtypes` map coming from the backend already holds the FINAL,
// classified category per column — "numeric" | "date" | "boolean" |
// "percent" | "text" — not a raw pandas dtype string like "int64" or
// "object". This component trusts the backend's category directly, for
// both the icon and the label, so the two can never disagree.
const TYPE_META = {
  numeric: { icon: "#",  cls: "numeric" },
  date:    { icon: "📅", cls: "date" },
  percent: { icon: "%",  cls: "percent" },
  // No dedicated CSS class exists yet for boolean, so this reuses "text"
  // for sizing/padding but overrides the color inline to keep it visually
  // distinct from plain text columns without needing a stylesheet change.
  boolean: { icon: "✓",  cls: "text", style: { background: "rgba(244,63,94,0.12)", color: "#F43F5E" } },
  text:    { icon: "T",  cls: "text" },
}

function getTypeMeta(category) {
  return TYPE_META[category] || TYPE_META.text
}

export default function SchemaPanel({ schema, sessionId, onLoaded }) {
  const [openTable,    setOpenTable]    = useState(schema?.active_table || null)
  const [removingName, setRemovingName] = useState(null)
  const tables = schema?.table_details || {}

  // Unloads a table — works for any table regardless of source (uploaded
  // sheet, MySQL table, or a prior combine result), same generic /table
  // endpoint MySQLConnect's own remove toggle uses. Reports the new schema
  // back through the same onLoaded callback FileUpload/MySQLConnect already
  // use after a load, so App.jsx handles "schema changed" uniformly
  // whether a table was just added or just removed.
  async function handleRemove(name, e) {
    e.stopPropagation()   // don't also trigger the row's open/close toggle
    if (removingName || !sessionId) return
    setRemovingName(name)
    try {
      const { data } = await axios.delete(`${API_BASE}/table/${sessionId}/${name}`)
      if (openTable === name) setOpenTable(data.active_table || null)
      onLoaded && onLoaded(data, data.active_table)
    } catch (err) {
      console.error(err)
    } finally {
      setRemovingName(null)
    }
  }

  return (
    <>
      <div className="sidebar-section">
        <div className="sidebar-section-title">Data</div>
        {Object.entries(tables).map(([name, info]) => (
          <div key={name}>
            <div
              className="sidebar-table-item"
              onClick={() =>
                setOpenTable(openTable === name ? null : name)
              }
            >
              <div className="sidebar-table-left" style={{ flex: 1, minWidth: 0 }}>
                <span>☰</span>
                <span
                  style={{
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                    minWidth: 0,
                  }}
                  title={name}
                >
                  {name}
                </span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "6px", flexShrink: 0 }}>
                <span className="sidebar-row-count">
                  {removingName === name ? "Removing…" : info.rows?.toLocaleString()}
                </span>
                <button
                  onClick={e => handleRemove(name, e)}
                  disabled={!!removingName}
                  title={`Remove ${name}`}
                  style={{
                    width: "18px",
                    height: "18px",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    border: "none",
                    background: "transparent",
                    color: "#94A3B8",
                    fontSize: "13px",
                    cursor: removingName ? "not-allowed" : "pointer",
                    borderRadius: "4px",
                    flexShrink: 0,
                  }}
                >
                  ✕
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {openTable && tables[openTable] && (
        <div className="sidebar-section">
          <div className="sidebar-section-title">Columns</div>
          <ul className="schema-cols">
            {tables[openTable].columns.map(col => {
              const category = tables[openTable].dtypes?.[col] || "text"
              const meta = getTypeMeta(category)
              return (
                <li key={col} className="schema-col-item">
                  <div className="schema-col-left">
                    <span className={`col-type-icon ${meta.cls}`} style={meta.style}>
                      {meta.icon}
                    </span>
                    <span className="col-name">{col}</span>
                  </div>
                  <span className="col-type-label">{category}</span>
                </li>
              )
            })}
          </ul>
        </div>
      )}
    </>
  )
}