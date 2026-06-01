import React, { useState } from "react"

function getTypeIcon(dtype) {
  if (!dtype) return { icon: "T", cls: "text" }
  if (dtype.includes("int") || dtype.includes("float"))
    return { icon: "#", cls: "numeric" }
  if (dtype.includes("date") || dtype.includes("time"))
    return { icon: "📅", cls: "date" }
  if (dtype.includes("bool"))
    return { icon: "%", cls: "percent" }
  return { icon: "T", cls: "text" }
}

export default function SchemaPanel({ schema }) {
  const [openTable, setOpenTable] = useState(
    schema?.active_table || null
  )
  const tables = schema?.table_details || {}

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
              <div className="sidebar-table-left">
                <span>☰</span>
                {name}
              </div>
              <span className="sidebar-row-count">
                {info.rows?.toLocaleString()}
              </span>
            </div>
          </div>
        ))}
      </div>

      {openTable && tables[openTable] && (
        <div className="sidebar-section">
          <div className="sidebar-section-title">Columns</div>
          <ul className="schema-cols">
            {tables[openTable].columns.map(col => {
              const dtype = tables[openTable].dtypes?.[col] || ""
              const { icon, cls } = getTypeIcon(dtype)
              return (
                <li key={col} className="schema-col-item">
                  <div className="schema-col-left">
                    <span className={`col-type-icon ${cls}`}>
                      {icon}
                    </span>
                    <span className="col-name">{col}</span>
                  </div>
                  <span className="col-type-label">
                    {dtype.includes("int") ? "numeric"
                      : dtype.includes("float") ? "numeric"
                      : dtype.includes("date") ? "date"
                      : "text"}
                  </span>
                </li>
              )
            })}
          </ul>
        </div>
      )}
    </>
  )
}