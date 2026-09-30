import React, { useState } from "react"
import { API_BASE } from "../config"

export default function TopBar({
  sidebarOpen,
  onToggleSidebar,
  sources,          // [{ name, rows }]
  joinSuggestions,  // [{ table_a, table_b, columns: [...] }]
  combiningKey,     // "table_a|table_b" currently combining, or null
  onCombine,
  sessionId,
}) {
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const [exportOpen,   setExportOpen]   = useState(false)

  const totalRows = (sources || []).reduce((s, t) => s + (t.rows || 0), 0)

  // Full-table download — hits the new /table/.../export route directly via
  // a plain navigation, since it's just a server-generated file with a
  // Content-Disposition header, no client-side blob handling needed.
  function exportTable(name, format) {
    window.location.href = `${API_BASE}/table/${sessionId}/${name}/export?format=${format}`
    setExportOpen(false)
  }

  return (
    <div className="topbar">
      <div className="topbar-left">

        <button
          className="topbar-icon-btn"
          onClick={onToggleSidebar}
          title={sidebarOpen ? "Collapse Sidebar" : "Expand Sidebar"}
          style={{
            marginRight: "8px",
            border: "none",
            background: "rgba(255,255,255,0.06)",
            width: "32px",
            height: "32px"
          }}
        >
          ☰
        </button>

        <div className="topbar-logo">AI</div>
        <div className="topbar-title">
          Premium Analytics <span>AI Platform</span>
        </div>
      </div>

      <div className="topbar-right">
        {sources && sources.length > 0 && (
          <div className="topbar-data-wrap">
            <button
              className="topbar-data-badge"
              onClick={() => setDropdownOpen(open => !open)}
            >
              <span className="live-dot" />
              {sources.length} {sources.length === 1 ? "table" : "tables"} loaded
              <span className="topbar-data-chevron">{dropdownOpen ? "︿" : "﹀"}</span>
            </button>

            {dropdownOpen && (
              <div className="topbar-data-dropdown">
                <div className="topbar-data-dropdown-title">Loaded data</div>

                {sources.map(src => (
                  <div key={src.name} className="topbar-data-source-row">
                    <span>{src.name}</span>
                    <span className="topbar-data-source-rows">
                      {(src.rows || 0).toLocaleString()} rows
                    </span>
                  </div>
                ))}

                <div className="topbar-data-total">
                  {totalRows.toLocaleString()} rows total
                </div>

                {joinSuggestions && joinSuggestions.length > 0 && (
                  <>
                    <div className="topbar-data-dropdown-divider" />
                    <div className="topbar-data-dropdown-title">Combine tables</div>
                    {joinSuggestions.map(sugg => {
                      const key = `${sugg.table_a}|${sugg.table_b}`
                      const isCombining = combiningKey === key
                      return (
                        <div key={key} className="topbar-join-suggestion">
                          <div className="topbar-join-suggestion-text">
                            <div>
                              <strong>{sugg.table_a}</strong> + <strong>{sugg.table_b}</strong>
                            </div>
                            <div className="topbar-join-columns">
                              {sugg.columns.map(col => (
                                <span key={col} className="topbar-join-col-tag">{col}</span>
                              ))}
                            </div>
                          </div>
                          <button
                            className="topbar-join-btn"
                            onClick={() => onCombine(sugg)}
                            disabled={combiningKey !== null}
                          >
                            {isCombining ? "…" : "Combine"}
                          </button>
                        </div>
                      )
                    })}
                  </>
                )}
              </div>
            )}
          </div>
        )}

        {sources && sources.length > 0 && (
          <div className="export-menu-wrap">
            <button
              className="topbar-btn"
              onClick={() => setExportOpen(open => !open)}
            >
              Export ▾
            </button>

            {exportOpen && (
              <div className="export-dropdown">
                <div className="export-section-label">Export a full table</div>
                {sources.map(src => (
                  <div
                    key={src.name}
                    style={{
                      display: "flex", alignItems: "center", justifyContent: "space-between",
                      padding: "6px 10px", gap: "8px",
                    }}
                  >
                    <span style={{ fontSize: "13px", color: "#334155", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {src.name}
                    </span>
                    <div style={{ display: "flex", gap: "4px", flexShrink: 0 }}>
                      <button className="copy-btn" onClick={() => exportTable(src.name, "csv")}>CSV</button>
                      <button className="copy-btn" onClick={() => exportTable(src.name, "xlsx")}>XLSX</button>
                      <button className="copy-btn" onClick={() => exportTable(src.name, "json")}>JSON</button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        <button className="topbar-icon-btn">⚙</button>
        <div className="topbar-avatar">A</div>
      </div>
    </div>
  )
}