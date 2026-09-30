import React, { useState } from "react"
import axios from "axios"
import { API_BASE } from "../config"

export default function MySQLConnect({ sessionId, onLoaded, loadedTableNames }) {
  // idle | loading_tables | browsing | error
  const [status,     setStatus]     = useState("idle")
  const [tables,     setTables]     = useState([])
  const [loadingRow, setLoadingRow] = useState(null)   // table name currently loading/removing — transient UI only

  const [errorMsg,   setErrorMsg]   = useState("")

  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [customTable,  setCustomTable]  = useState("")
  const [customQuery,  setCustomQuery]  = useState("")

  // "Loaded" is derived from the actual schema (passed down from App.jsx as
  // loadedTableNames) rather than tracked as separate local state here.
  // Previously this component kept its own `loadedRows` Set, which could
  // desync from reality: removing a table via SchemaPanel's ✕ button (or
  // any path other than clicking it again in THIS list) never updated that
  // local Set, so the checkmark stayed stuck on indefinitely even after the
  // table was genuinely gone. Reading straight from loadedTableNames means
  // there's only one source of truth, so this can't happen again.
  const loadedSet = new Set(loadedTableNames || [])

  async function openBrowser() {
    setStatus("loading_tables")
    setErrorMsg("")
    try {
      const { data } = await axios.get(`${API_BASE}/connect/mysql/tables`)
      setTables(data.tables || [])
      setStatus("browsing")
    } catch (err) {
      setStatus("error")
      setErrorMsg(
        err?.response?.data?.detail || "Couldn't list MySQL tables. Check the connection."
      )
      console.error(err)
    }
  }

  // Click a table: loads it if not yet loaded, REMOVES it if it already is
  // — a true toggle. The browser panel stays open either way, so you can
  // load/unload several tables in a row.
  async function loadTable(table) {
    if (loadingRow) return

    if (loadedSet.has(table.name)) {
      setLoadingRow(table.name)
      setErrorMsg("")
      try {
        const { data } = await axios.delete(`${API_BASE}/table/${sessionId}/${table.name}`)
        onLoaded(data, data.active_table)
      } catch (err) {
        setErrorMsg(err?.response?.data?.detail || `Failed to remove ${table.name}.`)
        console.error(err)
      } finally {
        setLoadingRow(null)
      }
      return
    }

    setLoadingRow(table.name)
    setErrorMsg("")
    try {
      const { data } = await axios.post(
        `${API_BASE}/connect/mysql/${sessionId}`,
        { query: `SELECT * FROM \`${table.name}\``, table_name: table.name }
      )
      onLoaded(data, table.name)
    } catch (err) {
      setErrorMsg(err?.response?.data?.detail || `Failed to load ${table.name}.`)
      console.error(err)
    } finally {
      setLoadingRow(null)
    }
  }

  function handleAdvancedConnect(e) {
    e.preventDefault()
    if (!customQuery.trim()) return
    const name = customTable.trim() || "mysql_data"
    setLoadingRow(name)
    setErrorMsg("")
    axios.post(
      `${API_BASE}/connect/mysql/${sessionId}`,
      { query: customQuery.trim(), table_name: name }
    ).then(({ data }) => {
      onLoaded(data, data.active_table || name)
      setAdvancedOpen(false)
      setCustomTable("")
      setCustomQuery("")
    }).catch(err => {
      setErrorMsg(err?.response?.data?.detail || "Failed to connect. Check your query and try again.")
      console.error(err)
    }).finally(() => setLoadingRow(null))
  }

  function closePanel() {
    setStatus("idle")
    setAdvancedOpen(false)
    setErrorMsg("")
    // Deliberately NOT clearing `tables` — closing this panel just hides
    // the browser. Whatever's loaded stays loaded (visible in the top bar
    // and reflected here via loadedTableNames whenever it's reopened).
  }

  // ── Collapsed / idle / initial table-list loading ──
  if (status === "idle" || status === "loading_tables") {
    return (
      <div
        className={`upload-zone ${status === "loading_tables" ? "uploading" : ""}`}
        onClick={status === "loading_tables" ? undefined : openBrowser}
      >
        {status === "loading_tables" ? (
          <>
            <div className="spinner" style={{ margin: "0 auto 8px" }} />
            <div className="upload-filename">Loading tables…</div>
          </>
        ) : (
          <>
            <span className="upload-icon">🗄️</span>
            <div className="upload-filename">Connect to MySQL</div>
            <div className="upload-sub">Browse tables — no SQL needed</div>
          </>
        )}
      </div>
    )
  }

  // ── Browsing: plain table list, click to load, click a loaded one again to remove it ──
  return (
    <div className="sidebar-section" style={{ paddingTop: 0 }}>
      <div className="sidebar-section-title">MySQL tables</div>

      {status === "error" && (
        <div className="mysql-connect-error" style={{ marginBottom: 8 }}>{errorMsg}</div>
      )}

      {tables.length === 0 && status === "browsing" && (
        <div className="upload-sub" style={{ padding: "4px 2px" }}>No tables found.</div>
      )}

      {tables.map(table => {
        const isLoaded  = loadedSet.has(table.name)
        const isLoading = loadingRow === table.name
        return (
          <div
            key={table.name}
            className={`sidebar-table-item ${isLoaded ? "loaded" : ""}`}
            onClick={() => !loadingRow && loadTable(table)}
            title={isLoaded ? "Click to remove" : "Click to load"}
            style={loadingRow && !isLoading ? { opacity: 0.5, cursor: "not-allowed" } : undefined}
          >
            <div className="sidebar-table-left">
              {isLoaded ? "✓" : "🗄️"} {table.name}
            </div>
            <div className="sidebar-row-count">
              {isLoading
                ? (isLoaded ? "Removing…" : "Loading…")
                : `${table.row_count.toLocaleString()} rows`}
            </div>
          </div>
        )
      })}

      {errorMsg && status !== "error" && (
        <div className="mysql-connect-error" style={{ marginTop: 4, marginBottom: 4 }}>{errorMsg}</div>
      )}

      <div
        className="mysql-connect-advanced-toggle"
        onClick={() => setAdvancedOpen(open => !open)}
      >
        <span>Advanced: write custom SQL</span>
        <span>{advancedOpen ? "︿" : "﹀"}</span>
      </div>

      {advancedOpen && (
        <form className="mysql-connect-panel" onSubmit={handleAdvancedConnect}>
          <input
            id="mysql-table-name"
            name="mysql_table_name"
            className="mysql-connect-input"
            type="text"
            placeholder="Table name (optional)"
            value={customTable}
            onChange={e => setCustomTable(e.target.value)}
            disabled={!!loadingRow}
          />
          <textarea
            id="mysql-custom-query"
            name="mysql_query"
            className="mysql-connect-textarea"
            placeholder="SELECT * FROM bo_test.ACR_Dummy"
            value={customQuery}
            onChange={e => setCustomQuery(e.target.value)}
            disabled={!!loadingRow}
            rows={3}
          />
          <div className="mysql-connect-actions">
            <button
              type="button"
              className="mysql-connect-cancel"
              onClick={() => setAdvancedOpen(false)}
              disabled={!!loadingRow}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="mysql-connect-submit"
              disabled={!!loadingRow || !customQuery.trim()}
            >
              Connect
            </button>
          </div>
        </form>
      )}

      <div className="mysql-connect-close" onClick={closePanel}>
        Close
      </div>
    </div>
  )
}