import React, { useState, useRef, useEffect } from "react"
import axios from "axios"
import { v4 as uuidv4 } from "uuid"
import { API_BASE } from "./config"
import TopBar        from "./components/TopBar"
import FileUpload     from "./components/FileUpload"
import MySQLConnect   from "./components/MySQLConnect"
import SnowflakeConnect from "./components/SnowflakeConnect"
import ChatPanel      from "./components/ChatPanel"
import ResultPanel    from "./components/ResultPanel"
import SchemaPanel    from "./components/SchemaPanel"
import KPIBar         from "./components/KPIBar"
import SkeletonCard   from "./components/SkeletonCard"
import JoinClarification from "./components/JoinClarification"

const SESSION_ID = uuidv4()

export default function App() {
  const [fileLoaded,    setFileLoaded]    = useState(false)
  const [schema,        setSchema]        = useState(null)
  const [results,       setResults]       = useState([])
  const [loading,       setLoading]       = useState(false)
  const [sidebarOpen,   setSidebarOpen]   = useState(true)   // ← collapsible
  const [sidebarPinned, setSidebarPinned] = useState(true)   // ← pin vs hover

  // Join suggestions live here (not inside MySQLConnect) because the top bar
  // — not the sidebar panel — is now what displays and acts on them, and
  // both FileUpload and MySQLConnect can trigger new suggestions becoming
  // relevant (loading a 2nd table from either source).
  const [joinSuggestions, setJoinSuggestions] = useState([])
  const [combiningKey,    setCombiningKey]    = useState(null)

  const chatRef  = useRef()
  const hoverRef = useRef()

  // Shared by FileUpload, MySQLConnect, and SchemaPanel (its remove
  // button) — all three funnel every schema-changing action (load OR
  // remove a table) through this same handler, since they all produce the
  // same response shape (tables/active_table/columns/table_details/overview).
  //
  // `info` can legitimately be null/empty now that tables can be REMOVED,
  // not just added (SchemaPanel's remove button, MySQLConnect's
  // click-to-unload toggle) — fileLoaded/schema are derived from whether
  // any tables actually remain, rather than fileLoaded always being forced
  // true, which previously would have left the chat panel enabled with no
  // data loaded after removing the last table.
  function handleFileLoaded(info, name) {
    const hasTables = !!(info && info.tables && info.tables.length > 0)
    setFileLoaded(hasTables)
    setSchema(hasTables ? info : null)
    setResults([])
  }

  // Re-check for combinable tables any time the loaded set changes — cheap
  // metadata-only call (see suggest_joins() on the backend), safe to fire
  // on every schema update rather than only after specific actions.
  useEffect(() => {
    if (!schema) return
    axios.get(`${API_BASE}/suggest_joins/${SESSION_ID}`)
      .then(({ data }) => setJoinSuggestions(data.suggestions || []))
      .catch(err => console.error(err))
  }, [schema])

  async function handleCombine(sugg) {
    const key = `${sugg.table_a}|${sugg.table_b}`
    setCombiningKey(key)
    try {
      const { data } = await axios.post(
        `${API_BASE}/combine_tables/${SESSION_ID}`,
        { table_a: sugg.table_a, table_b: sugg.table_b, columns: sugg.columns }
      )
      handleFileLoaded(data, data.active_table)
      setJoinSuggestions(prev => prev.filter(s =>
        !(s.table_a === sugg.table_a && s.table_b === sugg.table_b)
      ))
    } catch (err) {
      console.error(err)
    } finally {
      setCombiningKey(null)
    }
  }

  function handleResult(result) {
    setResults(prev => [result, ...prev])
  }

  function handleFollowUp(question) {
    if (chatRef.current) chatRef.current.setQuestion(question)
  }

  function handleSidebarMouseEnter() {
    if (!sidebarPinned) {
      clearTimeout(hoverRef.current)
      setSidebarOpen(true)
    }
  }

  function handleSidebarMouseLeave() {
    if (!sidebarPinned) {
      hoverRef.current = setTimeout(() => setSidebarOpen(false), 300)
    }
  }

  function togglePin() {
    const newPinned = !sidebarPinned
    setSidebarPinned(newPinned)
    if (!newPinned) setSidebarOpen(false)
    else setSidebarOpen(true)
  }

  // Derived, not separate state — schema.table_details is already the full,
  // cumulative set of everything loaded this session (files + MySQL), so
  // this always reflects reality without needing its own tracking.
  const sources = schema
    ? Object.entries(schema.table_details || {}).map(([name, t]) => ({ name, rows: t.rows || 0 }))
    : []

  // Also passed into MySQLConnect so its table browser can tell which rows
  // are currently loaded by checking against the ACTUAL schema, instead of
  // a separate local Set — that separate Set was the bug: removing a
  // MySQL-sourced table via SchemaPanel's ✕ never told MySQLConnect
  // anything happened, so its checkmark stayed stuck on indefinitely.
  const loadedTableNames = sources.map(s => s.name)

  return (
    <div className="app-shell">
      <TopBar
        sidebarOpen={sidebarOpen}
        onToggleSidebar={() => {
          setSidebarPinned(!sidebarPinned)
          setSidebarOpen(!sidebarOpen)
        }}
        sources={sources}
        joinSuggestions={joinSuggestions}
        combiningKey={combiningKey}
        onCombine={handleCombine}
        sessionId={SESSION_ID}
      />

      <div className="app-body">

        {/* ── Sidebar: rail (56px) or expanded (240px) ── */}
        <aside
          className={`sidebar ${sidebarOpen ? "sidebar-open" : "sidebar-collapsed"}`}
          onMouseEnter={handleSidebarMouseEnter}
          onMouseLeave={handleSidebarMouseLeave}
        >
          {/* Pin toggle */}
          <button
            className="sidebar-pin-btn"
            onClick={togglePin}
            title={sidebarPinned ? "Unpin sidebar" : "Pin sidebar"}
          >
            {sidebarPinned ? "⊣" : "⊢"}
          </button>

          {sidebarOpen && (
            <>
              <FileUpload
                sessionId={SESSION_ID}
                onLoaded={(info, name) => handleFileLoaded(info, name)}
              />

              <MySQLConnect
                sessionId={SESSION_ID}
                onLoaded={(info, name) => handleFileLoaded(info, name)}
                loadedTableNames={loadedTableNames}
              />

              <SnowflakeConnect
                sessionId={SESSION_ID}
                onLoaded={(info, name) => handleFileLoaded(info, name)}
                loadedTableNames={loadedTableNames}
              />

              {schema && (
                <SchemaPanel
                  schema={schema}
                  sessionId={SESSION_ID}
                  onLoaded={(info, name) => handleFileLoaded(info, name)}
                />
              )}
            </>
          )}

          {/* Collapsed icon rail */}
          {!sidebarOpen && (
            <div className="sidebar-rail">
              <button
                className="rail-icon"
                title="Upload file"
                onClick={() => { setSidebarOpen(true); setSidebarPinned(true) }}
              >
                📂
              </button>
              <button
                className="rail-icon"
                title="Connect to MySQL"
                onClick={() => { setSidebarOpen(true); setSidebarPinned(true) }}
              >
                🗄️
              </button>
              {schema && (
                <button
                  className="rail-icon"
                  title="View schema"
                  onClick={() => { setSidebarOpen(true); setSidebarPinned(true) }}
                >
                  🗂
                </button>
              )}
            </div>
          )}
        </aside>

        {/* ── Main content ── */}
        <main className={`main ${!sidebarOpen ? "main-expanded" : ""}`}>
          {schema && <KPIBar schema={schema} />}

          <div className="results-feed">
            {results.length === 0 && !loading && (
              <div className="empty-state">
                <div className="empty-state-icon">{fileLoaded ? "💬" : "📂"}</div>
                <h3>
                  {fileLoaded
                    ? "Ask anything about your data"
                    : "Upload a file or connect to MySQL to get started"}
                </h3>
                <p>
                  {fileLoaded
                    ? 'Try: "Show total sales by region" or "Forecast revenue for next 6 months"'
                    : "Supports CSV, XLSX, XLS files, or a live MySQL connection"}
                </p>
              </div>
            )}

            {loading && <SkeletonCard />}

            {results.map((r, i) => (
              r.type === "join_clarification" ? (
                <JoinClarification
                  key={i}
                  data={r}
                  onResolve={(extra) => chatRef.current.resubmit(r.question, extra)}
                />
              ) : (
                <ResultPanel
                  key={i}
                  result={r}
                  sessionId={SESSION_ID}
                  onFollowUp={handleFollowUp}
                />
              )
            ))}
          </div>

          <ChatPanel
            ref={chatRef}
            sessionId={SESSION_ID}
            disabled={!fileLoaded || loading}
            onResult={handleResult}
            onLoading={setLoading}
          />
        </main>
      </div>
    </div>
  )
}