import React, { useState, useRef } from "react"
import { v4 as uuidv4 } from "uuid"
import TopBar       from "./components/TopBar"
import FileUpload   from "./components/FileUpload"
import ChatPanel    from "./components/ChatPanel"
import ResultPanel  from "./components/ResultPanel"
import SchemaPanel  from "./components/SchemaPanel"
import KPIBar       from "./components/KPIBar"
import SkeletonCard from "./components/SkeletonCard"

const SESSION_ID = uuidv4()

export default function App() {
  const [fileLoaded,    setFileLoaded]    = useState(false)
  const [schema,        setSchema]        = useState(null)
  const [fileName,      setFileName]      = useState("")
  const [results,       setResults]       = useState([])
  const [loading,       setLoading]       = useState(false)
  const [sidebarOpen,   setSidebarOpen]   = useState(true)   // ← collapsible
  const [sidebarPinned, setSidebarPinned] = useState(true)   // ← pin vs hover
  const chatRef  = useRef()
  const hoverRef = useRef()

  function handleFileLoaded(info, name) {
    setFileLoaded(true)
    setSchema(info)
    setFileName(name)
    setResults([])
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

  const totalRows = schema
    ? Object.values(schema.table_details || {}).reduce((s, t) => s + (t.rows || 0), 0)
    : 0

  return (
    <div className="app-shell">
      <TopBar
        fileInfo={fileName}
        rowCount={totalRows}
        onToggleSidebar={() => {
          setSidebarPinned(!sidebarPinned)
          setSidebarOpen(!sidebarOpen)
        }}
        sidebarOpen={sidebarOpen}
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
              {schema && <SchemaPanel schema={schema} />}
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
                    : "Upload a file to get started"}
                </h3>
                <p>
                  {fileLoaded
                    ? 'Try: "Show total sales by region" or "Forecast revenue for next 6 months"'
                    : "Supports CSV, XLSX and XLS files"}
                </p>
              </div>
            )}

            {loading && <SkeletonCard />}

            {results.map((r, i) => (
              <ResultPanel
                key={i}
                result={r}
                sessionId={SESSION_ID}
                onFollowUp={handleFollowUp}
              />
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