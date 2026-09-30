import React, { useState } from "react"

// Inline "clarifying question" card — rendered in the results feed exactly
// where a normal ResultPanel would go, in place of one, whenever the
// backend detects a question needs two tables and pauses instead of
// silently committing to its own join guess. The AI's own decision
// (suggested_join_type) is pre-selected; the user can accept it, change
// it, change the join key, or skip entirely and let the AI's guess stand.
export default function JoinClarification({ data, onResolve }) {
  const {
    question, table_a, table_b,
    suggested_join_type, join_types, shared_columns, error,
  } = data

  const [joinType,     setJoinType]     = useState(suggested_join_type || "inner")
  const [selectedCols, setSelectedCols] = useState(shared_columns || [])
  const [resolved,     setResolved]     = useState(false)
  const [resolution,   setResolution]   = useState(null)   // { mode: "chosen"|"skipped", ... }

  const JOIN_LABELS = {
    inner: { label: "Inner",  hint: "Only rows that match in both" },
    left:  { label: "Left",   hint: `Keep every row from ${table_a}` },
    right: { label: "Right",  hint: `Keep every row from ${table_b}` },
    full:  { label: "Full",   hint: "Keep every row from both" },
  }

  function toggleCol(col) {
    setSelectedCols(prev => prev.includes(col) ? prev.filter(c => c !== col) : [...prev, col])
  }

  function handleUseThisJoin() {
    if (selectedCols.length === 0) return
    setResolved(true)
    setResolution({ mode: "chosen", joinType, columns: selectedCols })
    onResolve({
      join_override: { table_a, table_b, join_type: joinType, columns: selectedCols },
    })
  }

  function handleSkip() {
    setResolved(true)
    setResolution({ mode: "skipped" })
    onResolve({ skip_join_clarification: true })
  }

  return (
    <div className="result-card" style={{ borderLeft: "3px solid #6366F1" }}>
      <div className="result-header">
        <div className="result-header-left">
          <span className="intent-badge" style={{ background: "#EEF2FF", color: "#4338CA" }}>🔗 Join needed</span>
          <span className="result-question" style={{ fontSize: "15px", fontWeight: 600 }}>{question}</span>
        </div>
      </div>

      <div style={{ padding: "16px 18px" }}>
        {!resolved ? (
          <>
            <p style={{ fontSize: "13.5px", color: "#334155", margin: "0 0 14px 0" }}>
              This needs data from both <strong>{table_a}</strong> and <strong>{table_b}</strong>.
              Choose how they should join, or let the AI use its best guess.
            </p>

            {error && (
              <div className="mysql-connect-error" style={{ marginBottom: "12px" }}>{error}</div>
            )}

            <div style={{ fontSize: "11px", fontWeight: 700, color: "#64748B", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "8px" }}>
              Join type
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "8px", marginBottom: "16px" }}>
              {(join_types || ["inner", "left", "right", "full"]).map(jt => (
                <button
                  key={jt}
                  onClick={() => setJoinType(jt)}
                  style={{
                    border: `1.5px solid ${joinType === jt ? "#6366F1" : "#E2E8F0"}`,
                    background: joinType === jt ? "rgba(99,102,241,0.08)" : "white",
                    borderRadius: "10px", padding: "8px 6px", cursor: "pointer",
                    textAlign: "left", display: "flex", flexDirection: "column", gap: "2px",
                  }}
                >
                  <span style={{ fontSize: "13px", fontWeight: 700, color: "#0F172A" }}>
                    {JOIN_LABELS[jt]?.label || jt}
                    {jt === suggested_join_type && (
                      <span style={{ marginLeft: "6px", fontSize: "9.5px", fontWeight: 700, color: "#6366F1" }}>AI PICK</span>
                    )}
                  </span>
                  <span style={{ fontSize: "10px", color: "#94A3B8", lineHeight: 1.3 }}>{JOIN_LABELS[jt]?.hint}</span>
                </button>
              ))}
            </div>

            {shared_columns && shared_columns.length > 0 && (
              <>
                <div style={{ fontSize: "11px", fontWeight: 700, color: "#64748B", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "8px" }}>
                  Join key {shared_columns.length > 1 && "(composite)"}
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "8px", marginBottom: "18px" }}>
                  {shared_columns.map(col => (
                    <label
                      key={col}
                      style={{
                        display: "flex", alignItems: "center", gap: "6px",
                        border: "1px solid #E2E8F0", borderRadius: "999px",
                        padding: "5px 12px", fontSize: "12.5px", color: "#0F172A", cursor: "pointer",
                        background: selectedCols.includes(col) ? "#F5F3FF" : "white",
                      }}
                    >
                      <input
                        id={`join-col-${col}`}
                        name={`join-col-${col}`}
                        type="checkbox"
                        checked={selectedCols.includes(col)}
                        onChange={() => toggleCol(col)}
                      />
                      {col}
                    </label>
                  ))}
                </div>
              </>
            )}

            <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
              <button
                className="chat-submit"
                style={{ padding: "0 20px", height: "38px" }}
                disabled={selectedCols.length === 0}
                onClick={handleUseThisJoin}
              >
                Use this join
              </button>
              <button
                onClick={handleSkip}
                style={{ border: "none", background: "transparent", color: "#6366F1", fontSize: "13px", cursor: "pointer", fontWeight: 600 }}
              >
                Skip — use AI's best guess
              </button>
            </div>
          </>
        ) : (
          <p style={{ fontSize: "13px", color: "#64748B", margin: 0, fontStyle: "italic" }}>
            {resolution.mode === "chosen"
              ? <>✓ Joining <strong>{table_a}</strong> {(JOIN_LABELS[resolution.joinType]?.label || resolution.joinType).toUpperCase()} JOIN <strong>{table_b}</strong> on <strong>{resolution.columns.join(", ")}</strong> — answer above.</>
              : <>✓ Skipped — used the AI's own join choice. Answer above.</>
            }
          </p>
        )}
      </div>
    </div>
  )
}