import React, { useState, useEffect, useCallback } from "react"
import axios from "axios"
import { API_BASE } from "../config"

// Sidebar panel listing past questions asked this session, most recent
// first. Clicking one calls onSelect(question) — App.jsx wires this to the
// SAME handleFollowUp() already used by ResultPanel's "follow-up question"
// suggestions, which just populates the chat input via chatRef.setQuestion.
// Deliberately does NOT auto-submit: the point is "reuse this, or tweak it
// first" — re-typing a slightly different version of a question you just
// asked is exactly the friction this removes.
//
// refreshKey: pass something that changes after every new query (e.g.
// results.length) so this refetches and shows the just-run question
// without needing its own polling.
export default function QueryHistory({ sessionId, onSelect, refreshKey }) {
  const [history, setHistory] = useState([])
  const [open, setOpen] = useState(true)

  const fetchHistory = useCallback(() => {
    axios.get(`${API_BASE}/history/${sessionId}`)
      .then(({ data }) => setHistory(data.history || []))
      .catch(err => console.error(err))
  }, [sessionId])

  useEffect(() => { fetchHistory() }, [fetchHistory, refreshKey])

  if (history.length === 0) return null

  // Backend appends oldest→newest; show newest first, since the most
  // recent question is the one most likely to be re-run or tweaked.
  const recent = [...history].reverse()

  return (
    <div className="sidebar-section" style={{ paddingTop: 0 }}>
      <div
        className="sidebar-section-title"
        style={{ cursor: "pointer", display: "flex", justifyContent: "space-between", alignItems: "center" }}
        onClick={() => setOpen(o => !o)}
      >
        <span>Query history ({recent.length})</span>
        <span>{open ? "︿" : "﹀"}</span>
      </div>

      {open && recent.map((entry, i) => (
        <div
          key={i}
          className="sidebar-table-item"
          onClick={() => onSelect(entry.question)}
          title={entry.error ? `Failed: ${entry.error}` : "Click to reuse this question"}
        >
          <div
            className="sidebar-table-left"
            style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
          >
            {entry.error ? "⚠️ " : "🕘 "}{entry.question}
          </div>
        </div>
      ))}
    </div>
  )
}