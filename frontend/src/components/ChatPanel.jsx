import React, { useState, useImperativeHandle, forwardRef } from "react"
import axios from "axios"
import { API_BASE } from "../config"

const SUGGESTIONS = [
  "show trend over time",
  "top 10 employees by utilization",
  "forecast next 3 months",
  "average utilization by domain",
  "show employees on bench",
  "which factors influence billability",
]

const ChatPanel = forwardRef(function ChatPanel(
  { sessionId, disabled, onResult, onLoading }, ref
) {
  const [input, setInput] = useState("")

  useImperativeHandle(ref, () => ({
    setQuestion: (q) => { setInput(q) },
    // Re-runs a question that previously paused for a join clarification —
    // either with the user's explicit choice (extra.join_override) or with
    // skip_join_clarification set (so the backend proceeds with its own
    // decision instead of pausing to ask again). Pushes the real result
    // into the feed the same way a normal submit() does; the resolved
    // clarification card stays in place above/below it as a record of the
    // choice, handled entirely by JoinClarification's own local state.
    resubmit: (question, extra) => submit(question, extra),
  }))

  async function submit(question, extra) {
    const q = (question || input).trim()
    if (!q) return

    onLoading(true)
    if (!extra) setInput("")   // only clear the input box for a fresh, typed submission

    try {
      const { data } = await axios.post(
        `${API_BASE}/query/${sessionId}`,
        { question: q, ...(extra || {}) }
      )
      onResult({ question: q, ...data })
    } catch (e) {
      onResult({
        question: q,
        error: e.response?.data?.detail || "Something went wrong"
      })
    } finally {
      onLoading(false)
    }
  }

  return (
    <div className="chat-panel">
      <div className="suggestions">
        {SUGGESTIONS.map(s => (
          <button
            key={s}
            className="suggestion-chip"
            disabled={disabled}
            onClick={() => submit(s)}
          >
            {s}
          </button>
        ))}
      </div>

      <div className="chat-bar">
        <input
          id="chat-question-input"
          name="question"
          className="chat-input"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === "Enter" && submit()}
          placeholder="Ask anything about your data..."
          disabled={disabled}
        />

        <button
          className="chat-submit"
          onClick={() => submit()}
          disabled={disabled || !input.trim()}
        >
          Run ➤
        </button>
      </div>
    </div>
  )
})

export default ChatPanel