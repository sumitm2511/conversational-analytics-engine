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
  const [input,     setInput]     = useState("")
  const [chartType, setChartType] = useState("auto")

  useImperativeHandle(ref, () => ({
    setQuestion: (q) => { setInput(q) }
  }))

  async function submit(question) {
    const q = (question || input).trim()
    if (!q) return

    onLoading(true)
    setInput("")

    try {
      const { data } = await axios.post(
        `${API_BASE}/query/${sessionId}`,
        { question: q, chart_type: chartType }
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
        <select
          className="chart-select"
          value={chartType}
          onChange={e => setChartType(e.target.value)}
          disabled={disabled}
        >
          <option value="auto">⊞ Auto</option>
          <option value="bar">📊 Bar</option>
          <option value="line">📈 Line</option>
          <option value="area">〰 Area</option>
          <option value="pie">🥧 Pie</option>
          <option value="scatter">⊙ Scatter</option>
          <option value="histogram">▦ Histogram</option>
        </select>

        <input
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