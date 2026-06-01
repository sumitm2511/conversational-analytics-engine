import React, { useState, useRef } from "react"
import axios from "axios"
import { API_BASE } from "../config"

export default function FileUpload({ sessionId, onLoaded }) {
  const [status,   setStatus]   = useState("idle")
  const [fileName, setFileName] = useState("")
  const inputRef = useRef()

  async function handleFile(file) {
    if (!file) return
    setFileName(file.name)
    setStatus("uploading")

    const form = new FormData()
    form.append("file", file)

    try {
      const { data } = await axios.post(
        `${API_BASE}/upload/${sessionId}`, form
      )
      setStatus("done")
      onLoaded(data, file.name)
    } catch (e) {
      setStatus("error")
      console.error(e)
    }
  }

  function onDrop(e) {
    e.preventDefault()
    handleFile(e.dataTransfer.files[0])
  }

  return (
    <div
      className={`upload-zone ${status}`}
      onDrop={onDrop}
      onDragOver={e => e.preventDefault()}
      onClick={() => status !== "uploading" && inputRef.current.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".csv,.xlsx,.xls"
        style={{ display: "none" }}
        onChange={e => handleFile(e.target.files[0])}
      />

      {status === "idle" && (
        <>
          <span className="upload-icon">📁</span>
          <div className="upload-filename">Drop file here</div>
          <div className="upload-sub">.csv  .xlsx  .xls</div>
        </>
      )}

      {status === "uploading" && (
        <>
          <div className="spinner" style={{ margin: "0 auto 8px" }} />
          <div className="upload-filename">Uploading…</div>
          <div className="upload-sub">{fileName}</div>
        </>
      )}

      {status === "done" && (
        <>
          <div className="upload-checkmark">✓</div>
          <div className="upload-filename">{fileName}</div>
          <div className="upload-sub">Click to replace</div>
        </>
      )}

      {status === "error" && (
        <>
          <span className="upload-icon">⚠️</span>
          <div className="upload-filename">Upload failed</div>
          <div className="upload-sub">Click to retry</div>
        </>
      )}
    </div>
  )
}