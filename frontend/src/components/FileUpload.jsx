import React, { useState, useRef } from "react"
import axios from "axios"
import { API_BASE } from "../config"

export default function FileUpload({ sessionId, onLoaded }) {
  // idle | inspecting | picking | uploading | done | error
  const [status,        setStatus]        = useState("idle")
  const [fileName,      setFileName]      = useState("")
  const [sheets,        setSheets]        = useState([])   // all sheet names in the workbook
  const [selectedSheets, setSelectedSheets] = useState([]) // sheet names currently checked
  const inputRef = useRef()
  const fileRef  = useRef(null)   // holds the actual File object between inspect → confirm

  // Step 1: file picked/dropped — inspect it first so we know whether to
  // show a sheet picker (multi-sheet Excel) or just upload immediately
  // (CSV, or a single-sheet Excel — nothing to choose between).
  async function handleFile(file) {
    if (!file) return
    fileRef.current = file
    setFileName(file.name)
    setStatus("inspecting")

    const form = new FormData()
    form.append("file", file)

    try {
      const { data } = await axios.post(`${API_BASE}/inspect_file`, form)

      if (data.multi_sheet) {
        setSheets(data.sheets)
        setSelectedSheets(data.sheets)   // default: all sheets selected, same as old behavior
        setStatus("picking")
      } else {
        // CSV or single-sheet Excel — nothing to choose, load it directly.
        doUpload(file, null)
      }
    } catch (e) {
      setStatus("error")
      console.error(e)
    }
  }

  // Step 2: actually load the file — `sheets` is either null (load
  // everything, the old default behavior) or the list of sheet names the
  // user picked in the picker below.
  async function doUpload(file, sheetsToLoad) {
    setStatus("uploading")

    const form = new FormData()
    form.append("file", file)
    if (sheetsToLoad) {
      form.append("sheets", JSON.stringify(sheetsToLoad))
    }

    try {
      const { data } = await axios.post(`${API_BASE}/upload/${sessionId}`, form)
      setStatus("done")
      onLoaded(data, file.name)
    } catch (e) {
      setStatus("error")
      console.error(e)
    }
  }

  function toggleSheet(name) {
    setSelectedSheets(prev =>
      prev.includes(name) ? prev.filter(s => s !== name) : [...prev, name]
    )
  }

  function cancelPicking() {
    fileRef.current = null
    setSheets([])
    setSelectedSheets([])
    setStatus("idle")
  }

  function onDrop(e) {
    e.preventDefault()
    handleFile(e.dataTransfer.files[0])
  }

  const dropZoneClickable = status === "idle" || status === "done" || status === "error"

  return (
    <>
      <div
        className={`upload-zone ${status === "picking" ? "" : status}`}
        onDrop={onDrop}
        onDragOver={e => e.preventDefault()}
        onClick={() => dropZoneClickable && inputRef.current.click()}
      >
        <input
          ref={inputRef}
          id="file-upload-input"
          name="file-upload"
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

        {status === "inspecting" && (
          <>
            <div className="spinner" style={{ margin: "0 auto 8px" }} />
            <div className="upload-filename">Reading file…</div>
            <div className="upload-sub">{fileName}</div>
          </>
        )}

        {status === "picking" && (
          <>
            <span className="upload-icon">📑</span>
            <div className="upload-filename">{fileName}</div>
            <div className="upload-sub">Choose sheets below</div>
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

      {/* Sheet picker — only shown for a multi-sheet Excel file, right after
          inspection and before the actual /upload call. Reuses the same
          panel styling as the MySQL "Advanced" panel for visual consistency. */}
      {status === "picking" && (
        <div className="mysql-connect-panel">
          <div style={{ fontSize: "12px", fontWeight: 600, color: "#334155", marginBottom: "2px" }}>
            This file has {sheets.length} sheets — pick which to load:
          </div>

          {sheets.map(sheet => (
            <label
              key={sheet}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "8px",
                fontSize: "13px",
                color: "#0F172A",
                padding: "6px 2px",
                cursor: "pointer",
              }}
            >
              <input
                id={`sheet-checkbox-${sheet}`}
                name={`sheet-checkbox-${sheet}`}
                type="checkbox"
                checked={selectedSheets.includes(sheet)}
                onChange={() => toggleSheet(sheet)}
              />
              {sheet}
            </label>
          ))}

          <div className="mysql-connect-actions">
            <button className="mysql-connect-cancel" onClick={cancelPicking}>
              Cancel
            </button>
            <button
              className="mysql-connect-submit"
              disabled={selectedSheets.length === 0}
              onClick={() => doUpload(fileRef.current, selectedSheets)}
            >
              Load {selectedSheets.length || ""} sheet{selectedSheets.length === 1 ? "" : "s"}
            </button>
          </div>
        </div>
      )}
    </>
  )
}