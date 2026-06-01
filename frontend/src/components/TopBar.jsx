import React from "react"

export default function TopBar({ fileInfo, rowCount, onToggleSidebar, sidebarOpen }) {
  return (
    <div className="topbar">
      <div className="topbar-left">
        
        {/* 🔥 NEW: Sidebar Toggle Button */}
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
        {fileInfo && (
          <>
            <div className="topbar-file-info">
              <strong>File: {fileInfo}</strong>
            </div>
            <div className="live-badge">
              <span className="live-dot" />
              Live
            </div>
            {rowCount && (
              <div className="rows-badge">
                ⚡ {rowCount.toLocaleString()} rows
              </div>
            )}
          </>
        )}
        <button className="topbar-btn">Export ▾</button>
        <button className="topbar-icon-btn">⚙</button>
        <div className="topbar-avatar">A</div>
      </div>
    </div>
  )
}