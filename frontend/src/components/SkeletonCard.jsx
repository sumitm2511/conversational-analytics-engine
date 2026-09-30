import React from "react"

// Lightweight loading placeholder shown in place of a ResultPanel while a
// query is in flight. Shaped like a real result card (header + kpi strip +
// a few body lines) so the layout doesn't jump once the real result lands.
export default function SkeletonCard() {
  return (
    <div className="result-card skeleton-card" aria-hidden="true">
      <style>{`
        @keyframes skeleton-pulse {
          0%   { opacity: 0.55; }
          50%  { opacity: 1; }
          100% { opacity: 0.55; }
        }
        .skeleton-bar {
          background: #E2E8F0;
          border-radius: 6px;
          animation: skeleton-pulse 1.4s ease-in-out infinite;
        }
      `}</style>

      <div className="result-header">
        <div className="result-header-left" style={{ gap: "10px" }}>
          <div className="skeleton-bar" style={{ width: "70px", height: "22px", borderRadius: "999px" }} />
          <div className="skeleton-bar" style={{ width: "220px", height: "16px" }} />
        </div>
      </div>

      <div className="kpi-strip">
        {[1, 2, 3].map(i => (
          <div key={i} className="kpi-pill">
            <div className="skeleton-bar" style={{ width: "40px", height: "16px", marginBottom: "6px" }} />
            <div className="skeleton-bar" style={{ width: "60px", height: "10px" }} />
          </div>
        ))}
      </div>

      <div style={{ padding: "16px 18px", display: "flex", flexDirection: "column", gap: "10px" }}>
        <div className="skeleton-bar" style={{ width: "100%", height: "14px" }} />
        <div className="skeleton-bar" style={{ width: "90%", height: "14px" }} />
        <div className="skeleton-bar" style={{ width: "75%", height: "14px" }} />
      </div>
    </div>
  )
}