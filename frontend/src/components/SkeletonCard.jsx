import React from "react"

export default function SkeletonCard() {
  return (
    <div className="skeleton-card">
      <div className="skeleton-header">
        <div className="skeleton-badge" />
        <div className="skeleton-line long" />
      </div>
      <div className="skeleton-body">
        <div className="skeleton-line medium" />
        <div className="skeleton-line short" />
      </div>
      <div className="skeleton-chart" />
    </div>
  )
}