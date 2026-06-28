import React from "react";

export default function TrialStats(props) {
    const runtime = props.runtime || {};
    const probes = runtime.probe_records || [];
    return (
        <div className="stats-grid">
            <div className="stat-card"><span>Score</span><strong>{runtime.score || 0}</strong></div>
            <div className="stat-card"><span>Time left</span><strong>{runtime.time_left || 0}s</strong></div>
            <div className="stat-card"><span>Steps</span><strong>{runtime.step_count || 0}</strong></div>
            <div className="stat-card"><span>Probes</span><strong>{probes.length}</strong></div>
        </div>
    );
}
