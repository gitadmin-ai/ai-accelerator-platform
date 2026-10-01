import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { formatDateTime } from "../utils";

const RUNNING_STATUSES = new Set(["QUEUED", "RUNNING", "CHECKPOINTING", "EVALUATING"]);

export default function Dashboard() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([api.listDatasets(), api.listModels(), api.listJobs(), api.getActivity(10)])
      .then(([datasets, models, jobs, activity]) => setData({ datasets, models, jobs, activity }))
      .catch((err) => setError(String(err)));
  }, []);

  if (error) return <div className="page"><div className="error-banner">{error}</div></div>;

  const runningCount = data ? data.jobs.filter((j) => RUNNING_STATUSES.has(j.status)).length : null;

  return (
    <div className="page">
      <div className="dashboard-header">
        <h1>Welcome to NebulaAI</h1>
        <p className="dashboard-subtitle">Train, manage and deploy your models.</p>
      </div>

      <div className="stat-grid dashboard-stats">
        <SummaryCard label="Datasets" value={data?.datasets.length} to="/datasets" />
        <SummaryCard label="Models" value={data?.models.length} to="/models" />
        <SummaryCard label="Training Jobs" value={data?.jobs.length} to="/jobs" />
        <SummaryCard label="Running" value={runningCount} to="/jobs" />
      </div>

      <div className="dashboard-quick-actions">
        <Link to="/datasets/new" className="quick-action">+ Create Dataset</Link>
        <Link to="/models/new" className="quick-action">+ Add Model</Link>
        <Link to="/jobs/new" className="quick-action">+ Create Training Job</Link>
      </div>

      <div className="detail-card">
        <h3>Recent Activity</h3>
        {!data && <p className="empty-row">Loading…</p>}
        {data && data.activity.length === 0 && (
          <p className="empty-row">No activity yet -- create a dataset or model to get started.</p>
        )}
        {data && data.activity.length > 0 && (
          <ul className="activity-list">
            {data.activity.map((a, i) => (
              <li key={i} className="activity-item">
                <span className={`activity-dot activity-dot--${a.type}`} />
                <span className="activity-message">{a.message}</span>
                <span className="activity-time">{formatDateTime(a.timestamp)}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

function SummaryCard({ label, value, to }) {
  return (
    <Link to={to} className="stat-tile summary-card">
      <div className="stat-tile-label">{label}</div>
      <div className="stat-tile-value">{value == null ? "–" : value}</div>
    </Link>
  );
}
