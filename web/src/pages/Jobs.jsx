import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api";
import EmptyState from "../components/EmptyState";
import StatusBadge from "../components/StatusBadge";
import { formatDuration } from "../utils";

const DELETABLE = new Set(["SUBMITTED", "COMPLETED", "FAILED"]);

export default function Jobs() {
  const [jobs, setJobs] = useState([]);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  async function refreshJobs() {
    try {
      setJobs(await api.listJobs());
    } catch (err) {
      setError(String(err));
    }
  }

  useEffect(() => {
    refreshJobs();
    const interval = setInterval(refreshJobs, 3000);
    return () => clearInterval(interval);
  }, []);

  async function handleDelete(e, job) {
    e.stopPropagation();
    if (!window.confirm(`Delete job "${job.name}" and its runs/artifacts? This cannot be undone.`)) return;
    try {
      await api.deleteJob(job.job_id);
      refreshJobs();
    } catch (err) {
      setError(String(err));
    }
  }

  return (
    <div className="page">
      <div className="page-header">
        <h1>Fine-Tuning Jobs</h1>
        <button onClick={() => navigate("/jobs/new")}>+ Create Fine-Tuning Job</button>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {jobs.length === 0 ? (
        <EmptyState
          title="No training jobs yet."
          body="Select a model and dataset to create your first fine-tuning job."
          actionLabel="+ Create Training Job"
          actionTo="/jobs/new"
        />
      ) : (
        <table className="jobs-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Job ID</th>
              <th>Model</th>
              <th>Dataset</th>
              <th>Status</th>
              <th>Created</th>
              <th>Duration</th>
              <th>Storage</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((job) => (
              <tr key={job.job_id} onClick={() => navigate(`/jobs/${job.job_id}`)} className="job-row">
                <td><Link to={`/jobs/${job.job_id}`}>{job.name}</Link></td>
                <td className="mono-cell">{job.job_id}</td>
                <td>{job.model}</td>
                <td>{job.dataset}</td>
                <td><StatusBadge status={job.status} /></td>
                <td>{new Date(job.created_at * 1000).toLocaleString()}</td>
                <td>{formatDuration(job.duration_s)}</td>
                <td>{job.storage_backend}</td>
                <td>
                  {DELETABLE.has(job.status) && (
                    <button className="danger-link" onClick={(e) => handleDelete(e, job)}>Delete</button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
