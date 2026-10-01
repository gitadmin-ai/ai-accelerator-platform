import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api";
import EmptyState from "../components/EmptyState";
import { formatBytes, formatDateTime } from "../utils";

const IN_PROGRESS = new Set(["pending", "downloading"]);

export default function Models() {
  const [models, setModels] = useState(null);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  function refresh() {
    api.listModels().then(setModels).catch((err) => setError(String(err)));
  }

  useEffect(refresh, []);

  async function handleDelete(e, m) {
    e.stopPropagation();
    if (!window.confirm(`Delete model "${m.name}"? This cannot be undone.`)) return;
    try {
      await api.deleteModel(m.id);
      refresh();
    } catch (err) {
      setError(String(err));
    }
  }

  return (
    <div className="page">
      <div className="page-header">
        <h1>Models</h1>
        <button onClick={() => navigate("/models/new")}>+ Add Model</button>
      </div>

      {error && <div className="error-banner">{error}</div>}
      {!models && !error && <p>Loading…</p>}

      {models && models.length === 0 && (
        <EmptyState
          title="No models yet."
          body="Download a model from Hugging Face to get started."
          actionLabel="+ Add Model"
          actionTo="/models/new"
        />
      )}

      {models && models.length > 0 && (
        <table className="jobs-table">
          <thead>
            <tr>
              <th>Name</th><th>Source</th><th>Architecture</th><th>Size</th><th>Status</th><th>Created</th><th></th>
            </tr>
          </thead>
          <tbody>
            {models.map((m) => (
              <tr key={m.id} className="job-row" onClick={() => navigate(`/models/${m.id}`)}>
                <td><Link to={`/models/${m.id}`}>{m.name}</Link></td>
                <td>Hugging Face</td>
                <td>{m.architecture ?? "–"}</td>
                <td>{formatBytes(m.size_bytes)}</td>
                <td><span className={`badge badge--${m.status}`}>{m.status}</span></td>
                <td>{formatDateTime(m.created_at)}</td>
                <td>
                  {!IN_PROGRESS.has(m.status) && (
                    <button className="danger-link" onClick={(e) => handleDelete(e, m)}>Delete</button>
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
