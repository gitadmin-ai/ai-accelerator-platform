import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api";
import EmptyState from "../components/EmptyState";
import { formatBytes, formatDateTime, sourceLabel } from "../utils";

export default function Datasets() {
  const [datasets, setDatasets] = useState(null);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  function refresh() {
    api.listDatasets().then(setDatasets).catch((err) => setError(String(err)));
  }

  useEffect(refresh, []);

  async function handleDelete(e, d) {
    e.stopPropagation();
    if (!window.confirm(`Delete dataset "${d.name}"? This cannot be undone.`)) return;
    try {
      await api.deleteDataset(d.id);
      refresh();
    } catch (err) {
      setError(String(err));
    }
  }

  return (
    <div className="page">
      <div className="page-header">
        <h1>Datasets</h1>
        <button onClick={() => navigate("/datasets/new")}>+ Create Dataset</button>
      </div>

      {error && <div className="error-banner">{error}</div>}
      {!datasets && !error && <p>Loading…</p>}

      {datasets && datasets.length === 0 && (
        <EmptyState
          title="No datasets yet."
          body="Create a dataset by uploading a local file or importing one from Hugging Face."
          actionLabel="+ Create Dataset"
          actionTo="/datasets/new"
        />
      )}

      {datasets && datasets.length > 0 && (
        <table className="jobs-table">
          <thead>
            <tr>
              <th>Name</th><th>Source</th><th>Format</th><th>Size</th>
              <th>Examples</th><th>Created</th><th>Status</th><th></th>
            </tr>
          </thead>
          <tbody>
            {datasets.map((d) => (
              <tr key={d.id} className="job-row" onClick={() => navigate(`/datasets/${d.id}`)}>
                <td><Link to={`/datasets/${d.id}`}>{d.name}</Link></td>
                <td>{sourceLabel(d.source)}</td>
                <td>{d.format.toUpperCase()}</td>
                <td>{formatBytes(d.size_bytes)}</td>
                <td>{d.num_examples.toLocaleString()} examples</td>
                <td>{formatDateTime(d.created_at)}</td>
                <td><span className={`badge badge--${d.status}`}>{d.status}</span></td>
                <td><button className="danger-link" onClick={(e) => handleDelete(e, d)}>Delete</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
