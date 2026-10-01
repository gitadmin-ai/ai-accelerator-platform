import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api } from "../api";
import { formatBytes, formatDateTime } from "../utils";

const IN_PROGRESS = new Set(["pending", "downloading"]);

export default function ModelDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [model, setModel] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    function poll() {
      api.getModel(id)
        .then((m) => {
          if (cancelled) return;
          setModel(m);
          if (IN_PROGRESS.has(m.status)) setTimeout(poll, 2000);
        })
        .catch((err) => !cancelled && setError(String(err)));
    }
    poll();
    return () => { cancelled = true; };
  }, [id]);

  if (error) return <div className="page"><div className="error-banner">{error}</div></div>;
  if (!model) return <div className="page">Loading…</div>;

  async function handleDelete() {
    if (!window.confirm(`Delete model "${model.name}"? This cannot be undone.`)) return;
    try {
      await api.deleteModel(model.id);
      navigate("/models");
    } catch (err) {
      setError(String(err));
    }
  }

  return (
    <div className="page">
      <Link to="/models" className="back-link">&larr; Models</Link>

      <div className="job-header">
        <div>
          <h1>{model.name}</h1>
          <p className="job-subtitle">{model.repo_id}</p>
        </div>
        <div className="job-header-actions">
          <span className={`badge badge--${model.status}`}>{model.status}</span>
          {!IN_PROGRESS.has(model.status) && (
            <button className="danger-button" onClick={handleDelete}>Delete</button>
          )}
        </div>
      </div>

      {IN_PROGRESS.has(model.status) && (
        <div className="detail-card download-progress-card">
          <p className="download-progress-title">
            {model.status === "pending" ? "Preparing download…" : "Downloading model…"}
          </p>
          <div className="indeterminate-bar"><div className="indeterminate-bar-fill" /></div>
          <p className="wizard-hint">
            Exact progress isn't available from the download mechanism in use -- this is a real
            in-progress download, not a simulated one. The page will update automatically.
          </p>
        </div>
      )}

      {model.status === "ready" && (
        <div className="detail-card result-panel">
          <p className="result-artifact"><span className="result-artifact-check">✓</span> Model downloaded and ready</p>
          <div className="wizard-actions" style={{ borderTop: "none", paddingTop: 0, marginTop: 10 }}>
            <button onClick={() => navigate(`/jobs/new?model=${model.id}`)}>Create Training Job</button>
          </div>
        </div>
      )}

      {model.status === "error" && (
        <div className="detail-card failed-panel">
          <h3>Download Failed</h3>
          <p>{model.error}</p>
        </div>
      )}

      <div className="detail-card">
        <h3>Overview</h3>
        <dl>
          <dt>Name</dt><dd>{model.name}</dd>
          <dt>Hugging Face repository</dt><dd>{model.repo_id}</dd>
          <dt>Status</dt><dd><span className={`badge badge--${model.status}`}>{model.status}</span></dd>
          <dt>Size</dt><dd>{formatBytes(model.size_bytes)}</dd>
          <dt>Architecture</dt><dd>{model.architecture ?? "–"}</dd>
          <dt>Created</dt><dd>{formatDateTime(model.created_at)}</dd>
          <dt>Downloaded</dt><dd>{formatDateTime(model.completed_at)}</dd>
        </dl>
      </div>
    </div>
  );
}
