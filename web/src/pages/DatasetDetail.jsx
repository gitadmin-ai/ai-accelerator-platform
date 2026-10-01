import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api } from "../api";
import { formatBytes, formatDateTime, sourceLabel } from "../utils";

export default function DatasetDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [dataset, setDataset] = useState(null);
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    setDataset(null);
    setPreview(null);
    setError(null);
    Promise.all([api.getDataset(id), api.previewDataset(id, 5)])
      .then(([d, p]) => { setDataset(d); setPreview(p); })
      .catch((err) => setError(String(err)));
  }, [id]);

  if (error) return <div className="page"><div className="error-banner">{error}</div></div>;
  if (!dataset) return <div className="page">Loading…</div>;

  async function handleDelete() {
    if (!window.confirm(`Delete dataset "${dataset.name}"? This cannot be undone.`)) return;
    try {
      await api.deleteDataset(dataset.id);
      navigate("/datasets");
    } catch (err) {
      setError(String(err));
    }
  }

  return (
    <div className="page">
      <Link to="/datasets" className="back-link">&larr; Datasets</Link>

      <div className="job-header">
        <div>
          <h1>{dataset.name}</h1>
          <p className="job-subtitle">
            {sourceLabel(dataset.source)} · {dataset.format.toUpperCase()} · {dataset.num_examples.toLocaleString()} examples
          </p>
        </div>
        <div className="job-header-actions">
          <button onClick={() => navigate(`/jobs/new?dataset=${dataset.id}`)}>Use for Training</button>
          <button className="danger-button" onClick={handleDelete}>Delete</button>
        </div>
      </div>

      <div className="detail-card">
        <h3>Overview</h3>
        <dl>
          <dt>Name</dt><dd>{dataset.name}</dd>
          <dt>Source</dt><dd>{sourceLabel(dataset.source)}{dataset.hf_repo ? ` (${dataset.hf_repo})` : ""}</dd>
          <dt>Format</dt><dd>{dataset.format.toUpperCase()}</dd>
          <dt>Size</dt><dd>{formatBytes(dataset.size_bytes)}</dd>
          <dt>Examples</dt><dd>{dataset.num_examples.toLocaleString()}</dd>
          <dt>Created</dt><dd>{formatDateTime(dataset.created_at)}</dd>
          <dt>Status</dt><dd><span className={`badge badge--${dataset.status}`}>{dataset.status}</span></dd>
        </dl>
      </div>

      <div className="detail-card">
        <h3>Preview</h3>
        {preview.length === 0 && <p className="empty-row">No examples to preview.</p>}
        {preview.map((ex, i) => (
          <div key={i} className="dataset-example">
            <div className="dataset-example-label">Example #{i + 1}</div>
            {"user" in ex ? (
              <>
                <div className="dataset-example-turn"><strong>User:</strong> {ex.user}</div>
                <div className="dataset-example-turn"><strong>Assistant:</strong> {ex.assistant}</div>
              </>
            ) : "text" in ex ? (
              <div className="dataset-example-turn">{ex.text}</div>
            ) : (
              <pre className="dataset-example-raw">{JSON.stringify(ex.raw, null, 2)}</pre>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
