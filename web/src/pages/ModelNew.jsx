import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api";

const SOURCES = [
  { id: "huggingface", label: "Hugging Face", icon: "🤗", available: true },
  { id: "local", label: "Local model import", available: false },
  { id: "aws", label: "AWS", available: false },
  { id: "gcp", label: "GCP", available: false },
  { id: "azure", label: "Azure", available: false },
];

export default function ModelNew() {
  const navigate = useNavigate();
  const [source, setSource] = useState("huggingface");
  const [comingSoon, setComingSoon] = useState(null);
  const [name, setName] = useState("");
  const [repoId, setRepoId] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  function pickSource(s) {
    if (!s.available) {
      setComingSoon(`${s.label} integration is coming soon.`);
      return;
    }
    setComingSoon(null);
    setSource(s.id);
  }

  async function handleSubmit(e) {
    e.preventDefault();
    if (!name.trim() || !repoId.trim()) {
      setError("A model name and a Hugging Face repository are both required.");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const created = await api.downloadModel(name, repoId);
      navigate(`/models/${created.id}`);
    } catch (err) {
      setError(String(err));
      setSubmitting(false);
    }
  }

  return (
    <div className="page">
      <Link to="/models" className="back-link">&larr; Models</Link>
      <div className="page-header"><h1>Add Model</h1></div>

      <h3 className="wizard-subheading">Source</h3>
      {comingSoon && <div className="info-banner">{comingSoon}</div>}

      <div className="source-grid">
        {SOURCES.map((s) => (
          <button
            key={s.id}
            type="button"
            className={
              "source-card" +
              (s.available ? "" : " source-card--disabled") +
              (s.id === source ? " source-card--active" : "")
            }
            onClick={() => pickSource(s)}
          >
            {s.icon && <span className="source-icon">{s.icon}</span>}
            <span className="source-label">{s.label}</span>
            <span className={"source-status" + (s.available ? " source-status--available" : "")}>
              {s.available ? "Available" : "Coming Soon"}
            </span>
          </button>
        ))}
      </div>

      <form className="wizard-card" onSubmit={handleSubmit}>
        <div className="wizard-field">
          <span className="wizard-field-label">Model name</span>
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="my-model" />
        </div>
        <div className="wizard-field">
          <span className="wizard-field-label">Hugging Face model</span>
          <input value={repoId} onChange={(e) => setRepoId(e.target.value)} placeholder="Qwen/Qwen2.5-Coder-3B-Instruct" />
        </div>
        {error && <div className="error-banner">{error}</div>}
        <div className="wizard-actions">
          <div className="wizard-actions-spacer" />
          <button type="submit" disabled={submitting}>{submitting ? "Starting download…" : "Download Model"}</button>
        </div>
      </form>
    </div>
  );
}
