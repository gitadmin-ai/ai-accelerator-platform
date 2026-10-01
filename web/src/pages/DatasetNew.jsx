import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api";

const SOURCES = [
  { id: "local", label: "Local", icon: "📁", available: true },
  { id: "huggingface", label: "Hugging Face", icon: "🤗", available: true },
  { id: "s3", label: "AWS S3", available: false },
  { id: "gcs", label: "Google Cloud Storage", available: false },
  { id: "azure", label: "Azure Blob Storage", available: false },
];

export default function DatasetNew() {
  const navigate = useNavigate();
  const [source, setSource] = useState(null);
  const [comingSoon, setComingSoon] = useState(null);

  function pickSource(s) {
    if (!s.available) {
      setComingSoon(`${s.label} integration is coming soon. You will be able to import datasets directly from ${s.label}.`);
      return;
    }
    setComingSoon(null);
    setSource(s.id);
  }

  if (source === "local") {
    return <LocalUploadForm onBack={() => setSource(null)} onCreated={(d) => navigate(`/datasets/${d.id}`)} />;
  }
  if (source === "huggingface") {
    return <HfImportForm onBack={() => setSource(null)} onCreated={(d) => navigate(`/datasets/${d.id}`)} />;
  }

  return (
    <div className="page">
      <Link to="/datasets" className="back-link">&larr; Datasets</Link>
      <div className="page-header"><h1>Create Dataset</h1></div>

      <h3 className="wizard-subheading">Data Source</h3>
      {comingSoon && <div className="info-banner">{comingSoon}</div>}

      <div className="source-grid">
        {SOURCES.map((s) => (
          <button
            key={s.id}
            type="button"
            className={"source-card" + (s.available ? "" : " source-card--disabled")}
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
    </div>
  );
}

function LocalUploadForm({ onBack, onCreated }) {
  const [name, setName] = useState("");
  const [file, setFile] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!name.trim() || !file) {
      setError("A dataset name and a .jsonl file are both required.");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const created = await api.uploadDataset(name, file);
      onCreated(created);
    } catch (err) {
      setError(String(err));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="page">
      <button type="button" className="back-link back-link--button" onClick={onBack}>&larr; Data Source</button>
      <div className="page-header"><h1>Create Dataset · Local</h1></div>

      <form className="wizard-card" onSubmit={handleSubmit}>
        <div className="wizard-field">
          <span className="wizard-field-label">Dataset name</span>
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="my-dataset" />
        </div>
        <div className="wizard-field">
          <span className="wizard-field-label">File (.jsonl)</span>
          <input type="file" accept=".jsonl" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        </div>
        {error && <div className="error-banner">{error}</div>}
        <div className="wizard-actions">
          <div className="wizard-actions-spacer" />
          <button type="submit" disabled={submitting}>{submitting ? "Uploading…" : "Create Dataset"}</button>
        </div>
      </form>
    </div>
  );
}

function HfImportForm({ onBack, onCreated }) {
  const [name, setName] = useState("");
  const [repoId, setRepoId] = useState("");
  const [configName, setConfigName] = useState("");
  const [split, setSplit] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!name.trim() || !repoId.trim()) {
      setError("A dataset name and a Hugging Face repository are both required.");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const created = await api.importDataset(name, repoId, configName.trim(), split.trim());
      onCreated(created);
    } catch (err) {
      setError(String(err));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="page">
      <button type="button" className="back-link back-link--button" onClick={onBack}>&larr; Data Source</button>
      <div className="page-header"><h1>Create Dataset · Hugging Face</h1></div>

      <form className="wizard-card" onSubmit={handleSubmit}>
        <div className="wizard-field">
          <span className="wizard-field-label">Dataset name</span>
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="my-dataset" />
        </div>
        <div className="wizard-field">
          <span className="wizard-field-label">Hugging Face dataset</span>
          <input value={repoId} onChange={(e) => setRepoId(e.target.value)} placeholder="username/dataset-name" />
        </div>
        <div className="wizard-field">
          <span className="wizard-field-label">Config name (optional)</span>
          <input value={configName} onChange={(e) => setConfigName(e.target.value)} placeholder="e.g. mrpc, sst2" />
          <p className="wizard-hint">
            Only needed for multi-config datasets (e.g. <code>nyu-mll/glue</code>) -- leave blank
            otherwise. If required and left blank, the error message will list the valid options.
          </p>
        </div>
        <div className="wizard-field">
          <span className="wizard-field-label">Split</span>
          <input value={split} onChange={(e) => setSplit(e.target.value)} placeholder="train" />
          <p className="wizard-hint">
            Defaults to <code>train</code>. Some configs only have other splits (e.g. glue's{" "}
            <code>ax</code> diagnostic config only has <code>test</code>) -- if the default fails,
            the error message will list the splits that dataset actually has.
          </p>
        </div>
        {submitting && <p className="wizard-hint">Downloading and importing -- this can take a moment for larger datasets…</p>}
        {error && <div className="error-banner">{error}</div>}
        <div className="wizard-actions">
          <div className="wizard-actions-spacer" />
          <button type="submit" disabled={submitting}>{submitting ? "Importing…" : "Import Dataset"}</button>
        </div>
      </form>
    </div>
  );
}
