import { useEffect, useState } from "react";
import { api } from "../api";

const BACKENDS = [
  { id: "local", label: "Local", icon: "📁" },
  { id: "ddl", label: "Nebula (DDL)", icon: "⚡" },
];

const PICKERS = [
  { key: "model_storage_backend", title: "Model storage", hint: "Where newly downloaded models are stored." },
  { key: "dataset_storage_backend", title: "Dataset storage", hint: "Where newly created/imported datasets are stored." },
  { key: "checkpoint_storage_backend", title: "Checkpoint storage", hint: "Where a training job's checkpoints are written by default (overridable per job)." },
];

export default function Settings() {
  const [settings, setSettings] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState(null);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    api.getSettings()
      .then(setSettings)
      .catch((err) => setLoadError(String(err)));
  }, []);

  function set(patch) {
    setSaved(false);
    setSettings((s) => ({ ...s, ...patch }));
  }

  async function handleSave(e) {
    e.preventDefault();
    setSaving(true);
    setSaveError(null);
    setSaved(false);
    try {
      const updated = await api.updateSettings(settings);
      setSettings(updated);
      setSaved(true);
    } catch (err) {
      setSaveError(String(err));
    } finally {
      setSaving(false);
    }
  }

  if (loadError) return <div className="page"><div className="error-banner">{loadError}</div></div>;
  if (!settings) return <div className="page">Loading…</div>;

  const anyDdl = PICKERS.some((p) => settings[p.key] === "ddl");

  return (
    <div className="page">
      <div className="page-header"><h1>Settings</h1></div>
      <p className="wizard-hint">
        Choose local disk or Nebula (a distributed parallel object store) independently for models,
        datasets, and checkpoints. New uploads and training jobs use whatever is selected here at the
        time they run -- existing resources keep the backend they were created with.
      </p>

      <form className="wizard-card" onSubmit={handleSave}>
        {PICKERS.map((picker) => (
          <div key={picker.key} className="wizard-field">
            <h3 className="wizard-subheading">{picker.title}</h3>
            <p className="wizard-hint">{picker.hint}</p>
            <div className="source-grid">
              {BACKENDS.map((b) => (
                <button
                  key={b.id}
                  type="button"
                  className={"source-card" + (settings[picker.key] === b.id ? " source-card--active" : "")}
                  onClick={() => set({ [picker.key]: b.id })}
                >
                  <span className="source-icon">{b.icon}</span>
                  <span className="source-label">{b.label}</span>
                </button>
              ))}
            </div>
          </div>
        ))}

        {anyDdl && (
          <div className="wizard-field">
            <h3 className="wizard-subheading">Nebula (DDL) connection</h3>
            <div className="form-grid">
              <label className="wizard-field">
                <span className="wizard-field-label">Server</span>
                <input
                  value={settings.ddl_server ?? ""}
                  onChange={(e) => set({ ddl_server: e.target.value })}
                  placeholder="nebula.internal"
                />
              </label>
              <label className="wizard-field">
                <span className="wizard-field-label">Port</span>
                <input
                  type="number" min="1"
                  value={settings.ddl_port}
                  onChange={(e) => set({ ddl_port: Number(e.target.value) })}
                />
              </label>
              <label className="wizard-field">
                <span className="wizard-field-label">Tenant</span>
                <input
                  type="number" min="0"
                  value={settings.ddl_tenant}
                  onChange={(e) => set({ ddl_tenant: Number(e.target.value) })}
                />
              </label>
              <label className="wizard-field">
                <span className="wizard-field-label">CPU base</span>
                <input
                  type="number" min="0"
                  value={settings.ddl_cpu_base}
                  onChange={(e) => set({ ddl_cpu_base: Number(e.target.value) })}
                />
              </label>
            </div>
          </div>
        )}

        {saveError && <div className="error-banner">{saveError}</div>}
        {saved && !saveError && <div className="info-banner">Settings saved.</div>}

        <div className="wizard-actions">
          <div className="wizard-actions-spacer" />
          <button type="submit" disabled={saving}>{saving ? "Saving…" : "Save Settings"}</button>
        </div>
      </form>
    </div>
  );
}
