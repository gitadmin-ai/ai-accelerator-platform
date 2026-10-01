import { useEffect, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { api } from "../api";

const STEPS = ["Model", "Data", "Training"];

function defaultForm(resources, searchParams) {
  const preselectedModel = searchParams.get("model");
  const preselectedDataset = searchParams.get("dataset");
  const model = resources.models.find((m) => m.id === preselectedModel) ?? resources.models[0];
  const dataset = resources.datasets.find((d) => d.id === preselectedDataset) ?? resources.datasets[0];
  return {
    name: "onboarding-demo",
    modelId: model?.id ?? "",
    method: "lora",
    datasetId: dataset?.id ?? "",
    evaluate: true,
    evalSplitRatio: 0.1,
    epochs: 1,
    learning_rate: 0.0002,
    batch_size: 2,
    gradient_accumulation_steps: 1,
    max_seq_length: 256,
    lora_r: 8,
    lora_alpha: 16,
    lora_dropout: 0.05,
    checkpointFrequency: resources.checkpointFrequencies?.[0] ?? "epoch",
    // Pre-filled from Settings, but a per-job choice from here on -- this is
    // what lets you submit the same job twice, once local and once Nebula,
    // to compare checkpoint throughput.
    checkpointStorageBackend: resources.checkpointStorageBackend,
  };
}

export default function CreateJob() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [resources, setResources] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [step, setStep] = useState(0);
  const [form, setForm] = useState(null);
  const [errors, setErrors] = useState({});
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState(null);

  useEffect(() => {
    Promise.all([api.listModels(), api.listDatasets(), api.getConfig(), api.getSettings()])
      .then(([models, datasets, cfg, settings]) => {
        const r = {
          models: models.filter((m) => m.status === "ready"),
          datasets: datasets.filter((d) => d.status === "ready"),
          checkpointFrequencies: cfg.checkpoint_frequencies,
          storageBackends: cfg.storage_backends,
          checkpointStorageBackend: settings.checkpoint_storage_backend,
        };
        setResources(r);
        setForm(defaultForm(r, searchParams));
      })
      .catch((err) => setLoadError(String(err)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function set(patch) {
    setForm((f) => ({ ...f, ...patch }));
  }

  function validateStep(n) {
    const f = form;
    const e = {};
    if (n === 0) {
      if (resources.models.length === 0) e.model = "No ready models available yet";
      else if (!f.modelId) e.model = "Select a base model";
      if (f.method !== "lora") e.method = "Only LoRA is currently supported";
    }
    if (n === 1) {
      if (resources.datasets.length === 0) e.dataset = "No datasets available yet";
      else if (!f.datasetId) e.dataset = "Select a training dataset";
      if (f.evaluate && (f.evalSplitRatio <= 0 || f.evalSplitRatio >= 1)) {
        e.evalSplitRatio = "Must be between 0 and 1";
      }
    }
    if (n === 2) {
      if (!f.name.trim()) e.name = "Job name is required";
      if (!f.epochs || f.epochs < 1) e.epochs = "At least 1 epoch";
      if (!f.batch_size || f.batch_size < 1) e.batch_size = "At least 1";
      if (!f.gradient_accumulation_steps || f.gradient_accumulation_steps < 1) {
        e.gradient_accumulation_steps = "At least 1";
      }
      if (!f.learning_rate || f.learning_rate <= 0) e.learning_rate = "Must be positive";
      if (!f.max_seq_length || f.max_seq_length < 8) e.max_seq_length = "At least 8";
      if (!f.lora_r || f.lora_r < 1) e.lora_r = "At least 1";
      if (!f.lora_alpha || f.lora_alpha < 1) e.lora_alpha = "At least 1";
      if (f.lora_dropout < 0 || f.lora_dropout >= 1) e.lora_dropout = "Must be between 0 and 1";
    }
    setErrors(e);
    return Object.keys(e).length === 0;
  }

  function goNext() {
    if (validateStep(step)) setStep((s) => Math.min(s + 1, STEPS.length - 1));
  }

  function goBack() {
    setErrors({});
    setStep((s) => Math.max(s - 1, 0));
  }

  async function handleStart(e) {
    e.preventDefault();
    if (!validateStep(2)) return;
    setSubmitting(true);
    setSubmitError(null);
    try {
      const model = resources.models.find((m) => m.id === form.modelId);
      const dataset = resources.datasets.find((d) => d.id === form.datasetId);
      // A "ddl"-backed resource has no local path (see backend/models.py,
      // backend/datasets.py) -- its own id doubles as the Nebula resource
      // id the training worker materializes from.
      const modelSource = model.storage_backend === "ddl" ? "ddl" : "local";
      const datasetSource = dataset.storage_backend === "ddl" ? "ddl" : "local";
      const jobConfig = {
        name: form.name,
        model: { name: modelSource === "ddl" ? model.id : model.path, source: modelSource },
        dataset: { name: datasetSource === "ddl" ? dataset.id : dataset.path, source: datasetSource },
        storage: { backend: form.checkpointStorageBackend },
        training: {
          method: "lora",
          epochs: Number(form.epochs),
          batch_size: Number(form.batch_size),
          gradient_accumulation_steps: Number(form.gradient_accumulation_steps),
          learning_rate: Number(form.learning_rate),
          max_seq_length: Number(form.max_seq_length),
          lora_r: Number(form.lora_r),
          lora_alpha: Number(form.lora_alpha),
          lora_dropout: Number(form.lora_dropout),
        },
        checkpoint: { frequency: form.checkpointFrequency },
        evaluation: { enabled: form.evaluate, split_ratio: Number(form.evalSplitRatio) },
      };
      const created = await api.createJob(jobConfig);
      await api.startJob(created.job_id);
      navigate(`/jobs/${created.job_id}`);
    } catch (err) {
      setSubmitError(String(err));
    } finally {
      setSubmitting(false);
    }
  }

  if (loadError) return <div className="page"><div className="error-banner">{loadError}</div></div>;
  if (!resources || !form) return <div className="page">Loading…</div>;

  return (
    <div className="page">
      <Link to="/jobs" className="back-link">&larr; Training</Link>
      <div className="page-header">
        <h1>Create Fine-Tuning Job</h1>
      </div>

      <WizardSteps step={step} />

      <div className="wizard-card">
        {step === 0 && <StepModel resources={resources} form={form} set={set} errors={errors} />}
        {step === 1 && <StepData resources={resources} form={form} set={set} errors={errors} />}
        {step === 2 && <StepTraining resources={resources} form={form} set={set} errors={errors} />}

        {submitError && <div className="error-banner">{submitError}</div>}

        <div className="wizard-actions">
          {step > 0 && (
            <button type="button" className="secondary-button" onClick={goBack}>
              &larr; Back
            </button>
          )}
          <div className="wizard-actions-spacer" />
          {step < STEPS.length - 1 && (
            <button type="button" onClick={goNext}>Next &rarr;</button>
          )}
          {step === STEPS.length - 1 && (
            <button type="button" onClick={handleStart} disabled={submitting}>
              {submitting ? "Starting…" : "Start Training"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function WizardSteps({ step }) {
  return (
    <div className="wizard-steps" role="list">
      {STEPS.map((label, i) => (
        <div
          key={label}
          role="listitem"
          aria-current={i === step ? "step" : undefined}
          className={
            "wizard-step" +
            (i === step ? " wizard-step--active" : "") +
            (i < step ? " wizard-step--done" : "")
          }
        >
          <span className="wizard-step-index">{i < step ? "✓" : i + 1}</span>
          <span className="wizard-step-label">{label.toUpperCase()}</span>
        </div>
      ))}
    </div>
  );
}

function Field({ label, error, children }) {
  return (
    <label className="wizard-field">
      <span className="wizard-field-label">{label}</span>
      {children}
      {error && <span className="field-error">{error}</span>}
    </label>
  );
}

function StepModel({ resources, form, set, errors }) {
  if (resources.models.length === 0) {
    return (
      <fieldset>
        <legend>1. Model</legend>
        <p className="wizard-hint">
          No ready models yet. <Link to="/models/new">Add a model</Link> from Hugging Face first.
        </p>
      </fieldset>
    );
  }
  return (
    <fieldset>
      <legend>1. Model</legend>

      <Field label="Base model" error={errors.model}>
        <select value={form.modelId} onChange={(e) => set({ modelId: e.target.value })}>
          {resources.models.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
        </select>
      </Field>

      <div className="wizard-field">
        <span className="wizard-field-label">Fine-tuning method</span>
        <div className="method-options">
          <label className="method-option method-option--active">
            <input type="radio" name="method" checked readOnly />
            LoRA
          </label>
          <label className="method-option method-option--disabled">
            <input type="radio" name="method" disabled />
            Full fine-tuning <span className="coming-soon">Coming soon</span>
          </label>
          <label className="method-option method-option--disabled">
            <input type="radio" name="method" disabled />
            QLoRA <span className="coming-soon">Coming soon</span>
          </label>
        </div>
        {errors.method && <span className="field-error">{errors.method}</span>}
      </div>
    </fieldset>
  );
}

function StepData({ resources, form, set, errors }) {
  if (resources.datasets.length === 0) {
    return (
      <fieldset>
        <legend>2. Data</legend>
        <p className="wizard-hint">
          No datasets yet. <Link to="/datasets/new">Create a dataset</Link> first.
        </p>
      </fieldset>
    );
  }
  return (
    <fieldset>
      <legend>2. Data</legend>

      <Field label="Training dataset" error={errors.dataset}>
        <select value={form.datasetId} onChange={(e) => set({ datasetId: e.target.value })}>
          {resources.datasets.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
        </select>
      </Field>

      <p className="wizard-hint">
        Validation examples are held out automatically from the training dataset (Nebula
        doesn't yet support selecting a separate validation dataset resource). Example
        counts for this run become available once training starts.
      </p>

      <div className="wizard-field-row">
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={form.evaluate}
            onChange={(e) => set({ evaluate: e.target.checked })}
          />
          Hold out a validation split
        </label>
        {form.evaluate && (
          <Field label="Validation fraction" error={errors.evalSplitRatio}>
            <input
              type="number" step="0.01" min="0.01" max="0.9"
              value={form.evalSplitRatio}
              onChange={(e) => set({ evalSplitRatio: e.target.value })}
            />
          </Field>
        )}
      </div>
    </fieldset>
  );
}

function StepTraining({ resources, form, set, errors }) {
  return (
    <fieldset>
      <legend>3. Training</legend>

      <Field label="Job name" error={errors.name}>
        <input value={form.name} onChange={(e) => set({ name: e.target.value })} />
      </Field>

      <h3 className="wizard-subheading">LoRA configuration</h3>
      <div className="form-grid">
        <Field label="Rank" error={errors.lora_r}>
          <input type="number" min="1" value={form.lora_r} onChange={(e) => set({ lora_r: e.target.value })} />
        </Field>
        <Field label="Alpha" error={errors.lora_alpha}>
          <input type="number" min="1" value={form.lora_alpha} onChange={(e) => set({ lora_alpha: e.target.value })} />
        </Field>
        <Field label="Dropout" error={errors.lora_dropout}>
          <input type="number" step="0.01" min="0" max="0.99" value={form.lora_dropout}
                 onChange={(e) => set({ lora_dropout: e.target.value })} />
        </Field>
      </div>

      <h3 className="wizard-subheading">Training</h3>
      <div className="form-grid">
        <Field label="Epochs" error={errors.epochs}>
          <input type="number" min="1" value={form.epochs} onChange={(e) => set({ epochs: e.target.value })} />
        </Field>
        <Field label="Learning rate" error={errors.learning_rate}>
          <input type="number" step="0.00001" value={form.learning_rate}
                 onChange={(e) => set({ learning_rate: e.target.value })} />
        </Field>
        <Field label="Batch size" error={errors.batch_size}>
          <input type="number" min="1" value={form.batch_size} onChange={(e) => set({ batch_size: e.target.value })} />
        </Field>
        <Field label="Gradient accumulation" error={errors.gradient_accumulation_steps}>
          <input type="number" min="1" value={form.gradient_accumulation_steps}
                 onChange={(e) => set({ gradient_accumulation_steps: e.target.value })} />
        </Field>
        <Field label="Max sequence length" error={errors.max_seq_length}>
          <input type="number" min="8" value={form.max_seq_length}
                 onChange={(e) => set({ max_seq_length: e.target.value })} />
        </Field>
      </div>

      <Field label="Checkpoint frequency">
        <select
          value={form.checkpointFrequency}
          onChange={(e) => set({ checkpointFrequency: e.target.value })}
          disabled={resources.checkpointFrequencies.length <= 1}
        >
          {resources.checkpointFrequencies.map((f) => <option key={f} value={f}>Every {f}</option>)}
        </select>
      </Field>

      <div className="wizard-field">
        <span className="wizard-field-label">Checkpoint storage</span>
        <p className="wizard-hint">
          Defaults to the Settings page's choice; override it here to compare local vs. Nebula
          throughput without changing Settings between runs.
        </p>
        <div className="source-grid">
          {resources.storageBackends.map((b) => (
            <button
              key={b.id}
              type="button"
              className={"source-card" + (form.checkpointStorageBackend === b.id ? " source-card--active" : "")}
              onClick={() => set({ checkpointStorageBackend: b.id })}
            >
              <span className="source-label">{b.label}</span>
            </button>
          ))}
        </div>
      </div>
    </fieldset>
  );
}
