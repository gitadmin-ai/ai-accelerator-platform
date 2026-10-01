const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  if (res.status === 204) return null; // DELETE endpoints return no body
  return res.json();
}

// No Content-Type header here -- the browser sets the multipart boundary itself.
async function requestForm(path, formData) {
  const res = await fetch(`${API_BASE}${path}`, { method: "POST", body: formData });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json();
}

export const api = {
  getConfig: () => request("/config"),
  getActivity: (limit = 15) => request(`/activity?limit=${limit}`),

  listJobs: () => request("/jobs"),
  getJob: (id) => request(`/jobs/${id}`),
  createJob: (config) => request("/jobs", { method: "POST", body: JSON.stringify(config) }),
  startJob: (id) => request(`/jobs/${id}/start`, { method: "POST" }),
  getCheckpoints: (id) => request(`/jobs/${id}/checkpoints`),
  deleteJob: (id) => request(`/jobs/${id}`, { method: "DELETE" }),
  eventsUrl: (id) => `${API_BASE}/jobs/${id}/events`,
  artifactUrl: (id) => `${API_BASE}/jobs/${id}/artifact`,

  listDatasets: () => request("/datasets"),
  getDataset: (id) => request(`/datasets/${id}`),
  previewDataset: (id, limit = 5) => request(`/datasets/${id}/preview?limit=${limit}`),
  uploadDataset: (name, file) => {
    const form = new FormData();
    form.append("name", name);
    form.append("file", file);
    return requestForm("/datasets", form);
  },
  importDataset: (name, repoId, configName, split) =>
    request("/datasets/import", {
      method: "POST",
      body: JSON.stringify({
        name, repo_id: repoId, config_name: configName || undefined, split: split || undefined,
      }),
    }),
  deleteDataset: (id) => request(`/datasets/${id}`, { method: "DELETE" }),

  listModels: () => request("/models"),
  getModel: (id) => request(`/models/${id}`),
  downloadModel: (name, repoId) =>
    request("/models", { method: "POST", body: JSON.stringify({ name, repo_id: repoId }) }),
  deleteModel: (id) => request(`/models/${id}`, { method: "DELETE" }),

  getSettings: () => request("/settings"),
  updateSettings: (patch) => request("/settings", { method: "PUT", body: JSON.stringify(patch) }),
};
