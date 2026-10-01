export function formatDuration(seconds) {
  if (seconds == null) return "–";
  const total = Math.max(0, Math.round(seconds));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

export function formatClock(ts) {
  return new Date(ts * 1000).toLocaleTimeString([], { hour12: false });
}

export function formatDateTime(ts) {
  if (ts == null) return "–";
  return new Date(ts * 1000).toLocaleString([], {
    year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

export function formatBytes(bytes) {
  if (bytes == null) return "–";
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  return `${value.toFixed(value < 10 ? 2 : 1)} ${units[i]}`;
}

export function sourceLabel(source) {
  return { local: "Local", huggingface: "Hugging Face" }[source] ?? source;
}

// Lifecycle milestones shown in the collapsible Live Events log. Per-step
// "step" events are deliberately excluded here -- they already drive the
// Training Metrics stat tiles and loss chart, and showing all of them here
// would turn the log into an engineering console rather than a readable
// timeline (see Section 12/9 of the Phase 2 plan).
const HUMAN_LABELS = {
  queued: () => "Job queued",
  worker_started: () => "Worker started",
  model_loading_start: (e) => `Loading model ${e.model ?? ""}`.trim(),
  model_loaded: () => "Model loaded",
  dataset_loading_start: () => "Loading dataset",
  dataset_loaded: (e) => `Dataset loaded (${e.num_train_examples ?? "?"} train / ${e.num_eval_examples ?? 0} eval examples)`,
  epoch_start: (e) => `Epoch ${e.epoch}/${e.total_epochs ?? "?"} started`,
  epoch_complete: (e) => `Epoch ${e.epoch} completed (mean loss ${e.mean_loss?.toFixed(4) ?? "?"})`,
  checkpoint_start: () => "Checkpoint started",
  checkpoint_saved: (e) => `Checkpoint saved (${(e.size_bytes / 1024).toFixed(0)} KB, ${e.num_chunks} chunks)`,
  eval_start: () => "Evaluation started",
  eval_complete: (e) => `Evaluation completed (loss ${e.eval_loss?.toFixed(4) ?? "?"})`,
  job_complete: () => "Training complete",
  job_failed: () => "Job failed",
};

export const LIFECYCLE_EVENT_NAMES = Object.keys(HUMAN_LABELS);

export function humanEventLabel(event) {
  const fn = HUMAN_LABELS[event.event];
  return fn ? fn(event) : event.event;
}
