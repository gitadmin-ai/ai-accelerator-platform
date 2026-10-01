import { useEffect, useMemo, useRef, useState } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import { api } from "../api";
import PipelineTimeline from "../components/PipelineTimeline";
import StatusBadge from "../components/StatusBadge";
import LossChart from "../components/LossChart";
import { formatDuration, formatClock, humanEventLabel, LIFECYCLE_EVENT_NAMES } from "../utils";

const TERMINAL = new Set(["COMPLETED", "FAILED"]);
const DELETABLE = new Set(["SUBMITTED", "COMPLETED", "FAILED"]);

export default function JobDetail() {
  const { jobId } = useParams();
  const navigate = useNavigate();

  // Fetched exactly once on mount -- the only network request this page
  // makes outside the SSE connection (Section 15: no unnecessary polling).
  // Gives us name/model/dataset/config plus whatever status was true at
  // that instant, which matters for a job opened before it has emitted any
  // events yet (SUBMITTED/QUEUED).
  const [initial, setInitial] = useState(null);
  const [initialError, setInitialError] = useState(null);

  const [events, setEvents] = useState([]);
  const [now, setNow] = useState(Date.now() / 1000);
  const [logsExpanded, setLogsExpanded] = useState(false);
  const seenRef = useRef(new Set());

  useEffect(() => {
    setInitial(null);
    setInitialError(null);
    api.getJob(jobId).then(setInitial).catch((err) => setInitialError(String(err)));
  }, [jobId]);

  useEffect(() => {
    setEvents([]);
    seenRef.current = new Set();
    const es = new EventSource(api.eventsUrl(jobId));
    es.onmessage = (msg) => {
      let event;
      try {
        event = JSON.parse(msg.data);
      } catch {
        return; // keep-alive/comment lines never reach onmessage; defensive only
      }
      const key = `${event.ts}|${event.event}|${event.stage}`;
      if (seenRef.current.has(key)) return; // backlog + live can double-deliver, see api/main.py
      seenRef.current.add(key);
      setEvents((prev) => [...prev, event]);
      if (event.stage === "COMPLETED" || event.stage === "FAILED" || event.event === "stream_closed") {
        es.close();
      }
    };
    es.onerror = () => es.close();
    return () => es.close();
  }, [jobId]);

  // Local UI ticking clock for a live "Elapsed" readout -- not a network
  // request, just a re-render trigger; stops once the job is terminal.
  useEffect(() => {
    if (!initial) return; // first render(s) fire before the GET /jobs/{id} resolves
    const status = derivedStatus(initial, events);
    if (TERMINAL.has(status)) return;
    const interval = setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => clearInterval(interval);
  }, [initial, events]);

  if (initialError) return <div className="page"><div className="error-banner">{initialError}</div></div>;
  if (!initial) return <div className="page">Loading…</div>;

  const status = derivedStatus(initial, events);
  const datasetLoaded = events.some((e) => e.event === "dataset_loaded");
  const latestStep = findLast(events, (e) => e.event === "step");
  const latestEpoch = findLast(events, (e) => e.event === "epoch_start" || e.event === "epoch_complete");
  const latestEval = findLast(events, (e) => e.event === "eval_complete");
  const completeEvent = findLast(events, (e) => e.event === "job_complete");
  const failEvent = findLast(events, (e) => e.event === "job_failed");
  const startedAtEvent = events.find((e) => e.event === "worker_started");
  const checkpoints = events.filter((e) => e.event === "checkpoint_saved");
  const lossPoints = events.filter((e) => e.event === "step").map((e) => ({ step: e.global_step, loss: e.loss }));

  const startedAt = startedAtEvent?.ts ?? initial.started_at;
  let elapsedSeconds = null;
  if (completeEvent) elapsedSeconds = completeEvent.elapsed_s;
  else if (failEvent && startedAt) elapsedSeconds = failEvent.ts - startedAt;
  else if (startedAt) elapsedSeconds = now - startedAt;

  const lifecycleEvents = events.filter((e) => LIFECYCLE_EVENT_NAMES.includes(e.event));

  async function handleRetry() {
    const created = await api.createJob(initial.config);
    await api.startJob(created.job_id);
    navigate(`/jobs/${created.job_id}`);
  }

  async function handleDelete() {
    if (!window.confirm(`Delete job "${initial.name}" and its runs/artifacts? This cannot be undone.`)) return;
    try {
      await api.deleteJob(jobId);
      navigate("/jobs");
    } catch (err) {
      setInitialError(String(err));
    }
  }

  return (
    <div className="page">
      <Link to="/jobs" className="back-link">&larr; Jobs</Link>

      <div className="job-header">
        <div>
          <h1>{initial.name}</h1>
          <p className="job-subtitle">{initial.config.model.name} &middot; {initial.config.dataset.name}</p>
        </div>
        <div className="job-header-actions">
          <StatusBadge status={status} />
          {DELETABLE.has(status) && (
            <button className="danger-button" onClick={handleDelete}>Delete</button>
          )}
        </div>
      </div>

      <div className="job-summary-line">
        <span>Elapsed: <strong>{formatDuration(elapsedSeconds)}</strong></span>
        <span>Epoch: <strong>{latestEpoch ? `${latestEpoch.epoch} / ${latestEpoch.total_epochs ?? "?"}` : "–"}</strong></span>
        <span>Step: <strong>{latestStep ? `${latestStep.global_step} / ${latestStep.total_steps ?? "?"}` : "–"}</strong></span>
      </div>

      <PipelineTimeline status={status} datasetLoaded={datasetLoaded} />

      {status === "FAILED" && (
        <FailedPanel
          events={events}
          failEvent={failEvent}
          onViewLogs={() => setLogsExpanded(true)}
          onRetry={handleRetry}
        />
      )}

      {status === "COMPLETED" && (
        <ResultPanel
          jobId={jobId}
          latestEpoch={latestEpoch}
          latestEval={latestEval}
          completeEvent={completeEvent}
          elapsedSeconds={elapsedSeconds}
        />
      )}

      <div className="detail-card">
        <h3>Training Metrics</h3>
        <div className="stat-grid">
          <Stat label="Epoch" value={latestEpoch ? `${latestEpoch.epoch} / ${latestEpoch.total_epochs ?? "?"}` : "–"} />
          <Stat label="Step" value={latestStep ? `${latestStep.global_step} / ${latestStep.total_steps ?? "?"}` : "–"} />
          <Stat label="Loss" value={latestStep ? latestStep.loss.toFixed(4) : "–"} />
          <Stat label="Learning Rate" value={latestStep ? latestStep.learning_rate.toExponential(2) : "–"} />
          <Stat label="Training Duration" value={formatDuration(elapsedSeconds)} />
          <Stat label="Validation Loss" value={latestEval ? latestEval.eval_loss.toFixed(4) : "–"} />
        </div>
        <LossChart points={lossPoints} />
      </div>

      <div className="detail-card" id="checkpoints">
        <h3>Checkpoints</h3>
        <table className="checkpoints-table">
          <thead>
            <tr><th>Epoch</th><th>Size</th><th>Chunks</th><th>Duration</th><th>Throughput</th><th>Status</th></tr>
          </thead>
          <tbody>
            {checkpoints.map((c) => (
              <tr key={c.checkpoint_id}>
                <td>{c.epoch}</td>
                <td>{(c.size_bytes / 1024 / 1024).toFixed(2)} MB</td>
                <td>{c.num_chunks}</td>
                <td>{c.total_s.toFixed(2)}s</td>
                <td>{c.throughput_mb_s.toFixed(2)} MB/s</td>
                <td><span className="badge badge--completed">Complete</span></td>
              </tr>
            ))}
            {checkpoints.length === 0 && (
              <tr><td colSpan={6} className="empty-row">No checkpoints yet</td></tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="detail-card">
        <button className="link-button" onClick={() => setLogsExpanded((v) => !v)}>
          {logsExpanded ? "▾" : "▸"} Live Events {logsExpanded ? "" : `(${lifecycleEvents.length})`}
        </button>
        {logsExpanded && (
          <div className="event-log">
            {lifecycleEvents.map((e, i) => (
              <div key={i} className="event-log-line">
                <span className="event-log-time">{formatClock(e.ts)}</span> {humanEventLabel(e)}
              </div>
            ))}
            {lifecycleEvents.length === 0 && (
              <div className="empty-row">Waiting for the worker to start…</div>
            )}
            {failEvent && (
              <details className="event-log-raw">
                <summary>Full error detail</summary>
                <pre>{failEvent.error}</pre>
              </details>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function derivedStatus(initial, events) {
  const withStage = [...events].reverse().find((e) => e.stage);
  return withStage ? withStage.stage : initial.status;
}

function findLast(arr, pred) {
  for (let i = arr.length - 1; i >= 0; i--) {
    if (pred(arr[i])) return arr[i];
  }
  return null;
}

function Stat({ label, value }) {
  return (
    <div className="stat-tile">
      <div className="stat-tile-label">{label}</div>
      <div className="stat-tile-value">{value}</div>
    </div>
  );
}

function ResultPanel({ jobId, latestEpoch, latestEval, completeEvent, elapsedSeconds }) {
  return (
    <div className="detail-card result-panel">
      <h3>Training Complete</h3>
      <div className="stat-grid">
        <Stat label="Final Loss" value={latestEpoch?.mean_loss != null ? latestEpoch.mean_loss.toFixed(4) : "–"} />
        <Stat label="Validation Loss" value={latestEval ? latestEval.eval_loss.toFixed(4) : "–"} />
        <Stat label="Training Duration" value={formatDuration(elapsedSeconds)} />
        <Stat label="Epochs" value={completeEvent?.total_epochs ?? "–"} />
      </div>
      <div className="result-artifact">
        <span className="result-artifact-check">✓</span> Final model artifact saved to the job's workspace
        <a href="#checkpoints" className="link-button"> &middot; View checkpoint details</a>
      </div>
      <div className="result-actions">
        <a href={api.artifactUrl(jobId)} className="download-button" download>
          Download Model
        </a>
      </div>
    </div>
  );
}

function FailedPanel({ events, failEvent, onViewLogs, onRetry }) {
  const lastStageEvent = findLast(events, (e) => e.stage && e.stage !== "FAILED");
  const firstErrorLine = (failEvent?.error || "Job failed for an unknown reason.").split("\n")[0];
  return (
    <div className="detail-card failed-panel">
      <h3>Job Failed</h3>
      <dl>
        <dt>Stage</dt>
        <dd>{lastStageEvent?.stage ?? "Unknown"}</dd>
        <dt>Error</dt>
        <dd>{firstErrorLine}</dd>
      </dl>
      <div className="failed-actions">
        <button className="secondary-button" onClick={onViewLogs}>View Logs</button>
        <button onClick={onRetry}>Retry Job</button>
      </div>
    </div>
  );
}
