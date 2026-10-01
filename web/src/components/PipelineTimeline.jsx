const STEPS = ["Dataset", "Training", "Checkpoint", "Evaluation", "Model"];

// Maps job status + whether a "dataset_loaded" event has been seen yet to
// an index into STEPS -- CHECKPOINTING/EVALUATING are repeating sub-loops
// in the real state machine (see app/state_machine.py), so this only ever
// reflects the *current* stage, not a strictly one-way progress bar.
function stepIndexFor(status, datasetLoaded) {
  switch (status) {
    case "SUBMITTED":
    case "QUEUED":
      return -1;
    case "RUNNING":
      return datasetLoaded ? 1 : 0;
    case "CHECKPOINTING":
      return 2;
    case "EVALUATING":
      return 3;
    case "COMPLETED":
      return 4;
    default:
      return -1;
  }
}

const SYMBOL = { done: "✓", active: "●", pending: "○", failed: "✕" };

export default function PipelineTimeline({ status, datasetLoaded }) {
  const failed = status === "FAILED";
  const currentIndex = stepIndexFor(status, datasetLoaded);
  const failedIndex = Math.max(currentIndex, 0);

  return (
    <div className="timeline" role="list" aria-label="Pipeline stage">
      {STEPS.map((label, i) => {
        let state = "pending";
        if (failed && i === failedIndex) state = "failed";
        else if (i < currentIndex || status === "COMPLETED") state = "done";
        else if (i === currentIndex) state = "active";

        return (
          <div className="timeline-step" key={label} role="listitem" aria-current={state === "active"}>
            <div className={`timeline-dot timeline-dot--${state}`}>{SYMBOL[state]}</div>
            <div className={`timeline-label timeline-label--${state}`}>{label}</div>
            {i < STEPS.length - 1 && <div className={`timeline-connector timeline-connector--${state === "pending" ? "pending" : state}`} />}
          </div>
        );
      })}
    </div>
  );
}
