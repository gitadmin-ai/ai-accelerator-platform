import { Link } from "react-router-dom";

export default function Checkpoints() {
  return (
    <div className="page">
      <div className="page-header"><h1>Checkpoints</h1></div>
      <div className="detail-card">
        <p>Checkpoint management is available through individual training jobs.</p>
        <p className="job-subtitle">
          Open a training job's detail page to see its checkpoints -- size, chunk count,
          write throughput, and completion status for each one.
        </p>
        <Link to="/jobs" className="link-button">View Training Jobs &rarr;</Link>
      </div>
    </div>
  );
}
