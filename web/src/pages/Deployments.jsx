export default function Deployments() {
  return (
    <div className="page">
      <div className="page-header"><h1>Deployments</h1></div>
      <div className="detail-card">
        <p>Deployments aren't available yet.</p>
        <p className="job-subtitle">
          Once a fine-tuning job completes, its adapter can be downloaded from the job's detail
          page. A full deployment workflow -- serving a trained adapter as an endpoint -- is
          planned for a future phase.
        </p>
      </div>
    </div>
  );
}
