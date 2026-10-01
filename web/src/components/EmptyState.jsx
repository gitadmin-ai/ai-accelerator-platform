import { Link } from "react-router-dom";

export default function EmptyState({ title, body, actionLabel, actionTo }) {
  return (
    <div className="empty-state">
      <p className="empty-state-title">{title}</p>
      <p className="empty-state-body">{body}</p>
      {actionTo && (
        <Link to={actionTo} className="empty-state-action">{actionLabel}</Link>
      )}
    </div>
  );
}
