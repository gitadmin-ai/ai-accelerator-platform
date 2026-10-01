import { NavLink, Outlet } from "react-router-dom";
import {
  IconCheckpoints, IconDashboard, IconDatasets, IconDeployments, IconModels, IconSettings, IconTraining,
} from "../components/icons";

const NAV_ITEMS = [
  { to: "/", label: "Dashboard", icon: IconDashboard, end: true },
  { to: "/datasets", label: "Datasets", icon: IconDatasets },
  { to: "/models", label: "Models", icon: IconModels },
  { to: "/jobs", label: "Training", icon: IconTraining },
  { to: "/checkpoints", label: "Checkpoints", icon: IconCheckpoints },
  { to: "/deployments", label: "Deployments", icon: IconDeployments },
  { to: "/settings", label: "Settings", icon: IconSettings },
];

export default function AppLayout() {
  return (
    <div className="app-shell">
      <nav className="sidebar">
        <div className="sidebar-brand">NebulaAI</div>
        <ul className="sidebar-nav">
          {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
            <li key={to}>
              <NavLink
                to={to}
                end={end}
                className={({ isActive }) => "sidebar-link" + (isActive ? " sidebar-link--active" : "")}
              >
                <Icon />
                <span>{label}</span>
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
      <main className="app-content">
        <Outlet />
      </main>
    </div>
  );
}
