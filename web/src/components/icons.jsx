// Minimal hand-rolled line icons for the sidebar -- no icon library dependency,
// consistent with LossChart.jsx's approach of plain inline SVG elsewhere in
// this app. 16x16, stroke=currentColor so they inherit the nav link's color.
const common = { width: 16, height: 16, viewBox: "0 0 16 16", fill: "none", stroke: "currentColor", strokeWidth: 1.5, strokeLinecap: "round", strokeLinejoin: "round" };

export function IconDashboard() {
  return (
    <svg {...common}>
      <rect x="1.5" y="1.5" width="5.5" height="5.5" rx="1" />
      <rect x="9" y="1.5" width="5.5" height="5.5" rx="1" />
      <rect x="1.5" y="9" width="5.5" height="5.5" rx="1" />
      <rect x="9" y="9" width="5.5" height="5.5" rx="1" />
    </svg>
  );
}

export function IconDatasets() {
  return (
    <svg {...common}>
      <ellipse cx="8" cy="3.2" rx="5.5" ry="1.8" />
      <path d="M2.5 3.2v9.6c0 1 2.46 1.8 5.5 1.8s5.5-.8 5.5-1.8V3.2" />
      <path d="M2.5 8c0 1 2.46 1.8 5.5 1.8s5.5-.8 5.5-1.8" />
    </svg>
  );
}

export function IconModels() {
  return (
    <svg {...common}>
      <path d="M8 1.5l6 3.3v6.4L8 14.5l-6-3.3V4.8z" />
      <path d="M2 4.8L8 8l6-3.2" />
      <path d="M8 8v6.5" />
    </svg>
  );
}

export function IconTraining() {
  return (
    <svg {...common}>
      <path d="M8.8 1.5L3 9h4l-.8 5.5L13 7H9z" />
    </svg>
  );
}

export function IconCheckpoints() {
  return (
    <svg {...common}>
      <rect x="2" y="2" width="12" height="3.2" rx="0.8" />
      <rect x="2" y="6.4" width="12" height="3.2" rx="0.8" />
      <rect x="2" y="10.8" width="12" height="3.2" rx="0.8" />
    </svg>
  );
}

export function IconDeployments() {
  return (
    <svg {...common}>
      <path d="M8 1.5c2 2 3 4.5 3 7 0 1.4-.5 2.7-1.3 3.8L8 14.5l-1.7-2.2C5.5 11.2 5 9.9 5 8.5c0-2.5 1-5 3-7z" />
      <circle cx="8" cy="7" r="1.3" />
      <path d="M5.3 11.5L3 14M10.7 11.5L13 14" />
    </svg>
  );
}

export function IconSettings() {
  return (
    <svg {...common}>
      <circle cx="8" cy="8" r="2.2" />
      <path d="M8 1.8v1.6M8 12.6v1.6M14.2 8h-1.6M3.4 8H1.8M12.3 3.7l-1.1 1.1M4.8 11.2l-1.1 1.1M12.3 12.3l-1.1-1.1M4.8 4.8L3.7 3.7" />
    </svg>
  );
}
