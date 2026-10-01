import { useMemo, useState } from "react";

const WIDTH = 640;
const HEIGHT = 160;
const PAD_LEFT = 40;
const PAD_RIGHT = 12;
const PAD_TOP = 12;
const PAD_BOTTOM = 24;

// Single-series line chart, no charting library -- points are few (tens,
// not thousands) since they're one per logged training step. Follows the
// dataviz skill's single-series line spec: 2px line, rounded data-end,
// --series-1 color role (defined in index.css for both themes), a hover
// crosshair+tooltip (the default interaction for line charts), recessive
// gridlines, and a "not enough data yet" empty state rather than an empty
// plot.
export default function LossChart({ points, label = "Training loss" }) {
  const [hoverIndex, setHoverIndex] = useState(null);

  const { path, dot, xFor, yFor, minY, maxY } = useMemo(() => {
    if (points.length === 0) {
      return { path: "", dot: null, xFor: () => 0, yFor: () => 0, minY: 0, maxY: 0 };
    }
    const ys = points.map((p) => p.loss);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const range = maxY - minY || 1;
    const innerW = WIDTH - PAD_LEFT - PAD_RIGHT;
    const innerH = HEIGHT - PAD_TOP - PAD_BOTTOM;
    const xFor = (i) =>
      PAD_LEFT + (points.length === 1 ? innerW / 2 : (i / (points.length - 1)) * innerW);
    const yFor = (loss) => PAD_TOP + innerH - ((loss - minY) / range) * innerH;
    const path = points.map((p, i) => `${i === 0 ? "M" : "L"}${xFor(i)},${yFor(p.loss)}`).join(" ");
    const last = points.length - 1;
    const dot = { x: xFor(last), y: yFor(points[last].loss) };
    return { path, dot, xFor, yFor, minY, maxY };
  }, [points]);

  if (points.length < 2) {
    return (
      <div className="loss-chart loss-chart--empty">
        Waiting for more training steps to plot a loss curve…
      </div>
    );
  }

  const hovered = hoverIndex != null ? points[hoverIndex] : null;

  function handleMove(e) {
    const rect = e.currentTarget.getBoundingClientRect();
    const relX = ((e.clientX - rect.left) / rect.width) * WIDTH;
    let nearest = 0;
    let nearestDist = Infinity;
    points.forEach((_, i) => {
      const d = Math.abs(xFor(i) - relX);
      if (d < nearestDist) {
        nearestDist = d;
        nearest = i;
      }
    });
    setHoverIndex(nearest);
  }

  return (
    <div className="loss-chart">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="loss-chart-svg"
        onMouseMove={handleMove}
        onMouseLeave={() => setHoverIndex(null)}
        role="img"
        aria-label={`${label} over training steps, from ${maxY.toFixed(3)} down to ${minY.toFixed(3)}`}
      >
        <line
          x1={PAD_LEFT} y1={PAD_TOP} x2={PAD_LEFT} y2={HEIGHT - PAD_BOTTOM}
          className="loss-chart-axis"
        />
        <line
          x1={PAD_LEFT} y1={HEIGHT - PAD_BOTTOM} x2={WIDTH - PAD_RIGHT} y2={HEIGHT - PAD_BOTTOM}
          className="loss-chart-axis"
        />
        <text x={4} y={PAD_TOP + 4} className="loss-chart-tick">{maxY.toFixed(2)}</text>
        <text x={4} y={HEIGHT - PAD_BOTTOM} className="loss-chart-tick">{minY.toFixed(2)}</text>

        <path d={path} className="loss-chart-line" fill="none" />
        {dot && <circle cx={dot.x} cy={dot.y} r={4} className="loss-chart-dot" />}

        {hovered && (
          <>
            <line
              x1={xFor(hoverIndex)} y1={PAD_TOP} x2={xFor(hoverIndex)} y2={HEIGHT - PAD_BOTTOM}
              className="loss-chart-crosshair"
            />
            <circle cx={xFor(hoverIndex)} cy={yFor(hovered.loss)} r={4} className="loss-chart-dot" />
          </>
        )}
      </svg>
      <div className="loss-chart-tooltip">
        {hovered
          ? `step ${hovered.step} – loss ${hovered.loss.toFixed(4)}`
          : `latest: step ${points[points.length - 1].step} – loss ${points[points.length - 1].loss.toFixed(4)}`}
      </div>
    </div>
  );
}
