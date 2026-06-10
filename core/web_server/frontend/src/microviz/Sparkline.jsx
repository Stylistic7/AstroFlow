import React from "react";
import { useTheme } from "@mui/material/styles";

/**
 * Pure-SVG sparkline. No chart library — the data is just a polyline
 * through `(x, y)` points. Designed for inline use in dropdown
 * headers: tiny (default 60×16 px), no axes, no legend, no
 * interactivity. The "alive" feeling comes from the values
 * changing on each parent re-render.
 *
 * Props:
 *   values  — number[] (REQUIRED). Drawn in order; index = x position.
 *             Falsy / undefined / NaN entries collapse to gaps in the
 *             line. Empty array renders an empty <svg> (zero-width
 *             space holder).
 *   width   — px, default 60
 *   height  — px, default 16
 *   color   — stroke color; resolves through the theme's palette
 *             paths or any CSS color. Default: theme.palette.primary.main.
 *   fill    — boolean, default false. If true, the area under the
 *             polyline is filled with a 20% alpha of `color`.
 *   domain  — [min, max] tuple. If provided, the Y scale is pinned
 *             to this range; otherwise auto-scales to the data's
 *             min/max with a 5% padding on each side to keep the
 *             line off the edges. Useful when you want zero pinned
 *             (e.g. CPU%) or when streaming values where local
 *             min/max would jitter the baseline.
 *   strokeWidth — px, default 1.5.
 */
const Sparkline = ({
  values,
  width = 60,
  height = 16,
  color,
  fill = false,
  domain,
  strokeWidth = 1.5,
}) => {
  const theme = useTheme();
  const resolvedColor = color || theme.palette.primary.main;

  if (!values || values.length === 0) {
    return <svg width={width} height={height} />;
  }

  // Find the finite samples — preserve NaN / undefined as breaks in
  // the line so a "lost signal" frame renders as a visible gap, not
  // a wild spike to zero.
  const finite = values.map(v => (Number.isFinite(v) ? v : null));
  const samples = finite.filter(v => v !== null);
  if (samples.length === 0) {
    return <svg width={width} height={height} />;
  }

  const [yMin, yMax] = domain || [
    Math.min(...samples),
    Math.max(...samples),
  ];
  // 5% pad so the line doesn't kiss the edges; only applied to
  // auto-derived domains.
  const range = yMax - yMin || 1;
  const pad = domain ? 0 : range * 0.05;
  const yLo = yMin - pad;
  const yHi = yMax + pad;

  const xStep = values.length > 1 ? width / (values.length - 1) : 0;
  const norm = (v) => {
    if (v === null) return null;
    return height - ((v - yLo) / (yHi - yLo)) * height;
  };

  // Build the path as a series of M / L commands, breaking on null.
  let d = "";
  let inSegment = false;
  for (let i = 0; i < finite.length; i++) {
    const y = norm(finite[i]);
    const x = i * xStep;
    if (y === null) {
      inSegment = false;
      continue;
    }
    d += inSegment ? ` L${x.toFixed(2)},${y.toFixed(2)}` : ` M${x.toFixed(2)},${y.toFixed(2)}`;
    inSegment = true;
  }

  // For the fill, close the path back along the baseline at y=height.
  const fillD = fill && d
    ? `${d} L${(values.length - 1) * xStep},${height} L0,${height} Z`
    : "";

  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      {fill && (
        <path d={fillD} fill={resolvedColor} fillOpacity={0.2} stroke="none" />
      )}
      <path
        d={d}
        fill="none"
        stroke={resolvedColor}
        strokeWidth={strokeWidth}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
};

export default Sparkline;
