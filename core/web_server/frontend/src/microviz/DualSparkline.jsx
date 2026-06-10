import React from "react";

/**
 * Two overlaid sparkline polylines sharing a single set of axes,
 * pinned to a [0, 100] Y domain so the two series are directly
 * comparable. No grid, no axis labels — the chart is intentionally
 * legend-less in compact use; the parent renders a separate legend
 * for the expanded view.
 *
 * Aspect ratio defaults to ~2:1 (about an inch wide by half an inch
 * tall at typical screen DPI) — wide enough to read short-term
 * dynamics, short enough to sit in a toolbar chip.
 *
 * Props:
 *   primary    — number[]  REQUIRED. Drawn underneath / first.
 *   secondary  — number[]  REQUIRED. Drawn over the primary.
 *   primaryColor   — CSS color string. Default: theme info.light.
 *   secondaryColor — CSS color string. Default: theme success.light.
 *   width      — px, default 96
 *   height     — px, default 32 (2:1-ish; fits a 40 px chip)
 *   strokeWidth — px, default 1.5
 */
const DualSparkline = ({
  primary,
  secondary,
  primaryColor = "#64b5f6",   // theme.palette.info.light fallback
  secondaryColor = "#81c784", // theme.palette.success.light fallback
  width = 96,
  height = 32,
  strokeWidth = 1.5,
}) => {
  const yLo = 0;
  const yHi = 100;
  const range = yHi - yLo;

  const buildPath = (values) => {
    if (!values || values.length === 0) return "";
    const finite = values.map(v => (Number.isFinite(v) ? v : null));
    if (finite.every(v => v === null)) return "";
    const xStep = values.length > 1 ? width / (values.length - 1) : 0;
    const norm = (v) => {
      if (v === null) return null;
      return height - ((v - yLo) / range) * height;
    };
    let d = "";
    let inSegment = false;
    for (let i = 0; i < finite.length; i++) {
      const y = norm(finite[i]);
      const x = i * xStep;
      if (y === null) {
        inSegment = false;
        continue;
      }
      d += inSegment
        ? ` L${x.toFixed(2)},${y.toFixed(2)}`
        : ` M${x.toFixed(2)},${y.toFixed(2)}`;
      inSegment = true;
    }
    return d;
  };

  const primaryD = buildPath(primary);
  const secondaryD = buildPath(secondary);

  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      {primaryD && (
        <path
          d={primaryD}
          fill="none"
          stroke={primaryColor}
          strokeWidth={strokeWidth}
          strokeLinejoin="round"
          strokeLinecap="round"
        />
      )}
      {secondaryD && (
        <path
          d={secondaryD}
          fill="none"
          stroke={secondaryColor}
          strokeWidth={strokeWidth}
          strokeLinejoin="round"
          strokeLinecap="round"
        />
      )}
    </svg>
  );
};

export default DualSparkline;
