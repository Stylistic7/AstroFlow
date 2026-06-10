import React from "react";
import LinearProgress from "@mui/material/LinearProgress";
import { Box } from "@mui/material";

import { toneColor } from "../tones.js";

/**
 * A 2 px tall LinearProgress strip — fractional progress (e.g.
 * "12 of 23 frames stacked") shown inline in a dropdown header.
 * Sparkline is the right tool for streaming time series; this is
 * the right tool when the data is "X of Y, climbing toward 100%".
 *
 * Props:
 *   value     — number in [0, 100] (REQUIRED)
 *   width     — px, default 60
 *   tone      — "default" | "active" | "warning" | "error", default "active"
 *   indeterminate — boolean; if true, ignores `value` and shows the
 *                   MUI shimmer (use when work is in flight but the
 *                   total isn't known yet)
 */
const MiniProgress = ({ value, width = 60, tone = "active", indeterminate = false }) => (
  <Box sx={{ width, display: "inline-block" }}>
    <LinearProgress
      variant={indeterminate ? "indeterminate" : "determinate"}
      value={indeterminate ? undefined : Math.max(0, Math.min(100, value || 0))}
      sx={{
        height: 2,
        borderRadius: 1,
        backgroundColor: "rgba(255,255,255,0.08)",
        "& .MuiLinearProgress-bar": {
          backgroundColor: toneColor(tone),
        },
      }}
    />
  </Box>
);

export default MiniProgress;
