import React from "react";
import Circle from "@mui/icons-material/Circle";

import { toneColor } from "../tones.js";

/**
 * A 10 px circle colored by tone. Used as the row-level "is this
 * thing currently healthy?" signal inside the dropdown menus (Ray's
 * per-node health, AsiairPanel's per-row capture state, etc.) and
 * occasionally in the chip header itself when the tone alone
 * doesn't carry enough signal.
 *
 * Props:
 *   tone — "default" | "active" | "warning" | "error"
 *   size — px, default 10
 */
const StatusDot = ({ tone = "default", size = 10 }) => (
  <Circle sx={{ fontSize: size, color: toneColor(tone) }} />
);

export default StatusDot;
