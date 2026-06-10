import React from "react";
import ExpandMore from "@mui/icons-material/ExpandMore";

/**
 * The single chevron used by every dropdown — panels AND pipeline /
 * tap / filter pills. Rotates 180° on open instead of swapping
 * between ExpandMore and ExpandLess: the rotation gives a clearer
 * open/closed signal than the swap, costs less to read, and
 * matches the visual rhythm we want across the top bar.
 *
 * Props:
 *   open     — boolean, controlled by the parent's menu state
 *   size     — px, defaults to 18 (matches DropdownChip's leading icon)
 *   color    — any `sx`-compatible color string; defaults to inherit
 */
const Chevron = ({ open, size = 18, color = "inherit" }) => (
  <ExpandMore
    sx={{
      fontSize: size,
      color,
      transition: "transform 200ms ease-out",
      transform: open ? "rotate(180deg)" : "none",
      // Disabled-state opacity is handled by the parent — Chevron
      // itself doesn't know about disabled.
    }}
  />
);

export default Chevron;
