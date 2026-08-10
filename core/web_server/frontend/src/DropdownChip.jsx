import React, { useState, useRef } from "react";
import { Box, Menu, Tooltip, Typography } from "@mui/material";

import Chevron from "./Chevron.jsx";
import { toneColor } from "./tones.js";

/**
 * The shared header for every top-bar dropdown panel. One chip per
 * panel, all visually rhythmic: icon · summary · chevron. Click
 * anywhere on the chip to open its menu.
 *
 * Props:
 *   icon         — React node, the leading icon. Required. Rendered
 *                  at 18 px and colored by `tone`.
 *   summary      — React node, the glanceable middle. Typically a
 *                  small flex row of <Typography>, <Sparkline>,
 *                  <StatusDot>, <MiniProgress>. Optional; chip is
 *                  icon-only when omitted (e.g. for tiny "is it on"
 *                  badges where the icon alone says everything).
 *   tone         — "default" | "active" | "warning" | "error".
 *                  Drives the leading icon's color via `toneColor`.
 *                  Active green / warning amber / error red on the
 *                  icon alone is the peripheral-vision signal — the
 *                  chip background stays neutral so the toolbar
 *                  reads as a single rhythm and individual chips
 *                  don't shout for attention from across the screen.
 *   tooltipTitle — optional; rendered as an MUI Tooltip on hover
 *                  when the menu is CLOSED.
 *   ariaLabel    — accessibility label for the chip button. Falls
 *                  back to `tooltipTitle` when unset.
 *   children     — the dropdown menu body. Rendered inside an MUI
 *                  Menu anchored to the chip. Auto-closes when the
 *                  user clicks outside or hits Escape; explicit
 *                  close is via the `onClose` callback Menu provides.
 *
 * Visually:
 *   - 32 px tall, padded 8 px horizontally.
 *   - Border-radius 8 px for a softly rounded chip.
 *   - Cursor: pointer.
 *   - Hover: very slight background lift even at default tone.
 */
const DropdownChip = ({
  icon,
  summary,
  tone = "default",
  tooltipTitle,
  ariaLabel,
  children,
}) => {
  const [anchorEl, setAnchorEl] = useState(null);
  const chipRef = useRef(null);
  const open = Boolean(anchorEl);
  const iconColor = toneColor(tone);

  const chip = (
    <Box
      ref={chipRef}
      role="button"
      aria-haspopup="menu"
      aria-expanded={open}
      aria-label={ariaLabel || tooltipTitle}
      tabIndex={0}
      onClick={() => setAnchorEl(chipRef.current)}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          setAnchorEl(chipRef.current);
        }
      }}
      sx={{
        display: "flex",
        alignItems: "center",
        gap: 1,
        cursor: "pointer",
        padding: "4px 8px",
        height: 32,
        borderRadius: 1,
        transition: "background-color 200ms ease-out",
        "&:hover": {
          backgroundColor: "rgba(255,255,255,0.05)",
        },
        // Keyboard-focus ring matches the MUI button rhythm.
        "&:focus-visible": {
          outline: "2px solid",
          outlineColor: "primary.main",
          outlineOffset: 1,
        },
      }}
    >
      {/* Leading icon, sized + colored consistently. */}
      {React.cloneElement(icon, {
        sx: {
          fontSize: 18,
          color: iconColor,
          ...(icon.props.sx || {}),
        },
      })}
      {summary && (
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.75, minWidth: 0 }}>
          {typeof summary === "string"
            ? <Typography variant="body2" noWrap>{summary}</Typography>
            : summary}
        </Box>
      )}
      <Chevron open={open} size={16} color="text.secondary" />
    </Box>
  );

  // Always wrap in Tooltip (when a title is set) — flipping
  // Tooltip on/off based on `open` was remounting the chip Box and
  // detaching the ref the Menu anchored to, which sent the dropdown
  // to (0,0). Tooltip suppresses itself when the user isn't
  // hovering, so the only cost is a redundant tooltip flash if the
  // user happens to hover the chip while the menu is already open
  // — minor cosmetic, not a regression.
  const chipMaybeTooltipped = tooltipTitle
    ? <Tooltip title={tooltipTitle}>{chip}</Tooltip>
    : chip;

  return (
    <>
      {chipMaybeTooltipped}
      <Menu
        anchorEl={anchorEl}
        open={open}
        onClose={() => setAnchorEl(null)}
        slotProps={{
          paper: { sx: { minWidth: 240, maxWidth: 480 } },
        }}
      >
        {/* children-as-function receives {close} so an item that NAVIGATES
            can dismiss the menu first (a plain-node menu otherwise stays
            open, floating over the new page). Plain children unchanged. */}
        {typeof children === "function"
          ? children({ close: () => setAnchorEl(null) })
          : children}
      </Menu>
    </>
  );
};

export default DropdownChip;
