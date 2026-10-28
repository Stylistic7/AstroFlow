import React, { useMemo, useState } from "react";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  IconButton,
  Box,
  Typography,
  Snackbar,
  Button,
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import { CopyToClipboard } from "react-copy-to-clipboard";

import { ra_to_hms, dec_to_dms } from "./units.js";

const monospaceSx = {
  fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
  fontSize: 12,
  whiteSpace: "pre-wrap",
  lineHeight: 1.5,
};

// `center` = where the pane is aimed. For a panel that is the Target's
// `stable_center`, the oldest frame's own solved centre and the point the
// planner's ASIAir push aims at (2026-10-27) — never the mean `center`, which
// moves with every frame added, so a mis-pointed night would re-aim the next.
// Two aim points for one panel is how the 7098 seams opened (2026-10-14).
// Coordinates are J2000 (plate solutions), what the ASIAir CSV import expects.
function paneLineForStack(stack, pane, center = stack.center) {
  const [ra, dec] = center;
  const rotation_angle = (stack.rotation_angle * 180.0) / Math.PI;
  const pane_width = (stack.image_scale * stack.width_pixels) / 60.0;
  const pane_height = (stack.image_scale * stack.height_pixels) / 60.0;
  return `Pane ${pane + 1}, ${ra_to_hms(ra)}, ${dec_to_dms(dec)}, ${rotation_angle.toFixed(1)}, ${pane_width.toFixed(2)}, ${pane_height.toFixed(2)}, 20%, 1, 1`;
}

function paneLineForTarget(stacks, targets, target_id, pane) {
  const target = targets[target_id];
  const stack = stacks[target.stack_ids[0]];
  return paneLineForStack(stack, pane, target.stable_center);
}

const HEADER = "Pane, RA, DEC, Position Angle (East), Pane width (arcmins), Pane height (arcmins), Overlap, Row, Column";

/**
 * Generates Telescopius-compatible pane text for a mosaic or a single stack.
 *
 * Props:
 *   open, onClose: dialog state
 *   stacks, targets: context data
 *   mosaic: mosaic object (optional — if provided, generates panes for all targets)
 *   stack: single stack object (optional — if provided, generates one pane)
 */
export default function TelescopiusDialog({ open, onClose, stacks, targets, mosaic, stack }) {
  const [copiedOpen, setCopiedOpen] = useState(false);

  const text = useMemo(() => {
    if (stack) {
      return [HEADER, paneLineForStack(stack, 0)].join("\n");
    }
    if (mosaic && targets && stacks) {
      // The Mosaic's own panel order, so "Pane N" is polygon "N" in the viewer.
      const lines = mosaic.panel_target_ids.map((tid, i) => paneLineForTarget(stacks, targets, tid, i));
      return [HEADER, ...lines].join("\n");
    }
    return null;
  }, [stacks, targets, mosaic, stack]);

  if (!text) return null;

  return (
    <>
      <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
        <DialogTitle sx={{ pr: 6 }}>
          Telescopius
          <IconButton
            size="small"
            onClick={onClose}
            sx={{ position: "absolute", right: 12, top: 12 }}
          >
            <CloseIcon fontSize="small" />
          </IconButton>
        </DialogTitle>
        <DialogContent>
          <Box sx={{ ...monospaceSx, bgcolor: "background.default", p: 2, borderRadius: 1, mb: 2 }}>
            {text}
          </Box>
          <CopyToClipboard text={text} onCopy={() => setCopiedOpen(true)}>
            <Button variant="outlined" size="small" startIcon={<ContentCopyIcon />}>
              Copy to clipboard
            </Button>
          </CopyToClipboard>
        </DialogContent>
      </Dialog>
      <Snackbar
        anchorOrigin={{ vertical: "top", horizontal: "center" }}
        open={copiedOpen}
        autoHideDuration={3000}
        onClose={() => setCopiedOpen(false)}
        message="Copied to clipboard"
      />
    </>
  );
}
