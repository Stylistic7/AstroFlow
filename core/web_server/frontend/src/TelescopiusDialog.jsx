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

function paneLineForStack(stack, pane) {
  const [ra, dec] = stack.center;
  const rotation_angle = (stack.rotation_angle * 180.0) / Math.PI;
  const pane_width = (stack.image_scale * stack.width_pixels) / 60.0;
  const pane_height = (stack.image_scale * stack.height_pixels) / 60.0;
  return `Pane ${pane + 1}, ${ra_to_hms(ra)}, ${dec_to_dms(dec)}, ${rotation_angle.toFixed(1)}, ${pane_width.toFixed(2)}, ${pane_height.toFixed(2)}, 20%, 1, 1`;
}

function paneLineForTarget(stacks, targets, target_id, pane) {
  const target = targets[target_id];
  const stack = stacks[target.stack_ids[0]];
  return paneLineForStack(stack, pane);
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
      const sorted = [...mosaic.target_ids].sort((a, b) => {
        const ra_a = stacks[targets[a]?.stack_ids?.[0]]?.center?.[0] ?? 0;
        const ra_b = stacks[targets[b]?.stack_ids?.[0]]?.center?.[0] ?? 0;
        return ra_a - ra_b;
      });
      const lines = sorted.map((tid, i) => paneLineForTarget(stacks, targets, tid, i));
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
