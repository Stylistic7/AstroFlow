// DiscardResultsDialog.jsx — lists every op in a pipeline's trace for one
// entity (grouped by function) and lets the user check ops to discard their
// cached results (files + sidecars + tile pyramid + diskcache) so the pipeline
// re-runs them. Backs the A/B-testing loop: tweak one op → clear its results →
// re-run → compare. Scoped to (objectType, objectId, pipeline).
//
// Data: GET /api/pipeline_results/<type>/<id>/<pipeline> → {ops:[{op_name,
// display, is_final, count, on_disk, bytes}]}. Confirm POSTs the selected
// op_names to /api/discard_results/..., which deletes + drain_and_restarts;
// the dashboard re-pulls on the resulting computeGraphStatus change.
import { useState, useEffect, useCallback } from "react";
import {
  Box, Chip, Dialog, DialogTitle, DialogContent,
  DialogContentText, DialogActions, Button, List, ListItem,
  ListItemButton, ListItemIcon, Checkbox, ListItemText, Alert,
  CircularProgress,
} from "@mui/material";


function fmtSize(bytes) {
  if (!bytes) return "—";
  const gb = bytes / 1e9;
  return gb >= 1 ? `${gb.toFixed(1)} GB` : `${(bytes / 1e6).toFixed(0)} MB`;
}


export default function DiscardResultsDialog({
  objectType, objectId, pipeline, open, onClose,
}) {
  const [ops, setOps] = useState(null);            // null = loading
  const [selected, setSelected] = useState(() => new Set());
  const [busy, setBusy] = useState(false);

  const base = `/${objectType}/${objectId}/${encodeURIComponent(pipeline)}`;

  useEffect(() => {
    if (!open) return;
    setOps(null);
    setSelected(new Set());
    fetch(`/api/pipeline_results${base}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => setOps(d?.ops || []))
      .catch(() => setOps([]));
  }, [open, base]);

  const toggle = useCallback((name) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }, []);

  const confirm = useCallback(async () => {
    setBusy(true);
    try {
      await fetch(`/api/discard_results${base}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ op_names: [...selected] }),
      });
      onClose();
    } finally {
      setBusy(false);
    }
  }, [base, selected, onClose]);

  const totalBytes = ops
    ? ops.filter((o) => selected.has(o.op_name)).reduce((s, o) => s + o.bytes, 0)
    : 0;

  return (
    <Dialog open={open} onClose={busy ? undefined : onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Discard cached results</DialogTitle>
      <DialogContent>
        <DialogContentText>
          Select the ops whose cached outputs to delete for <b>{pipeline}</b>.
          They re-run on the next graph build.
        </DialogContentText>
        <Alert severity="info" sx={{ mt: 1, mb: 1 }}>
          Re-running an op alone won't refresh ops that consumed its earlier
          (cached) output — if a result still looks wrong, discard the upstream
          ops too (listed top-to-bottom in dependency order).
        </Alert>
        {ops === null ? (
          <Box sx={{ display: "flex", justifyContent: "center", py: 3 }}>
            <CircularProgress size={24} />
          </Box>
        ) : (
          <List dense>
            {ops.map((o) => (
              <ListItem key={o.op_name} disablePadding>
                <ListItemButton onClick={() => toggle(o.op_name)} dense>
                  <ListItemIcon sx={{ minWidth: 36 }}>
                    <Checkbox
                      edge="start"
                      checked={selected.has(o.op_name)}
                      tabIndex={-1}
                      disableRipple
                    />
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                        <span>{o.op_name}</span>
                        {o.display && (
                          <Chip
                            size="small"
                            label={o.display}
                            variant="outlined"
                            color={o.is_final ? "success" : "default"}
                          />
                        )}
                      </Box>
                    }
                    secondary={`${o.on_disk}/${o.count} on disk · ${fmtSize(o.bytes)}`}
                  />
                </ListItemButton>
              </ListItem>
            ))}
          </List>
        )}
      </DialogContent>
      <DialogActions sx={{ px: 3, pb: 2 }}>
        <Button onClick={onClose} disabled={busy}>Cancel</Button>
        <Box sx={{ flexGrow: 1 }} />
        <Button
          onClick={confirm}
          color="warning"
          variant="contained"
          disabled={busy || selected.size === 0}
        >
          {busy
            ? "Discarding…"
            : `Discard ${selected.size} op${selected.size === 1 ? "" : "s"} · ${fmtSize(totalBytes)}`}
        </Button>
      </DialogActions>
    </Dialog>
  );
}
