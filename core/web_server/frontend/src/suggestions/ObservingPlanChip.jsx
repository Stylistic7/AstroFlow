import { useContext, useState } from 'react';
import {
  Box,
  Badge,
  Button,
  Chip,
  IconButton,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Menu,
  Stack,
  Typography,
  Divider,
} from '@mui/material';
import LightbulbIcon from '@mui/icons-material/Lightbulb';
import DeleteIcon from '@mui/icons-material/Delete';

import AstroFlowContext from '../AstroFlowContext.jsx';

function hours(s) {
  return (s / 3600).toFixed(1);
}

async function downloadExport(entries, format, refresh) {
  const res = await fetch('/api/observing_plan/export', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ entries, format }),
  });
  if (!res.ok) return;
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const ext = format === 'telescopius' ? 'csv' : format === 'json' ? 'json' : 'txt';
  const today = new Date().toISOString().slice(0, 10);
  const a = document.createElement('a');
  a.href = url;
  a.download = `observing_plan_${today}.${ext}`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export default function ObservingPlanChip() {
  const { observingPlan, refreshObservingPlan } = useContext(AstroFlowContext);
  const [anchorEl, setAnchorEl] = useState(null);
  const entries = observingPlan?.entries || [];

  const totalSeconds = entries.reduce(
    (a, e) => a + (e.exposure_per_frame_s || 0) * (e.frame_count || 0),
    0
  );

  const perFilter = new Map();
  for (const e of entries) {
    const t = (e.exposure_per_frame_s || 0) * (e.frame_count || 0);
    perFilter.set(e.filter, (perFilter.get(e.filter) || 0) + t);
  }

  const removeEntry = async (key) => {
    await fetch(`/api/observing_plan/entry/${encodeURIComponent(key)}`, {
      method: 'DELETE',
    });
    if (refreshObservingPlan) await refreshObservingPlan();
  };

  const clearAll = async () => {
    if (!window.confirm('Clear all suggestions from the observing plan?')) return;
    await fetch('/api/observing_plan/clear', { method: 'POST' });
    if (refreshObservingPlan) await refreshObservingPlan();
  };

  return (
    <>
      <IconButton
        size="small"
        color="inherit"
        onClick={(e) => setAnchorEl(e.currentTarget)}
      >
        <Badge
          badgeContent={entries.length}
          color="warning"
          max={99}
        >
          <LightbulbIcon />
        </Badge>
      </IconButton>
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={() => setAnchorEl(null)}
        PaperProps={{ sx: { minWidth: 340, maxWidth: 440, p: 1 } }}
      >
        <Box sx={{ px: 1 }}>
          <Typography variant="subtitle2">
            Observing plan{entries.length > 0 ? ` — ${hours(totalSeconds)}h` : ''}
          </Typography>
          {perFilter.size > 0 && (
            <Stack direction="row" spacing={0.5} sx={{ mt: 0.5, flexWrap: 'wrap' }}>
              {Array.from(perFilter.entries()).map(([f, t]) => (
                <Chip key={f} size="small" label={`${f}: ${hours(t)}h`} />
              ))}
            </Stack>
          )}
        </Box>
        <Divider sx={{ my: 1 }} />
        {entries.length === 0 ? (
          <Box sx={{ px: 2, py: 1 }}>
            <Typography variant="body2" color="text.secondary">
              No suggestions accepted yet. Click a lightbulb on a target or
              mosaic to add one.
            </Typography>
          </Box>
        ) : (
          <List dense disablePadding>
            {entries.map((e) => {
              const label = e.panel_label || '(panel)';
              const primary = e.panel_number != null
                ? `Panel ${e.panel_number}: ${label} — ${e.filter}`
                : `${label} — ${e.filter}`;
              return (
                <ListItem key={e.suggestion_key} sx={{ pr: 7 }}>
                  <ListItemText
                    primary={primary}
                    secondary={`${e.frame_count} × ${Math.round(e.exposure_per_frame_s)}s = ${hours(e.exposure_per_frame_s * e.frame_count)}h`}
                  />
                  <ListItemSecondaryAction>
                    <IconButton size="small" onClick={() => removeEntry(e.suggestion_key)}>
                      <DeleteIcon fontSize="small" />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
              );
            })}
          </List>
        )}
        <Divider sx={{ my: 1 }} />
        <Box sx={{ px: 1, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          <Button
            size="small"
            variant="contained"
            onClick={() => downloadExport(entries, 'telescopius')}
            disabled={entries.length === 0}
          >
            Telescopius
          </Button>
          <Button
            size="small"
            variant="outlined"
            onClick={() => downloadExport(entries, 'text')}
            disabled={entries.length === 0}
          >
            Text
          </Button>
          <Button
            size="small"
            variant="outlined"
            onClick={() => downloadExport(entries, 'json')}
            disabled={entries.length === 0}
          >
            JSON
          </Button>
          <Box sx={{ flex: 1 }} />
          <Button
            size="small"
            color="error"
            onClick={clearAll}
            disabled={entries.length === 0}
          >
            Clear all
          </Button>
        </Box>
      </Menu>
    </>
  );
}
