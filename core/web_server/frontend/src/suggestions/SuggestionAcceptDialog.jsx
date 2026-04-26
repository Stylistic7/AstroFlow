import { useState } from 'react';
import {
  Box,
  Dialog,
  DialogTitle,
  DialogContent,
  List,
  ListItem,
  Typography,
  Chip,
  Stack,
  IconButton,
  TextField,
  Button,
  LinearProgress,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';

function hours(s) {
  return (s / 3600).toFixed(1);
}

// Group suggestions by their target (one panel = one target across
// filters), sorted by panel_number so panels appear in the same order
// as their polygon labels in the mosaic viewer. Returns
// [[panel_number, panel_label, suggestions[]], ...].
function groupByPanel(suggestions) {
  const groups = new Map();
  for (const s of suggestions) {
    const key = s.target_id ?? `${s.panel_number}-${s.panel_label}`;
    if (!groups.has(key)) {
      groups.set(key, {
        panel_number: s.panel_number ?? null,
        panel_label: s.panel_label,
        suggestions: [],
      });
    }
    groups.get(key).suggestions.push(s);
  }
  const arr = Array.from(groups.values()).map((g) => [
    g.panel_number,
    g.panel_label,
    // Sort suggestions within a panel by filter for stable display.
    g.suggestions.sort((a, b) => (a.filter || '').localeCompare(b.filter || '')),
  ]);
  arr.sort((a, b) => {
    // Numbered panels before unnumbered, then by panel number.
    const an = a[0], bn = b[0];
    if (an === null && bn === null) return a[1].localeCompare(b[1]);
    if (an === null) return 1;
    if (bn === null) return -1;
    return an - bn;
  });
  return arr;
}

export default function SuggestionAcceptDialog({
  open,
  onClose,
  suggestions,
  acceptedKeys,
  onChanged,
}) {
  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="sm"
      fullWidth
      onClick={(e) => e.stopPropagation()}
    >
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        Suggestions
        <IconButton onClick={onClose} size="small">
          <CloseIcon />
        </IconButton>
      </DialogTitle>
      <DialogContent>
        {suggestions.length === 0 ? (
          <Typography color="text.secondary">No suggestions.</Typography>
        ) : (
          groupByPanel(suggestions).map(([panelNumber, panelLabel, group]) => (
            <Box
              key={`${panelNumber ?? 'np'}-${panelLabel}`}
              sx={{
                mb: 3,
                pb: 1,
                borderBottom: '2px solid',
                borderColor: 'divider',
              }}
            >
              <Typography
                variant="subtitle1"
                sx={{ fontWeight: 600, mt: 1, mb: 0.5 }}
              >
                {panelNumber !== null
                  ? `Panel ${panelNumber}: ${panelLabel}`
                  : panelLabel}
              </Typography>
              <List disablePadding>
                {group.map((s) => (
                  <SuggestionItem
                    key={s.stable_key}
                    suggestion={s}
                    accepted={acceptedKeys.has(s.stable_key)}
                    onChanged={onChanged}
                  />
                ))}
              </List>
            </Box>
          ))
        )}
      </DialogContent>
    </Dialog>
  );
}

function SuggestionItem({ suggestion, accepted, onChanged }) {
  const [exposure, setExposure] = useState(suggestion.default_exposure_per_frame_s);
  const [count, setCount] = useState(suggestion.default_frame_count);
  const [busy, setBusy] = useState(false);

  const kindChip = suggestion.kind === 'missing_channel' ? (
    <Chip size="small" label="missing" color="warning" />
  ) : (
    <Chip size="small" label="completion" />
  );

  const progress = Math.min(
    100,
    (100 * suggestion.current_exposure_s) / Math.max(suggestion.target_exposure_s, 1)
  );

  const accept = async () => {
    setBusy(true);
    try {
      await fetch('/api/observing_plan/accept', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          suggestion_key: suggestion.stable_key,
          exposure_per_frame_s: exposure,
          frame_count: count,
          filter: suggestion.filter,
          panel_label: suggestion.panel_label,
        }),
      });
      if (onChanged) await onChanged();
    } finally {
      setBusy(false);
    }
  };

  const remove = async () => {
    setBusy(true);
    try {
      await fetch(
        `/api/observing_plan/entry/${encodeURIComponent(suggestion.stable_key)}`,
        { method: 'DELETE' }
      );
      if (onChanged) await onChanged();
    } finally {
      setBusy(false);
    }
  };

  return (
    <ListItem
      sx={{
        flexDirection: 'column',
        alignItems: 'flex-start',
        borderBottom: '1px solid',
        borderColor: 'divider',
        py: 2,
        gap: 1,
      }}
    >
      <Stack direction="row" spacing={1} alignItems="center" sx={{ width: '100%' }}>
        <Chip size="small" label={suggestion.filter} sx={{ minWidth: 36 }} />
        {kindChip}
        <Box sx={{ flex: 1 }} />
      </Stack>
      <Typography variant="caption" color="text.secondary">
        {hours(suggestion.current_exposure_s)}h → {hours(suggestion.target_exposure_s)}h
      </Typography>
      <LinearProgress
        variant="determinate"
        value={progress}
        sx={{ width: '100%', height: 6, borderRadius: 3 }}
      />
      <Typography variant="body2" color="text.secondary">
        {suggestion.reason}
      </Typography>
      <Stack direction="row" spacing={1} alignItems="center" sx={{ mt: 0.5 }}>
        <TextField
          label="Exposure (s)"
          type="number"
          size="small"
          value={exposure}
          onChange={(e) => setExposure(Number(e.target.value))}
          disabled={accepted}
          inputProps={{ min: 1, step: 1 }}
          sx={{ width: 120 }}
        />
        <TextField
          label="Frames"
          type="number"
          size="small"
          value={count}
          onChange={(e) => setCount(Number(e.target.value))}
          disabled={accepted}
          inputProps={{ min: 1, step: 1 }}
          sx={{ width: 100 }}
        />
        <Typography variant="body2" color="text.secondary">
          = {hours(exposure * count)}h
        </Typography>
        {accepted ? (
          <Button
            size="small"
            color="success"
            startIcon={<CheckCircleIcon />}
            onClick={remove}
            disabled={busy}
          >
            In plan (click to remove)
          </Button>
        ) : (
          <Button
            size="small"
            variant="contained"
            onClick={accept}
            disabled={busy}
          >
            Accept
          </Button>
        )}
      </Stack>
    </ListItem>
  );
}
