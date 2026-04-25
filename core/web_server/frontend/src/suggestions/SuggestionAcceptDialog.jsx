import { useState } from 'react';
import {
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
          <List disablePadding>
            {suggestions.map((s) => (
              <SuggestionItem
                key={s.stable_key}
                suggestion={s}
                accepted={acceptedKeys.has(s.stable_key)}
                onChanged={onChanged}
              />
            ))}
          </List>
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
        <Typography variant="subtitle2" sx={{ flex: 1 }}>
          {suggestion.panel_label}
        </Typography>
        <Chip size="small" label={suggestion.filter} />
        {kindChip}
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
