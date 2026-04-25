import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import Chip from '@mui/material/Chip';
import Checkbox from '@mui/material/Checkbox';
import TextField from '@mui/material/TextField';
import LinearProgress from '@mui/material/LinearProgress';

function hours(s) {
  return (s / 3600).toFixed(1);
}

export default function SuggestionCard({ suggestion, entry, onToggle, onUpdate }) {
  const enabled = !!entry?.enabled;
  const target = suggestion.target_exposure_s || 1;
  const progress = Math.min(100, (100 * suggestion.current_exposure_s) / target);

  const kindChip = suggestion.kind === 'missing_channel' ? (
    <Chip size="small" label="missing channel" color="warning" />
  ) : (
    <Chip size="small" label="completion" />
  );

  return (
    <Paper
      variant="outlined"
      sx={{
        p: 2,
        opacity: enabled ? 1.0 : 0.75,
        borderColor: enabled ? 'primary.main' : 'divider',
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <Checkbox checked={enabled} onChange={onToggle} size="small" />
        <Typography variant="subtitle2" sx={{ flex: 1 }}>
          {suggestion.panel_label}
        </Typography>
        <Chip size="small" label={suggestion.filter} />
        {kindChip}
      </Box>
      <Box sx={{ mt: 1 }}>
        <Typography variant="caption" color="text.secondary">
          {hours(suggestion.current_exposure_s)}h → {hours(suggestion.target_exposure_s)}h
        </Typography>
        <LinearProgress
          variant="determinate"
          value={progress}
          sx={{ mt: 0.5, height: 6, borderRadius: 3 }}
        />
      </Box>
      <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
        {suggestion.reason}
      </Typography>
      {enabled && (
        <Box sx={{ mt: 1.5, display: 'flex', gap: 1 }}>
          <TextField
            label="Exposure (s)"
            type="number"
            size="small"
            value={entry?.exposurePerFrame ?? suggestion.default_exposure_per_frame_s}
            onChange={(e) => onUpdate({ exposurePerFrame: Number(e.target.value) })}
            inputProps={{ min: 1, step: 1 }}
            sx={{ width: 120 }}
          />
          <TextField
            label="Frames"
            type="number"
            size="small"
            value={entry?.frameCount ?? suggestion.default_frame_count}
            onChange={(e) => onUpdate({ frameCount: Number(e.target.value) })}
            inputProps={{ min: 1, step: 1 }}
            sx={{ width: 100 }}
          />
          <Typography variant="body2" color="text.secondary" sx={{ alignSelf: 'center' }}>
            = {hours((entry?.exposurePerFrame ?? 0) * (entry?.frameCount ?? 0))}h
          </Typography>
        </Box>
      )}
    </Paper>
  );
}
