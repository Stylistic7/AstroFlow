import { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  List,
  ListItem,
  ListItemText,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
  Chip,
  Stack,
  IconButton,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';

/**
 * Dialog listing decision points for an object, with override controls.
 *
 * Props:
 *   open: boolean
 *   onClose: () => void
 *   decisions: array of decision objects from /api/decisions
 */
export default function DecisionReviewDialog({ open, onClose, decisions }) {
  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="sm"
      fullWidth
      onClick={(e) => e.stopPropagation()}
    >
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        Decision Points
        <IconButton onClick={onClose} size="small">
          <CloseIcon />
        </IconButton>
      </DialogTitle>
      <DialogContent>
        {decisions.length === 0 ? (
          <Typography color="text.secondary">No decisions to review.</Typography>
        ) : (
          <List disablePadding>
            {decisions.map((d, i) => (
              <DecisionItem key={JSON.stringify(d.key)} decision={d} />
            ))}
          </List>
        )}
      </DialogContent>
    </Dialog>
  );
}

function DecisionItem({ decision }) {
  const [loading, setLoading] = useState(false);
  const [effectiveChoice, setEffectiveChoice] = useState(decision.effective_choice);
  const [isOverridden, setIsOverridden] = useState(decision.is_overridden);

  const handleChange = async (event, newValue) => {
    if (newValue === null) return;

    setLoading(true);
    try {
      // "default" means clear the override; otherwise set the override
      const choice = newValue === 'default' ? null : newValue;
      const resp = await fetch('/api/decisions/override', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key: decision.key, choice }),
      });
      await resp.json();

      if (newValue === 'default') {
        setEffectiveChoice(decision.default_choice);
        setIsOverridden(false);
      } else {
        setEffectiveChoice(newValue);
        setIsOverridden(true);
      }
    } finally {
      setLoading(false);
    }
  };

  // Determine current toggle value
  const toggleValue = isOverridden ? effectiveChoice : 'default';

  return (
    <ListItem
      sx={{
        flexDirection: 'column',
        alignItems: 'flex-start',
        borderBottom: '1px solid',
        borderColor: 'divider',
        py: 2,
      }}
    >
      <ListItemText
        primary={decision.title}
        secondary={decision.description}
        sx={{ mb: 1, width: '100%' }}
      />
      <Stack direction="row" spacing={1} sx={{ mb: 1, flexWrap: 'wrap', gap: 0.5 }}>
        {Object.entries(decision.metadata || {}).map(([k, v]) => (
          <Chip
            key={k}
            label={`${k}: ${v}`}
            size="small"
            variant="outlined"
          />
        ))}
      </Stack>
      <ToggleButtonGroup
        value={toggleValue}
        exclusive
        onChange={handleChange}
        disabled={loading}
        size="small"
      >
        <ToggleButton value="include" color="success">
          Include
        </ToggleButton>
        <ToggleButton value="exclude" color="error">
          Exclude
        </ToggleButton>
        <ToggleButton value="default">
          Default ({decision.default_choice})
        </ToggleButton>
      </ToggleButtonGroup>
    </ListItem>
  );
}
