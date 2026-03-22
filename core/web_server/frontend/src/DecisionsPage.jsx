import { useState, useContext, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Chip,
  Paper,
  Stack,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from '@mui/material';
import AstroFlowContext from './AstroFlowContext.jsx';

// Priority order: lower-level decisions first.
const OBJECT_TYPE_PRIORITY = { stack: 0, target: 1, mosaic: 2 };

function sortedDecisions(decisions) {
  return [...decisions].sort((a, b) => {
    const pa = OBJECT_TYPE_PRIORITY[a.object_type] ?? 99;
    const pb = OBJECT_TYPE_PRIORITY[b.object_type] ?? 99;
    if (pa !== pb) return pa - pb;
    // Within the same level, show unreviewed first.
    if (a.is_overridden !== b.is_overridden) return a.is_overridden ? 1 : -1;
    return 0;
  });
}

export default function DecisionsPage() {
  const { decisions, stacks } = useContext(AstroFlowContext);
  const navigate = useNavigate();

  const sorted = useMemo(() => sortedDecisions(decisions || []), [decisions]);

  if (sorted.length === 0) {
    return (
      <Box sx={{ p: 4 }}>
        <Typography variant="h5" gutterBottom>Decisions</Typography>
        <Typography color="text.secondary">
          No decision points have been detected. Decisions appear here when the
          system finds borderline data that you may want to review.
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 4, maxWidth: 900, mx: 'auto' }}>
      <Typography variant="h5" gutterBottom>Decisions</Typography>
      <Typography color="text.secondary" sx={{ mb: 3 }}>
        The system flagged these data points for your review.
        Lower-level decisions (stacks) are shown first.
      </Typography>
      <Stack spacing={2}>
        {sorted.map((d) => (
          <DecisionCard
            key={JSON.stringify(d.key)}
            decision={d}
            stacks={stacks}
            navigate={navigate}
          />
        ))}
      </Stack>
    </Box>
  );
}

function DecisionCard({ decision, stacks, navigate }) {
  const [loading, setLoading] = useState(false);
  const [effectiveChoice, setEffectiveChoice] = useState(decision.effective_choice);
  const [isOverridden, setIsOverridden] = useState(decision.is_overridden);

  const handleChange = async (_event, newValue) => {
    if (newValue === null) return;
    setLoading(true);
    try {
      const choice = newValue === 'default' ? null : newValue;
      await fetch('/api/decisions/override', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key: decision.key, choice }),
      });
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

  const toggleValue = isOverridden ? effectiveChoice : 'default';

  // Resolve a human-readable name for the parent object.
  const objectName = stacks?.[decision.object_id]?.name ?? decision.object_id.slice(0, 8);

  return (
    <Paper
      sx={{
        p: 2,
        opacity: isOverridden ? 0.7 : 1,
        borderLeft: 4,
        borderColor: isOverridden ? 'info.main' : 'warning.main',
      }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 2 }}>
        <Box sx={{ flex: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
            <Chip
              label={decision.object_type}
              size="small"
              variant="outlined"
            />
            <Typography
              variant="subtitle2"
              sx={{ cursor: 'pointer', '&:hover': { textDecoration: 'underline' } }}
              onClick={() => navigate(`/${decision.object_type}/${decision.object_id}`)}
            >
              {objectName}
            </Typography>
          </Box>
          <Typography variant="body1" sx={{ fontWeight: 500 }}>
            {decision.title}
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
            {decision.description}
          </Typography>
          <Stack direction="row" spacing={1} sx={{ mt: 1, flexWrap: 'wrap', gap: 0.5 }}>
            {Object.entries(decision.metadata || {}).map(([k, v]) => (
              <Chip key={k} label={`${k}: ${v}`} size="small" variant="outlined" />
            ))}
          </Stack>
        </Box>
        <ToggleButtonGroup
          value={toggleValue}
          exclusive
          onChange={handleChange}
          disabled={loading}
          size="small"
          orientation="vertical"
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
      </Box>
    </Paper>
  );
}
