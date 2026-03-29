import { useState, useEffect, useCallback } from 'react';
import { IconButton, Tooltip, Badge } from '@mui/material';
import TuneIcon from '@mui/icons-material/Tune';
import DecisionReviewDialog from './DecisionReviewDialog.jsx';

/**
 * Shows a subtle badge on objects that have detected decision points —
 * things the user could review or improve, not errors.
 * Fetches decisions per-object from /api/decisions/for/<type>/<id>.
 *
 * Props:
 *   objectType: "stack" | "target" | "mosaic"
 *   objectId: the stable_hash of the object
 */
export default function DecisionBadge({ objectType, objectId }) {
  const [decisions, setDecisions] = useState([]);
  const [dialogOpen, setDialogOpen] = useState(false);

  const loadDecisions = useCallback(async () => {
    try {
      const res = await fetch(`/api/decisions/for/${objectType}/${objectId}`);
      setDecisions(await res.json());
    } catch { /* ignore */ }
  }, [objectType, objectId]);

  useEffect(() => { loadDecisions(); }, [loadDecisions]);

  if (decisions.length === 0) return null;

  const hasOverrides = decisions.some(d => d.is_overridden);

  return (
    <>
      <Tooltip title={`${decisions.length} suggestion${decisions.length > 1 ? 's' : ''} to review`}>
        <IconButton
          size="small"
          onClick={(e) => { e.stopPropagation(); setDialogOpen(true); }}
          sx={{
            color: hasOverrides ? 'rgba(255,255,255,0.7)' : 'rgba(255,255,255,0.45)',
            '&:hover': { color: 'rgba(255,255,255,0.9)' },
            padding: '2px',
          }}
        >
          <Badge
            badgeContent={decisions.length}
            max={99}
            sx={{
              '& .MuiBadge-badge': {
                bgcolor: 'rgba(60,130,240,0.85)',
                color: '#fff',
                fontSize: 9,
                minWidth: 14,
                height: 14,
              },
            }}
          >
            <TuneIcon sx={{ fontSize: 28 }} />
          </Badge>
        </IconButton>
      </Tooltip>
      <DecisionReviewDialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        decisions={decisions}
      />
    </>
  );
}
