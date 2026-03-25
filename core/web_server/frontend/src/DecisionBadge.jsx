import { useState, useEffect, useCallback } from 'react';
import { IconButton, Tooltip, Badge } from '@mui/material';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import DecisionReviewDialog from './DecisionReviewDialog.jsx';

/**
 * Shows an amber warning badge on objects that have detected decision points.
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
      <Tooltip title={`${decisions.length} decision(s) flagged for review`}>
        <IconButton
          onClick={(e) => { e.stopPropagation(); setDialogOpen(true); }}
          sx={{ color: hasOverrides ? 'info.main' : 'warning.main' }}
        >
          <Badge badgeContent={decisions.length} color="warning" max={99}>
            <WarningAmberIcon />
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
