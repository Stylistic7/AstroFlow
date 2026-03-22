import { useState, useContext, useMemo } from 'react';
import { IconButton, Tooltip, Badge } from '@mui/material';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import AstroFlowContext from './AstroFlowContext.jsx';
import DecisionReviewDialog from './DecisionReviewDialog.jsx';

/**
 * Shows an amber warning badge on objects that have detected decision points.
 *
 * Props:
 *   objectType: "stack" | "target" | "mosaic"
 *   objectId: the stable_hash of the object
 */
export default function DecisionBadge({ objectType, objectId }) {
  const { decisions } = useContext(AstroFlowContext);
  const [dialogOpen, setDialogOpen] = useState(false);

  const relevant = useMemo(
    () => (decisions || []).filter(
      d => d.object_type === objectType && d.object_id === objectId
    ),
    [decisions, objectType, objectId]
  );

  if (relevant.length === 0) return null;

  const hasOverrides = relevant.some(d => d.is_overridden);

  return (
    <>
      <Tooltip title={`${relevant.length} decision(s) flagged for review`}>
        <IconButton
          onClick={(e) => { e.stopPropagation(); setDialogOpen(true); }}
          sx={{ color: hasOverrides ? 'info.main' : 'warning.main' }}
        >
          <Badge badgeContent={relevant.length} color="warning" max={99}>
            <WarningAmberIcon />
          </Badge>
        </IconButton>
      </Tooltip>
      <DecisionReviewDialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        decisions={relevant}
      />
    </>
  );
}
