import { useCallback, useContext, useEffect, useState } from 'react';
import { IconButton, Tooltip, Badge } from '@mui/material';
import LightbulbIcon from '@mui/icons-material/Lightbulb';

import AstroFlowContext from '../AstroFlowContext.jsx';
import SuggestionAcceptDialog from './SuggestionAcceptDialog.jsx';

/**
 * Lightbulb badge shown beside DecisionBadge on an object that has
 * at least one unaccepted suggestion. Clicking opens the accept dialog.
 *
 * Props:
 *   objectType: "target" | "mosaic"
 *   objectId: stable_hash
 */
export default function SuggestionBadge({ objectType, objectId }) {
  const [suggestions, setSuggestions] = useState([]);
  const [dialogOpen, setDialogOpen] = useState(false);
  const ctx = useContext(AstroFlowContext);
  const planEntries = ctx?.observingPlan?.entries || [];
  const acceptedKeys = new Set(planEntries.map((e) => e.suggestion_key));

  const load = useCallback(async () => {
    try {
      const res = await fetch(`/api/suggestions/by_object/${objectType}/${objectId}`);
      const data = await res.json();
      setSuggestions(data.suggestions || []);
    } catch (e) {
      console.warn(e);
    }
  }, [objectType, objectId]);

  useEffect(() => {
    load();
  }, [load]);

  const unaccepted = suggestions.filter((s) => !acceptedKeys.has(s.stable_key));
  if (unaccepted.length === 0) return null;

  return (
    <>
      <Tooltip title={`${unaccepted.length} suggestion${unaccepted.length > 1 ? 's' : ''} to consider`}>
        <IconButton
          size="small"
          onClick={(e) => { e.stopPropagation(); setDialogOpen(true); }}
          sx={{
            color: 'rgba(255,220,100,0.85)',
            '&:hover': { color: 'rgba(255,230,140,1.0)' },
            padding: '2px',
          }}
        >
          <Badge
            badgeContent={unaccepted.length}
            max={99}
            sx={{
              '& .MuiBadge-badge': {
                bgcolor: 'rgba(220,160,40,0.9)',
                color: '#fff',
                fontSize: 9,
                minWidth: 14,
                height: 14,
              },
            }}
          >
            <LightbulbIcon sx={{ fontSize: 28 }} />
          </Badge>
        </IconButton>
      </Tooltip>
      <SuggestionAcceptDialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        suggestions={suggestions}
        acceptedKeys={acceptedKeys}
        onChanged={ctx?.refreshObservingPlan}
      />
    </>
  );
}
