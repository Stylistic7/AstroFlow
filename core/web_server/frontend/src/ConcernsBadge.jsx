import { useState, useEffect, useCallback } from 'react';
import { IconButton, Tooltip, Badge } from '@mui/material';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import ConcernsDialog from './ConcernsDialog.jsx';

/**
 * Shows a "!" badge on objects whose most recent producer run flagged
 * one or more "concerns" — non-fatal anomalies the user should review
 * (e.g., WBPP dropped frames, plate-solving failed on most subframes).
 *
 * Yellow when the highest severity is warning, red when error.
 * Fetches /api/concerns/<type>/<id>?pipeline=&display=&filter=.
 *
 * Props:
 *   objectType: "stack" | "target" | "mosaic"
 *   objectId: the stable_hash of the object
 *   pipeline: e.g., "PixInsight Master" (default)
 *   display: e.g., "Master" (default)
 *   filter: required for objectType="mosaic"
 */
export default function ConcernsBadge({
  objectType,
  objectId,
  pipeline = "PixInsight Master",
  display = "Master",
  filter,
}) {
  const [concerns, setConcerns] = useState([]);
  const [dialogOpen, setDialogOpen] = useState(false);

  const load = useCallback(async () => {
    try {
      const params = new URLSearchParams({ pipeline, display });
      if (filter) params.set('filter', filter);
      const url = `/api/concerns/${objectType}/${objectId}?${params}`;
      const res = await fetch(url);
      if (!res.ok) { setConcerns([]); return; }
      const data = await res.json();
      setConcerns(Array.isArray(data) ? data : []);
    } catch (e) { console.warn(e); }
  }, [objectType, objectId, pipeline, display, filter]);

  useEffect(() => { load(); }, [load]);

  if (concerns.length === 0) return null;

  const hasError = concerns.some(c => c._concern === "error");
  const color = hasError ? 'rgba(255,80,80,0.95)' : 'rgba(255,196,0,0.95)';
  const label = `${concerns.length} concern${concerns.length > 1 ? 's' : ''} to review`;

  return (
    <>
      <Tooltip title={label}>
        <IconButton
          size="small"
          onClick={(e) => { e.stopPropagation(); setDialogOpen(true); }}
          sx={{
            color,
            '&:hover': { color: hasError ? 'rgba(255,120,120,1)' : 'rgba(255,220,80,1)' },
            padding: '2px',
          }}
        >
          <Badge
            badgeContent={concerns.length}
            max={99}
            sx={{
              '& .MuiBadge-badge': {
                bgcolor: color,
                color: '#000',
                fontSize: 9,
                minWidth: 14,
                height: 14,
              },
            }}
          >
            <WarningAmberIcon sx={{ fontSize: 28 }} />
          </Badge>
        </IconButton>
      </Tooltip>
      <ConcernsDialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        concerns={concerns}
      />
    </>
  );
}
