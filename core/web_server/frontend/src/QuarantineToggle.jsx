import { useState } from 'react';
import { IconButton, Tooltip } from '@mui/material';
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';

/**
 * A toggle button for quarantining/un-quarantining an object.
 *
 * When quarantined, the object is excluded from downstream computations
 * (stacking, mosaic assembly, etc.) but remains visible in the UI.
 *
 * Props:
 *   objectType: "stack" | "target" | "mosaic"
 *   objectId: the stable_hash of the object
 *   isQuarantined: current quarantine state
 *   onToggle: optional callback after toggle completes
 */
export default function QuarantineToggle({ objectType, objectId, isQuarantined, onToggle }) {
  const [quarantined, setQuarantined] = useState(isQuarantined);
  const [loading, setLoading] = useState(false);

  const handleToggle = async () => {
    setLoading(true);
    try {
      const resp = await fetch(`/api/quarantine/${objectType}/${objectId}`, {
        method: 'POST',
      });
      const data = await resp.json();
      setQuarantined(data.is_quarantined);
      if (onToggle) onToggle(data.is_quarantined);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Tooltip title={quarantined ? "Un-quarantine (include in processing)" : "Quarantine (exclude from processing)"}>
      <IconButton
        onClick={handleToggle}
        disabled={loading}
        sx={{ opacity: quarantined ? 0.5 : 1 }}
      >
        {quarantined ? <VisibilityOffIcon /> : <VisibilityIcon />}
      </IconButton>
    </Tooltip>
  );
}
