import { useState } from 'react';
import { Tooltip } from '@mui/material';
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';
import ToggleIconButton from './ToggleIconButton';

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
      <span>
        <ToggleIconButton
          name="quarantine"
          icon={quarantined ? <VisibilityOffIcon /> : <VisibilityIcon />}
          checked={quarantined}
          handleChange={handleToggle}
          disabled={loading}
        />
      </span>
    </Tooltip>
  );
}
