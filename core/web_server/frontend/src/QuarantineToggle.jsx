import { useState, useContext } from 'react';
import { Tooltip } from '@mui/material';
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';
import ToggleIconButton from './ToggleIconButton';
import AstroFlowContext from './AstroFlowContext.jsx';

export default function QuarantineToggle({ objectType, objectId }) {
  const { isQuarantined, refreshDecisions } = useContext(AstroFlowContext);
  const [loading, setLoading] = useState(false);

  const quarantined = isQuarantined(objectType, objectId);

  const handleToggle = async () => {
    setLoading(true);
    try {
      await fetch(`/api/quarantine/${objectType}/${objectId}`, {
        method: 'POST',
      });
      await refreshDecisions();
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
