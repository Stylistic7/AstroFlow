import { useState, useContext } from 'react';
import { Tooltip } from '@mui/material';
import LinkIcon from '@mui/icons-material/Link';
import LinkOffIcon from '@mui/icons-material/LinkOff';
import ToggleIconButton from './ToggleIconButton';
import AstroFlowContext from './AstroFlowContext.jsx';

// Mosaic membership is inferred from footprint overlap; only the user knows
// two overlapping fields are different projects. One click records the
// decision "standalone" and re-identifies; the same button, now lit, undoes
// it. Shown only on a target that is in a mosaic or has been kept out of one.
export default function MosaicMembershipToggle({ targetId }) {
  const { mosaics, standaloneTargets, refreshDecisions } = useContext(AstroFlowContext);
  const [loading, setLoading] = useState(false);

  const standalone = standaloneTargets.has(targetId);
  const entry = Object.entries(mosaics || {}).find(([, m]) => m.target_ids?.includes(targetId));
  if (!standalone && !entry) return null;
  const label = entry ? `${entry[0].slice(0, 4)} ${entry[1].name}` : null;

  const handleToggle = async () => {
    setLoading(true);
    try {
      await fetch(`/api/mosaic_membership/target/${targetId}`, { method: standalone ? 'DELETE' : 'PUT' });
      await refreshDecisions();
    } finally {
      setLoading(false);
    }
  };

  return (
    <Tooltip title={standalone
      ? "Standalone: kept out of every mosaic. Click to let it join one again."
      : `Remove this Target from the ${label} mosaic (it becomes a project of its own)`}>
      <span>
        <ToggleIconButton
          name="mosaic-membership"
          icon={standalone ? <LinkOffIcon /> : <LinkIcon />}
          checked={standalone}
          handleChange={handleToggle}
          disabled={loading}
        />
      </span>
    </Tooltip>
  );
}
