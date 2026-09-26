import { useState, useContext } from 'react';
import { Tooltip } from '@mui/material';
import BlurOnIcon from '@mui/icons-material/BlurOn';
import ToggleIconButton from './ToggleIconButton';
import AstroFlowContext from './AstroFlowContext.jsx';

// Promotes a stack of blank sky to a NIGHT-SKY FLAT source. Blank-sky frames
// arrive stamped `Light` like every other frame, so nothing in a header can
// identify them — this button is the only place that knowledge exists. Once
// promoted, the flat built from these frames is an ordinary flat: it appears
// in the flat picker, resolves onto lights of the same optical train ahead of
// the star flat, and is DIVIDED, re-scaled to each stack's own sky
// (core/stacks/sky_flat.py).
export default function SkyFlatSourceToggle({ stackId }) {
  const { isSkyFlatSource, refreshDecisions } = useContext(AstroFlowContext);
  const [loading, setLoading] = useState(false);

  const promoted = isSkyFlatSource(stackId);

  const handleToggle = async () => {
    setLoading(true);
    try {
      await fetch(`/api/sky_flat_source/${stackId}`, { method: 'POST' });
      await refreshDecisions();
    } finally {
      setLoading(false);
    }
  };

  return (
    <Tooltip title={promoted
      ? "Stop building a night-sky flat from these frames"
      : "Blank sky: build a night-sky flat from these frames"}>
      <span>
        <ToggleIconButton
          name="sky-flat-source"
          icon={<BlurOnIcon />}
          checked={promoted}
          handleChange={handleToggle}
          disabled={loading}
        />
      </span>
    </Tooltip>
  );
}
