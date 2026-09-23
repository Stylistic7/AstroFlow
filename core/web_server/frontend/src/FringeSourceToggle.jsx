import { useState, useContext } from 'react';
import { Tooltip } from '@mui/material';
import BlurOnIcon from '@mui/icons-material/BlurOn';
import ToggleIconButton from './ToggleIconButton';
import AstroFlowContext from './AstroFlowContext.jsx';

// Promotes a stack of blank sky to a FRINGE SOURCE. Blank-sky frames arrive
// stamped `Light` like every other frame, so nothing in a header can identify
// them — this button is the only place that knowledge exists. Once promoted,
// the fringe frame built from these frames resolves onto lights of the same
// optical train the way a flat does, and is SUBTRACTED, scaled to each run's
// own sky (core/stacks/fringe.py).
export default function FringeSourceToggle({ stackId }) {
  const { isFringeSource, refreshDecisions } = useContext(AstroFlowContext);
  const [loading, setLoading] = useState(false);

  const promoted = isFringeSource(stackId);

  const handleToggle = async () => {
    setLoading(true);
    try {
      await fetch(`/api/fringe_source/${stackId}`, { method: 'POST' });
      await refreshDecisions();
    } finally {
      setLoading(false);
    }
  };

  return (
    <Tooltip title={promoted
      ? "Stop building a fringe frame from these frames"
      : "Blank sky: build a fringe frame from these frames"}>
      <span>
        <ToggleIconButton
          name="fringe-source"
          icon={<BlurOnIcon />}
          checked={promoted}
          handleChange={handleToggle}
          disabled={loading}
        />
      </span>
    </Tooltip>
  );
}
