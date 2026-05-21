import { useState, useContext } from 'react';
import { useParams } from 'react-router-dom';
import { IconButton, Tooltip } from '@mui/material';
import ShoppingCartIcon from '@mui/icons-material/ShoppingCart';
import ShoppingCartOutlinedIcon from '@mui/icons-material/ShoppingCartOutlined';
import AstroFlowContext from './AstroFlowContext.jsx';

// Legacy (implementation, quality) URL pair → canonical pipeline name.
// The cart entry is now a 3-tuple (type, id, pipeline_name); URLs
// still carry (implementation, quality) for the viewer pages until
// those are refactored, so we derive the pipeline name at cart-add
// time from whatever the user is currently looking at.
function legacyUrlParamsToPipelineName(implementation, quality) {
  if (!implementation && !quality) return null;
  if (quality === "Subframe") return "Subframe";
  if (quality === "DownscaledMaster") return `${implementation} Master Downscaled`;
  // "Master" / "PostProcessed" / no quality default
  return `${implementation} Master`;
}

export default function ShoppingCartMenu({ objectType, objectId }) {
  const { shoppingCart } = useContext(AstroFlowContext);
  const [loading, setLoading] = useState(false);

  // Pipeline derived from the current URL. The viewer routes today
  // carry `:implementation` and `:quality` separately; we combine them.
  // After the URL refactor (separate change), the URL will carry the
  // pipeline name directly.
  const { implementation, quality } = useParams();
  const pipelineName = legacyUrlParamsToPipelineName(implementation, quality);

  const cart = shoppingCart || [];
  const isQueued = cart.some(
    ([t, id, pipe]) => t === objectType && id === objectId && pipe === pipelineName,
  );

  const handleToggle = async () => {
    if (!pipelineName) return;
    setLoading(true);
    try {
      const method = isQueued ? 'DELETE' : 'POST';
      await fetch(
        `/api/shopping_cart/${objectType}/${objectId}/${encodeURIComponent(pipelineName)}`,
        { method },
      );
    } finally {
      setLoading(false);
    }
  };

  const tooltip = !pipelineName
    ? "Pick a pipeline first"
    : isQueued
      ? `Remove from queue: ${pipelineName}`
      : `Queue with ${pipelineName}`;

  return (
    <Tooltip title={tooltip}>
      <span>
        <IconButton
          size="small"
          onClick={handleToggle}
          disabled={loading || !pipelineName}
          sx={{ color: isQueued ? 'primary.main' : 'text.secondary' }}
        >
          {isQueued ? <ShoppingCartIcon /> : <ShoppingCartOutlinedIcon />}
        </IconButton>
      </span>
    </Tooltip>
  );
}
