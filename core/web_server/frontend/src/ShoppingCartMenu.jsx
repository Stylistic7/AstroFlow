import { useState, useContext } from 'react';
import { useParams } from 'react-router-dom';
import { IconButton, Popover, FormGroup, FormControlLabel, Checkbox, Tooltip, Typography } from '@mui/material';
import ShoppingCartIcon from '@mui/icons-material/ShoppingCart';
import ShoppingCartOutlinedIcon from '@mui/icons-material/ShoppingCartOutlined';
import AstroFlowContext from './AstroFlowContext.jsx';

const QUALITY_LEVELS = [
  { value: "DownscaledMaster", label: "Downscaled Master" },
  { value: "Master", label: "Master" },
  { value: "PostProcessed", label: "Post-processed" },
];

export default function ShoppingCartMenu({ objectType, objectId }) {
  const { shoppingCart } = useContext(AstroFlowContext);
  const [anchorEl, setAnchorEl] = useState(null);
  const [loading, setLoading] = useState(false);

  // The current pipeline comes from the URL's `:implementation` param —
  // when the user is on a Mosaic / Stack viewer page with a pipeline
  // pill selected, that pill's name flows into cart-adds from this
  // menu. Pages without a pipeline in the URL (e.g. the targets list)
  // get pipeline=undefined → backend defaults to legacy PixInsight.
  const { implementation: pipelineName } = useParams();

  const cart = shoppingCart || [];
  // Match cart entries for this object regardless of pipeline — the
  // checkbox shows "is this quality queued at all," not "is it queued
  // with the currently-active pipeline." Cleanest for the user; the
  // backend dispatcher stores the pipeline per-entry so two different
  // pipelines on the same object coexist as two separate entries.
  const checkedLevels = new Set(
    cart
      .filter(([t, id]) => t === objectType && id === objectId)
      .map(([, , quality]) => quality)
  );

  const hasAny = checkedLevels.size > 0;

  const handleToggle = async (quality) => {
    setLoading(true);
    try {
      const method = checkedLevels.has(quality) ? 'DELETE' : 'POST';
      // Pipeline name rides as the optional 4th URL segment so the
      // backend can store it on the cart entry. Omitted entirely when
      // the user is on a no-pipeline page (legacy default).
      const suffix = pipelineName ? `/${pipelineName}` : '';
      await fetch(
        `/api/shopping_cart/${objectType}/${objectId}/${quality}${suffix}`,
        { method },
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <Tooltip title="Processing queue">
        <IconButton
          size="small"
          onClick={(e) => setAnchorEl(e.currentTarget)}
          sx={{ color: hasAny ? 'primary.main' : 'text.secondary' }}
        >
          {hasAny ? <ShoppingCartIcon /> : <ShoppingCartOutlinedIcon />}
        </IconButton>
      </Tooltip>
      <Popover
        open={Boolean(anchorEl)}
        anchorEl={anchorEl}
        onClose={() => setAnchorEl(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
      >
        <FormGroup sx={{ px: 2, py: 1 }}>
          <Typography variant="caption" sx={{ mb: 0.5, color: 'text.secondary' }}>
            Process at quality:
          </Typography>
          {QUALITY_LEVELS.map(({ value, label }) => (
            <FormControlLabel
              key={value}
              control={
                <Checkbox
                  size="small"
                  checked={checkedLevels.has(value)}
                  onChange={() => handleToggle(value)}
                  disabled={loading}
                />
              }
              label={<Typography variant="body2">{label}</Typography>}
            />
          ))}
        </FormGroup>
      </Popover>
    </>
  );
}
