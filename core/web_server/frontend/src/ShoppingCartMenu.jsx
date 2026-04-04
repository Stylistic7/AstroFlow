import { useState, useContext } from 'react';
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

  const cart = shoppingCart || [];
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
      await fetch(`/api/shopping_cart/${objectType}/${objectId}/${quality}`, { method });
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
