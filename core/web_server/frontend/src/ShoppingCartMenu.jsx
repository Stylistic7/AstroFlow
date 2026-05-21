// ShoppingCartMenu: click the cart icon → dropdown of registered
// pipelines. Each row toggles `(objectType, objectId, pipelineName)`
// in the cart. A checkmark indicates the pair is already queued.
//
// Replaces the previous URL-derived behavior (which grayed out the
// button on dashboard pages where no pipeline was selected). The
// pipeline is now picked explicitly from this menu.
import { useState, useContext, useRef } from 'react';
import {
  IconButton, Tooltip, Menu, MenuItem, Checkbox, ListItemText,
} from '@mui/material';
import ShoppingCartIcon from '@mui/icons-material/ShoppingCart';
import ShoppingCartOutlinedIcon from '@mui/icons-material/ShoppingCartOutlined';
import AstroFlowContext from './AstroFlowContext.jsx';
import { usePipelines } from './useVersions.js';


export default function ShoppingCartMenu({ objectType, objectId }) {
  const { shoppingCart } = useContext(AstroFlowContext);
  const pipelines = usePipelines();
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const anchorRef = useRef(null);

  const cart = shoppingCart || [];
  const queuedPipelines = new Set(
    cart
      .filter(([t, id]) => t === objectType && id === objectId)
      .map(([, , pipe]) => pipe),
  );
  const anyQueued = queuedPipelines.size > 0;

  const toggle = async (pipelineName) => {
    if (busy) return;
    setBusy(true);
    try {
      const isQueued = queuedPipelines.has(pipelineName);
      await fetch(
        `/api/shopping_cart/${objectType}/${objectId}/${encodeURIComponent(pipelineName)}`,
        { method: isQueued ? 'DELETE' : 'POST' },
      );
    } finally {
      setBusy(false);
    }
  };

  const tooltip = anyQueued
    ? `Queued: ${[...queuedPipelines].join(', ')}`
    : 'Add to processing queue';

  return (
    <>
      <Tooltip title={tooltip}>
        <IconButton
          ref={anchorRef}
          size="small"
          onClick={() => setOpen((o) => !o)}
          sx={{ color: anyQueued ? 'primary.main' : 'text.secondary' }}
        >
          {anyQueued ? <ShoppingCartIcon /> : <ShoppingCartOutlinedIcon />}
        </IconButton>
      </Tooltip>
      <Menu
        anchorEl={anchorRef.current}
        open={open}
        onClose={() => setOpen(false)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
      >
        {pipelines.length === 0 ? (
          <MenuItem disabled>No pipelines registered</MenuItem>
        ) : (
          pipelines.map((p) => {
            const isQueued = queuedPipelines.has(p.name);
            return (
              <MenuItem
                key={p.name}
                onClick={() => toggle(p.name)}
                disabled={busy}
                dense
              >
                <Checkbox
                  edge="start"
                  checked={isQueued}
                  tabIndex={-1}
                  disableRipple
                  size="small"
                />
                <ListItemText>{p.name}</ListItemText>
              </MenuItem>
            );
          })
        )}
      </Menu>
    </>
  );
}
