import { useState, useContext } from "react";
import {
  Box,
  MenuItem,
  Typography,
  IconButton,
  Chip,
  Divider,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Button,
  Tooltip,
} from "@mui/material";
import {
  Delete, Unarchive, Archive,
} from "@mui/icons-material";
import ShoppingCartIcon from "@mui/icons-material/ShoppingCart";
import ShoppingCartOutlinedIcon from "@mui/icons-material/ShoppingCartOutlined";
import AstroFlowContext from "./AstroFlowContext.jsx";
import DropdownChip from "./DropdownChip.jsx";


// Removing an item from the live cart pops this modal. The three
// choices map 1:1 to the backend's `preserve` field:
//   - "all"    → snapshot every traced path into context.archive
//   - "finals" → snapshot only final-tagged paths (@makes_image final=True)
//   - "none"   → no archive entry; everything orphan-eligible next sweep
function RemoveCartItemDialog({ item, open, onClose, onConfirm }) {
  if (!item) return null;
  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Remove from cart</DialogTitle>
      <DialogContent>
        <DialogContentText>
          What should happen to this item's on-disk outputs? You can
          always move the item back to the cart later and re-run the
          pipeline to regenerate missing intermediates.
        </DialogContentText>
        <Box sx={{ mt: 2 }}>
          <Typography variant="body2" sx={{ fontWeight: 500 }}>
            {item.name}
          </Typography>
          <Box sx={{ display: "flex", gap: 0.5, mt: 0.5 }}>
            <Chip size="small" label={item.type} variant="outlined" />
            <Chip size="small" label={item.pipeline} variant="outlined" color="info" />
          </Box>
        </Box>
      </DialogContent>
      <DialogActions sx={{ flexWrap: "wrap", gap: 1, px: 3, pb: 2 }}>
        <Button onClick={onClose}>Cancel</Button>
        <Box sx={{ flexGrow: 1 }} />
        <Button
          onClick={() => onConfirm("none")}
          color="warning"
        >
          Just remove (delete outputs)
        </Button>
        <Button
          onClick={() => onConfirm("all")}
        >
          Archive everything
        </Button>
        <Button
          onClick={() => onConfirm("finals")}
          variant="contained"
        >
          Archive finals only
        </Button>
      </DialogActions>
    </Dialog>
  );
}


// Confirmation when removing an archived item — the preserved paths
// become orphan-eligible on the next sweep.
function RemoveArchiveItemDialog({ item, open, onClose, onConfirm }) {
  if (!item) return null;
  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Remove from archive</DialogTitle>
      <DialogContent>
        <DialogContentText>
          This will mark {item.pathCount} preserved
          {item.pathCount === 1 ? " path" : " paths"} as orphan-
          eligible. The next sweep will delete them.
        </DialogContentText>
        <Box sx={{ mt: 2 }}>
          <Typography variant="body2" sx={{ fontWeight: 500 }}>
            {item.name}
          </Typography>
          <Box sx={{ display: "flex", gap: 0.5, mt: 0.5 }}>
            <Chip size="small" label={item.type} variant="outlined" />
            <Chip size="small" label={item.pipeline} variant="outlined" color="info" />
          </Box>
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button onClick={onConfirm} color="warning" variant="contained">
          Delete
        </Button>
      </DialogActions>
    </Dialog>
  );
}


export default function ShoppingCartPanel() {
  const { shoppingCart, archive, stacks, targets, mosaics } =
    useContext(AstroFlowContext);
  const [removeCartTarget, setRemoveCartTarget] = useState(null);
  const [removeArchiveTarget, setRemoveArchiveTarget] = useState(null);

  const cart = shoppingCart || [];
  const arch = archive || [];

  const getName = (type, id) => {
    const sources = { stack: stacks, target: targets, mosaic: mosaics };
    const obj = sources[type]?.[id];
    return obj?.name || id.slice(0, 8);
  };

  const cartItems = cart.map(([type, id, pipeline]) => ({
    type, id, pipeline, name: getName(type, id),
  }));
  const archItems = arch.map(([type, id, pipeline, paths]) => ({
    type, id, pipeline, name: getName(type, id),
    pathCount: (paths || []).length,
  }));

  const cartUrl = (type, id, pipeline) =>
    `/api/shopping_cart/${type}/${id}/${encodeURIComponent(pipeline)}`;
  const archiveUrl = (type, id, pipeline) =>
    `/api/archive/${type}/${id}/${encodeURIComponent(pipeline)}`;

  const confirmRemoveCart = async (preserve) => {
    const item = removeCartTarget;
    setRemoveCartTarget(null);
    await fetch(cartUrl(item.type, item.id, item.pipeline), {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ preserve }),
    });
  };

  const confirmRemoveArchive = async () => {
    const item = removeArchiveTarget;
    setRemoveArchiveTarget(null);
    await fetch(archiveUrl(item.type, item.id, item.pipeline), {
      method: "DELETE",
    });
  };

  const moveArchiveToCart = async (item) => {
    await fetch(`${archiveUrl(item.type, item.id, item.pipeline)}/to_cart`, {
      method: "POST",
    });
  };

  const totalCount = cartItems.length + archItems.length;

  // Summary slot. When the cart's empty we still show a muted label so
  // the chip carries the same information density as the other panels
  // (otherwise it'd be an icon floating in space).
  const summaryNode = totalCount === 0
    ? <Typography variant="body2" color="text.secondary">empty</Typography>
    : <Typography variant="body2">
        {`${cartItems.length} live · ${archItems.length} archived`}
      </Typography>;

  return (
    <>
      <DropdownChip
        icon={cartItems.length > 0 ? <ShoppingCartIcon /> : <ShoppingCartOutlinedIcon />}
        summary={summaryNode}
        tone={cartItems.length > 0 ? "active" : "default"}
        tooltipTitle="Shopping cart"
        ariaLabel="Shopping cart"
      >
        <MenuItem disabled sx={{ opacity: "1 !important" }}>
          <Typography variant="caption" sx={{ fontWeight: 600 }}>
            LIVE CART
          </Typography>
        </MenuItem>
        {cartItems.length === 0 ? (
          <MenuItem disabled>
            <Typography variant="body2" color="text.secondary">
              (empty)
            </Typography>
          </MenuItem>
        ) : (
          cartItems.map((it) => (
            <MenuItem
              key={`cart:${it.type}:${it.id}:${it.pipeline}`}
              sx={{ display: "flex", flexDirection: "column", alignItems: "flex-start", py: 1 }}
              disableRipple
            >
              <Box sx={{ display: "flex", alignItems: "center", width: "100%", justifyContent: "space-between" }}>
                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                  {it.name}
                </Typography>
                <Tooltip title="Remove…">
                  <IconButton
                    size="small"
                    onClick={(e) => { e.stopPropagation(); setRemoveCartTarget(it); }}
                  >
                    <Delete fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>
              <Box sx={{ display: "flex", gap: 0.5, mt: 0.5, flexWrap: "wrap" }}>
                <Chip size="small" label={it.type} variant="outlined" />
                <Chip size="small" label={it.pipeline} variant="outlined" color="info" />
              </Box>
            </MenuItem>
          ))
        )}

        <Divider sx={{ my: 0.5 }} />

        <MenuItem disabled sx={{ opacity: "1 !important" }}>
          <Typography variant="caption" sx={{ fontWeight: 600 }}>
            ARCHIVE
          </Typography>
        </MenuItem>
        {archItems.length === 0 ? (
          <MenuItem disabled>
            <Typography variant="body2" color="text.secondary">
              (empty)
            </Typography>
          </MenuItem>
        ) : (
          archItems.map((it) => (
            <MenuItem
              key={`arch:${it.type}:${it.id}:${it.pipeline}`}
              sx={{ display: "flex", flexDirection: "column", alignItems: "flex-start", py: 1 }}
              disableRipple
            >
              <Box sx={{ display: "flex", alignItems: "center", width: "100%", justifyContent: "space-between" }}>
                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                  {it.name}
                </Typography>
                <Box>
                  <Tooltip title="Move back to live cart">
                    <IconButton
                      size="small"
                      onClick={(e) => { e.stopPropagation(); moveArchiveToCart(it); }}
                    >
                      <Unarchive fontSize="small" />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="Remove from archive">
                    <IconButton
                      size="small"
                      onClick={(e) => { e.stopPropagation(); setRemoveArchiveTarget(it); }}
                    >
                      <Delete fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </Box>
              </Box>
              <Box sx={{ display: "flex", gap: 0.5, mt: 0.5, flexWrap: "wrap" }}>
                <Chip size="small" label={it.type} variant="outlined" />
                <Chip size="small" label={it.pipeline} variant="outlined" color="info" />
                <Chip
                  size="small"
                  icon={<Archive sx={{ fontSize: 14 }} />}
                  label={`${it.pathCount} path${it.pathCount === 1 ? "" : "s"}`}
                  variant="outlined"
                />
              </Box>
            </MenuItem>
          ))
        )}
      </DropdownChip>

      <RemoveCartItemDialog
        item={removeCartTarget}
        open={Boolean(removeCartTarget)}
        onClose={() => setRemoveCartTarget(null)}
        onConfirm={confirmRemoveCart}
      />
      <RemoveArchiveItemDialog
        item={removeArchiveTarget}
        open={Boolean(removeArchiveTarget)}
        onClose={() => setRemoveArchiveTarget(null)}
        onConfirm={confirmRemoveArchive}
      />
    </>
  );
}
