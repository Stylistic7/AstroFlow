import { useState, useContext, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  Box,
  MenuItem,
  Typography,
  IconButton,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Button,
  Tooltip,
} from "@mui/material";
import { Delete, Star, StarBorder } from "@mui/icons-material";
import ShoppingCartIcon from "@mui/icons-material/ShoppingCart";
import ShoppingCartOutlinedIcon from "@mui/icons-material/ShoppingCartOutlined";
import AstroFlowContext from "./AstroFlowContext.jsx";
import DropdownChip from "./DropdownChip.jsx";


const gb = (bytes) => `${(bytes / 1e9).toFixed(bytes >= 1e10 ? 0 : 1)} GB`;

// The cart is the declaration of what is kept: removing an entry deletes its
// products on the next pass. The dialog says what that frees — the footprint
// the last pass's orphan sweep measured — before the user confirms.
function RemoveCartItemDialog({ item, open, onClose, onConfirm }) {
  const [footprint, setFootprint] = useState(null);
  useEffect(() => {
    if (!item) return;
    setFootprint(null);
    fetch("/api/shopping_cart/footprints")
      .then((r) => (r.ok ? r.json() : {}))
      .then((f) => setFootprint(f[`${item.type}/${item.id}/${item.pipeline}`] ?? 0))
      .catch(() => setFootprint(0));
  }, [item]);
  if (!item) return null;
  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Remove from cart</DialogTitle>
      <DialogContent>
        <DialogContentText>
          Its products are deleted on the next pass
          {footprint === null ? "" : footprint > 0 ? ` — about ${gb(footprint)} on disk` : " (nothing on disk yet)"}.
          Raw frames are never touched; putting it back in the cart rebuilds everything.
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
          Remove
        </Button>
      </DialogActions>
    </Dialog>
  );
}


export default function ShoppingCartPanel() {
  const { shoppingCart, stacks, targets, mosaics, plannerStars } =
    useContext(AstroFlowContext);
  const navigate = useNavigate();
  const [removeCartTarget, setRemoveCartTarget] = useState(null);

  const cart = shoppingCart || [];

  const getName = (type, id) => {
    const sources = { stack: stacks, target: targets, mosaic: mosaics };
    const obj = sources[type]?.[id];
    return obj?.name || id.slice(0, 8);
  };

  const cartItems = cart.map(([type, id, pipeline]) => ({
    type, id, pipeline, name: getName(type, id),
  }));

  // One row per OBJECT, not per (object, pipeline) pair. Running the same
  // target through two pipelines is the A/B workflow, so it should read as
  // "Heart Nebula, two pipelines" rather than as two unrelated rows that
  // happen to share a name. Insertion order is preserved so the list doesn't
  // reshuffle when a pipeline is added to an object already in the cart.
  const cartGroups = [];
  const groupByKey = new Map();
  for (const it of cartItems) {
    const key = `${it.type}:${it.id}`;
    let group = groupByKey.get(key);
    if (!group) {
      group = { type: it.type, id: it.id, name: it.name, pipelines: [] };
      groupByKey.set(key, group);
      cartGroups.push(group);
    }
    group.pipelines.push(it.pipeline);
  }

  const cartUrl = (type, id, pipeline) =>
    `/api/shopping_cart/${type}/${id}/${encodeURIComponent(pipeline)}`;

  const confirmRemoveCart = async () => {
    const item = removeCartTarget;
    setRemoveCartTarget(null);
    await fetch(cartUrl(item.type, item.id, item.pipeline), { method: "DELETE" });
  };

  // The star is priority, in the sky and in the cluster: the planner gives
  // starred objects every hour they can still use, and the compute graph runs
  // their pipelines before unstarred work. Stars, not a ranking.
  const isStarred = (g) => (plannerStars || []).some(([t, id]) => t === g.type && id === g.id);
  const toggleStar = async (g) => {
    await fetch(`/api/planner/star/${g.type}/${g.id}`, { method: isStarred(g) ? "DELETE" : "PUT" });
  };

  // Summary slot. When the cart's empty we still show a muted label so
  // the chip carries the same information density as the other panels
  // (otherwise it'd be an icon floating in space).
  const starred = cartGroups.filter(isStarred).length;
  const summaryNode = cartItems.length === 0
    ? <Typography variant="body2" color="text.secondary">empty</Typography>
    : <Typography variant="body2">
        {`${cartItems.length} in the cart${starred ? ` · ${starred} starred` : ""}`}
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
        {({ close }) => (
          <>
            <MenuItem disabled sx={{ opacity: "1 !important" }}>
              <Typography variant="caption" sx={{ fontWeight: 600 }}>
                IN THE CART — processed, accumulating, kept
              </Typography>
            </MenuItem>
            {cartItems.length === 0 ? (
              <MenuItem disabled>
                <Typography variant="body2" color="text.secondary">
                  (empty)
                </Typography>
              </MenuItem>
            ) : (
              cartGroups.map((g) => {
                // The cart's unit of removal is (object, pipeline), so with
                // several pipelines the delete belongs on each CHIP — a single
                // row-level trash would be ambiguous about which build it
                // drops. With one pipeline the familiar row-level trash stays,
                // so the common case is unchanged and there is never more than
                // one delete affordance for the same thing.
                const multi = g.pipelines.length > 1;
                const itemFor = (pipeline) => ({
                  type: g.type, id: g.id, name: g.name, pipeline,
                });
                return (
                  <MenuItem
                    key={`cart:${g.type}:${g.id}`}
                    sx={{ display: "flex", flexDirection: "column", alignItems: "flex-start", py: 1 }}
                    onClick={() => { close(); navigate(`/${g.type}/${g.id}`); }}
                  >
                    <Box sx={{ display: "flex", alignItems: "center", width: "100%", justifyContent: "space-between" }}>
                      <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                        <Tooltip title={isStarred(g)
                          ? "Starred: shot and processed before anything unstarred"
                          : "Star: shoot and process this before anything unstarred"}>
                          <IconButton
                            size="small"
                            color={isStarred(g) ? "warning" : "default"}
                            onClick={(e) => { e.stopPropagation(); toggleStar(g); }}
                          >
                            {isStarred(g) ? <Star fontSize="small" /> : <StarBorder fontSize="small" />}
                          </IconButton>
                        </Tooltip>
                        <Typography variant="body2" sx={{ fontWeight: isStarred(g) ? 700 : 500 }}>
                          {g.name}
                        </Typography>
                      </Box>
                      <Box sx={{ display: "flex", alignItems: "center" }}>
                      {!multi && (
                        <Tooltip title="Remove…">
                          <IconButton
                            size="small"
                            onClick={(e) => {
                              e.stopPropagation();
                              setRemoveCartTarget(itemFor(g.pipelines[0]));
                            }}
                          >
                            <Delete fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      )}
                      </Box>
                    </Box>
                    <Box sx={{ display: "flex", gap: 0.5, mt: 0.5, flexWrap: "wrap" }}>
                      <Chip size="small" label={g.type} variant="outlined" />
                      {g.pipelines.map((pipeline) => (
                        <Chip
                          key={pipeline}
                          size="small"
                          label={pipeline}
                          variant="outlined"
                          color="info"
                          {...(multi ? {
                            onDelete: (e) => {
                              e.stopPropagation();
                              setRemoveCartTarget(itemFor(pipeline));
                            },
                            deleteIcon: (
                              <Delete
                                titleAccess={`Remove ${pipeline}`}
                                onMouseDown={(e) => e.stopPropagation()}
                              />
                            ),
                          } : {})}
                        />
                      ))}
                    </Box>
                  </MenuItem>
                );
              })
            )}
          </>
        )}
      </DropdownChip>

      <RemoveCartItemDialog
        item={removeCartTarget}
        open={Boolean(removeCartTarget)}
        onClose={() => setRemoveCartTarget(null)}
        onConfirm={confirmRemoveCart}
      />
    </>
  );
}
