import { useState, useContext } from "react";
import {
  Box,
  Menu,
  MenuItem,
  Typography,
  IconButton,
  Badge,
  Chip,
} from "@mui/material";
import { ExpandMore, ExpandLess, Delete } from "@mui/icons-material";
import ShoppingCartIcon from "@mui/icons-material/ShoppingCart";
import ShoppingCartOutlinedIcon from "@mui/icons-material/ShoppingCartOutlined";
import AstroFlowContext from "./AstroFlowContext.jsx";

export default function ShoppingCartPanel() {
  const { shoppingCart, stacks, targets, mosaics } = useContext(AstroFlowContext);
  const [anchorEl, setAnchorEl] = useState(null);

  // Cart entries are now `(type, id, pipeline_name)` 3-tuples. Each
  // entry is one row; pipeline is the variant axis.
  const cart = shoppingCart || [];
  const items = cart.map(([type, id, pipeline]) => ({ type, id, pipeline }));

  const cartUrl = (type, id, pipeline) =>
    `/api/shopping_cart/${type}/${id}/${encodeURIComponent(pipeline)}`;

  const handleRemove = async (type, id, pipeline) => {
    await fetch(cartUrl(type, id, pipeline), { method: "DELETE" });
  };

  const getName = (type, id) => {
    const sources = { stack: stacks, target: targets, mosaic: mosaics };
    const obj = sources[type]?.[id];
    return obj?.name || id.slice(0, 8);
  };

  return (
    <>
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          cursor: "pointer",
          padding: "8px",
          borderRadius: "8px",
        }}
        onClick={(e) => setAnchorEl(e.currentTarget)}
      >
        <Badge badgeContent={items.length} color="primary" max={99}>
          {items.length > 0 ? (
            <ShoppingCartIcon sx={{ color: "primary.main" }} />
          ) : (
            <ShoppingCartOutlinedIcon sx={{ color: "text.secondary" }} />
          )}
        </Badge>
        {items.length > 0 && (
          <Typography variant="body2" sx={{ ml: 1 }}>
            {items.length} {items.length === 1 ? "item" : "items"}
          </Typography>
        )}
        <IconButton size="small">
          {anchorEl ? <ExpandLess /> : <ExpandMore />}
        </IconButton>
      </Box>

      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={() => setAnchorEl(null)}
        PaperProps={{ sx: { minWidth: 320, maxHeight: 400 } }}
      >
        {items.length === 0 ? (
          <MenuItem disabled>
            <Typography variant="body2" color="text.secondary">
              Shopping cart is empty
            </Typography>
          </MenuItem>
        ) : (
          items.map(({ type, id, pipeline }) => (
            <MenuItem
              key={`${type}:${id}:${pipeline}`}
              sx={{ display: "flex", flexDirection: "column", alignItems: "flex-start", py: 1 }}
              disableRipple
            >
              <Box sx={{ display: "flex", alignItems: "center", width: "100%", justifyContent: "space-between" }}>
                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                  {getName(type, id)}
                </Typography>
                <IconButton
                  size="small"
                  onClick={(e) => { e.stopPropagation(); handleRemove(type, id, pipeline); }}
                >
                  <Delete fontSize="small" />
                </IconButton>
              </Box>
              <Box sx={{ display: "flex", gap: 0.5, mt: 0.5, flexWrap: "wrap" }}>
                <Chip size="small" label={type} variant="outlined" />
                <Chip size="small" label={pipeline} variant="outlined" color="info" />
              </Box>
            </MenuItem>
          ))
        )}
      </Menu>
    </>
  );
}
