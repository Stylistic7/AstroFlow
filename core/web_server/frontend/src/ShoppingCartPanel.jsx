import { useState, useContext } from "react";
import {
  Box,
  Menu,
  MenuItem,
  Typography,
  IconButton,
  Badge,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Chip,
} from "@mui/material";
import { ExpandMore, ExpandLess, Delete } from "@mui/icons-material";
import ShoppingCartIcon from "@mui/icons-material/ShoppingCart";
import ShoppingCartOutlinedIcon from "@mui/icons-material/ShoppingCartOutlined";
import AstroFlowContext from "./AstroFlowContext.jsx";

const qualityLabels = {
  DownscaledMaster: "Downscaled",
  Master: "Master",
  PostProcessed: "Post-processed",
};

const qualityColors = {
  DownscaledMaster: "secondary",
  Master: "primary",
  PostProcessed: "success",
};

export default function ShoppingCartPanel() {
  const { shoppingCart, stacks, targets, mosaics } = useContext(AstroFlowContext);
  const [anchorEl, setAnchorEl] = useState(null);

  const cart = shoppingCart || [];

  // Group cart entries by (type, id)
  const grouped = {};
  for (const [type, id, quality] of cart) {
    const key = `${type}:${id}`;
    if (!grouped[key]) {
      grouped[key] = { type, id, qualities: [] };
    }
    grouped[key].qualities.push(quality);
  }
  const items = Object.values(grouped);

  const handleRemove = async (type, id, quality) => {
    await fetch(`/api/shopping_cart/${type}/${id}/${quality}`, { method: "DELETE" });
  };

  const handleRemoveAll = async (type, id, qualities) => {
    for (const q of qualities) {
      await fetch(`/api/shopping_cart/${type}/${id}/${q}`, { method: "DELETE" });
    }
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
          items.map(({ type, id, qualities }) => (
            <MenuItem
              key={`${type}:${id}`}
              sx={{ display: "flex", flexDirection: "column", alignItems: "flex-start", py: 1 }}
              disableRipple
            >
              <Box sx={{ display: "flex", alignItems: "center", width: "100%", justifyContent: "space-between" }}>
                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                  {getName(type, id)}
                </Typography>
                <IconButton
                  size="small"
                  onClick={(e) => { e.stopPropagation(); handleRemoveAll(type, id, qualities); }}
                >
                  <Delete fontSize="small" />
                </IconButton>
              </Box>
              <Box sx={{ display: "flex", gap: 0.5, mt: 0.5 }}>
                <Chip size="small" label={type} variant="outlined" />
                {qualities.map((q) => (
                  <Chip
                    key={q}
                    size="small"
                    label={qualityLabels[q] || q}
                    color={qualityColors[q] || "default"}
                    onDelete={() => handleRemove(type, id, q)}
                  />
                ))}
              </Box>
            </MenuItem>
          ))
        )}
      </Menu>
    </>
  );
}
