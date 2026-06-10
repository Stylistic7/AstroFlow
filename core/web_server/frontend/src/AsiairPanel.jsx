import React, { useState, useContext } from "react";
import {
  Box,
  Menu,
  MenuItem,
  Typography,
  IconButton,
  List,
  ListItem,
  ListItemText,
  Divider,
  TextField,
  Button,
  Tooltip,
} from "@mui/material";
import { ExpandMore, ExpandLess, Circle, Wifi } from "@mui/icons-material";
import AstroFlowContext from "./AstroFlowContext.jsx";


const _STATE_COLOR = {
  green: "success.light",
  yellow: "warning.light",
  gray: "text.disabled",
};


const _formatRelativeTime = (epochSeconds) => {
  if (!epochSeconds) return "no images yet";
  const ageS = Math.max(0, Date.now() / 1000 - epochSeconds);
  if (ageS < 60) return "moments ago";
  if (ageS < 60 * 60) return `${Math.round(ageS / 60)} min ago`;
  if (ageS < 60 * 60 * 24) return `${Math.round(ageS / 3600)} h ago`;
  return `${Math.round(ageS / 86400)} d ago`;
};


const RigRow = ({ rig, onRename }) => {
  // Local UI state — expand/collapse for the diagnostics block,
  // and a draft nickname so the input doesn't fight the canonical
  // value from props as the user types.
  const [expanded, setExpanded] = useState(false);
  const [draftNickname, setDraftNickname] = useState(rig.nickname || "");

  const titleLine = rig.anchored
    ? (rig.nickname || `Camera ${rig.camera_hash_short}`)
    : `Awaiting fingerprint`;

  // Secondary line: last-image relative time + filter + target +
  // which ASIAir we currently believe is delivering for this camera.
  // Each segment is optional — only render the ones we have data for,
  // joined by " · ", to avoid the empty "·  · " runs that look broken.
  const secondaryParts = [];
  if (rig.last_image_mtime) {
    secondaryParts.push(`last image: ${_formatRelativeTime(rig.last_image_mtime)}`);
  }
  if (rig.last_filter) secondaryParts.push(`filter ${rig.last_filter}`);
  if (rig.last_object) secondaryParts.push(`target ${rig.last_object}`);
  if (rig.asiair_hostname) secondaryParts.push(`via ${rig.asiair_hostname}`);

  const submitNickname = () => {
    if (!rig.anchored) return;
    onRename(rig.key, draftNickname);
  };

  return (
    <Box sx={{ display: "flex", flexDirection: "column", px: 2, py: 1 }}>
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          cursor: "pointer",
        }}
        onClick={() => setExpanded(e => !e)}
      >
        <Circle sx={{ fontSize: 10, color: _STATE_COLOR[rig.active_state], mr: 1 }} />
        <Box sx={{ flexGrow: 1, minWidth: 0 }}>
          <Typography variant="body2" noWrap>{titleLine}</Typography>
          <Typography variant="caption" color="text.secondary" noWrap>
            {secondaryParts.join(" · ")}
          </Typography>
        </Box>
        <IconButton size="small" tabIndex={-1}>
          {expanded ? <ExpandLess /> : <ExpandMore />}
        </IconButton>
      </Box>

      {expanded && (
        <Box sx={{ pl: 3, pt: 1 }}>
          {rig.anchored && (
            <>
              <Typography variant="caption" color="text.secondary" component="div">
                fingerprint: <code>{rig.key}</code>
              </Typography>
              {rig.last_instrume && (
                <Typography variant="caption" color="text.secondary" component="div">
                  camera: {rig.last_instrume}
                </Typography>
              )}
              <Typography variant="caption" color="text.secondary" component="div">
                ASIAir: {rig.asiair_hostname || "unknown"} ({rig.asiair_ip || "?"})
              </Typography>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1, mt: 1 }}>
                <TextField
                  size="small"
                  variant="outlined"
                  label="Nickname"
                  value={draftNickname}
                  onChange={(e) => setDraftNickname(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter") submitNickname(); }}
                  sx={{ flexGrow: 1 }}
                />
                <Button size="small" onClick={submitNickname}>Save</Button>
              </Box>
            </>
          )}
          {!rig.anchored && (
            <Tooltip title="The camera's fingerprint anchors after ~20 frames from it have been processed. The rename input unlocks then.">
              <Typography variant="caption" color="text.secondary" component="div">
                ASIAir: {rig.asiair_hostname || "unknown"} ({rig.asiair_ip || "?"}) — naming unlocks
                once the fingerprint anchors.
              </Typography>
            </Tooltip>
          )}
        </Box>
      )}
    </Box>
  );
};


const AsiairPanel = () => {
  const { asiairDevices, refreshAsiairDevices } = useContext(AstroFlowContext);
  const [anchorEl, setAnchorEl] = useState(null);

  const rigs = (asiairDevices && asiairDevices.rigs) || [];
  const summary = (asiairDevices && asiairDevices.summary) || { total: 0, active: 0 };

  const renameRig = (cameraHash, nickname) => {
    fetch(`/api/asiair/devices/${encodeURIComponent(cameraHash)}/nickname`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ nickname }),
    })
      .then(r => r.json())
      .then(() => { if (refreshAsiairDevices) refreshAsiairDevices(); })
      .catch(console.error);
  };

  const label = summary.total === 0
    ? "no rigs"
    : `${summary.total} ${summary.total === 1 ? "rig" : "rigs"} · ${summary.active} active`;
  const labelColor = summary.total === 0 ? "text.secondary" : "text.primary";

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
        onClick={(event) => setAnchorEl(event.currentTarget)}
      >
        <Wifi sx={{ fontSize: 16, mr: 1, color: labelColor }} />
        <Typography variant="body2" sx={{ mr: 1, color: labelColor }}>
          {label}
        </Typography>
        <IconButton size="small">{anchorEl ? <ExpandLess /> : <ExpandMore />}</IconButton>
      </Box>

      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={() => setAnchorEl(null)}
        slotProps={{ paper: { sx: { minWidth: 360, maxWidth: 480 } } }}
      >
        {rigs.length === 0 && (
          <MenuItem disabled>
            <ListItemText
              primary="No ASIAir devices discovered"
              secondary="Discovery runs over mDNS — connect an ASIAir to the same LAN."
            />
          </MenuItem>
        )}
        {rigs.length > 0 && (
          <List dense disablePadding>
            {rigs.map((rig, i) => (
              <React.Fragment key={rig.key}>
                {i > 0 && <Divider />}
                <RigRow rig={rig} onRename={renameRig} />
              </React.Fragment>
            ))}
          </List>
        )}
      </Menu>
    </>
  );
};

export default AsiairPanel;
