import React, { useState, useContext } from "react";
import {
  Box,
  MenuItem,
  Typography,
  IconButton,
  List,
  Divider,
  Tooltip,
} from "@mui/material";
import { ExpandMore, ExpandLess, Circle, AutoMode, Warning } from "@mui/icons-material";
import AstroFlowContext from "./AstroFlowContext.jsx";
import DropdownChip from "./DropdownChip.jsx";


const _STATE_COLOR = {
  green: "success.light",
  yellow: "warning.light",
  gray: "text.disabled",
};


const _formatRelativeTime = (epochSeconds) => {
  if (!epochSeconds) return null;
  const ageS = Math.max(0, Date.now() / 1000 - epochSeconds);
  if (ageS < 60) return "moments ago";
  if (ageS < 60 * 60) return `${Math.round(ageS / 60)} min ago`;
  if (ageS < 60 * 60 * 24) return `${Math.round(ageS / 3600)} h ago`;
  return `${Math.round(ageS / 86400)} d ago`;
};


const RigRow = ({ rig }) => {
  const [expanded, setExpanded] = useState(false);
  const live = rig.live || null;

  // Primary line: the device's user-set name (from the ZWO app),
  // taken from `get_device_state.result.ap.ssid` by the live poller.
  // Falls back to the mDNS hostname (stripped of `.local.`) when the
  // poll hasn't succeeded yet.
  const titleLine = rig.label || "ASIAir";

  // Secondary line: live progress signals. "capturing X/Y" is the
  // headline the user cares about — that the rig is actively making
  // frames. Guide state surfaces alongside; the warning state shows
  // as a ⚠️ icon (rendered separately) when the guider stopped
  // mid-capture.
  const secondaryParts = [];
  if (rig.capture_state && rig.capture_state !== "idle") {
    if (rig.stack_total > 0) {
      secondaryParts.push(`capturing ${rig.stack_done}/${rig.stack_total}`);
    } else {
      secondaryParts.push(rig.capture_state);
    }
  } else if (rig.reachable) {
    secondaryParts.push("idle");
  } else {
    secondaryParts.push("offline");
  }
  if (rig.guide_state && rig.guide_state !== "stopped") {
    secondaryParts.push(`guide: ${rig.guide_state}`);
  }
  if (rig.last_goto_target) secondaryParts.push(`target ${rig.last_goto_target}`);

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
        {rig.guide_warning && (
          <Tooltip title="Capturing but guider has stopped — possibly clouds, lost star, or guide-camera disconnect.">
            <Warning sx={{ fontSize: 16, color: "warning.main", mr: 1 }} />
          </Tooltip>
        )}
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
          <Typography variant="caption" color="text.secondary" component="div">
            hostname: {rig.hostname || "unknown"} ({rig.ip || "?"})
          </Typography>
          {rig.camera_name && (
            <Typography variant="caption" color="text.secondary" component="div">
              camera: {rig.camera_name} ({rig.camera_state || "?"})
            </Typography>
          )}
          {rig.camera_hash_short && (
            <Tooltip title="Defect-pixel fingerprint of the connected camera. Used internally to associate calibration frames; not normally something you'd act on.">
              <Typography variant="caption" color="text.secondary" component="div">
                camera fingerprint: <code>{rig.camera_hash_short}</code>
              </Typography>
            </Tooltip>
          )}
          {live && live.reachable && (
            <Box sx={{ mt: 1, pl: 1, borderLeft: "2px solid", borderColor: "divider" }}>
              <Typography variant="caption" color="text.secondary" component="div" sx={{ fontWeight: 600 }}>
                Live
              </Typography>
              {live.firmware && (
                <Typography variant="caption" color="text.secondary" component="div">
                  firmware: {live.firmware}
                </Typography>
              )}
              {live.model && (
                <Typography variant="caption" color="text.secondary" component="div">
                  model: {live.model}
                </Typography>
              )}
              {live.focal_len_mm && (
                <Typography variant="caption" color="text.secondary" component="div">
                  focal length: {live.focal_len_mm} mm
                </Typography>
              )}
              {live.focuser_state && (
                <Typography variant="caption" color="text.secondary" component="div">
                  focuser: {live.focuser_state}
                </Typography>
              )}
              {typeof live.pi_temp_c === "number" && (
                <Typography variant="caption" color="text.secondary" component="div">
                  Pi CPU temp: {live.pi_temp_c.toFixed(1)}°C
                </Typography>
              )}
              {Array.isArray(live.location) && live.location.length === 2 && (
                <Typography variant="caption" color="text.secondary" component="div">
                  observing at: {live.location[1].toFixed(4)}°N, {live.location[0].toFixed(4)}°E
                </Typography>
              )}
              {live.guide_app_state_raw !== undefined && (
                <Typography variant="caption" color="text.secondary" component="div">
                  guider state: {String(live.guide_app_state_raw)}
                </Typography>
              )}
              {live.queried_at && (
                <Typography variant="caption" color="text.secondary" component="div">
                  last polled: {_formatRelativeTime(live.queried_at) || "just now"}
                </Typography>
              )}
            </Box>
          )}
          {!rig.reachable && (
            <Typography variant="caption" color="error.light" component="div" sx={{ mt: 1 }}>
              Not responding to live polls. The mDNS record is still alive
              but TCP 4700 / 4400 didn't answer. Network blip, or the
              device powered down.
            </Typography>
          )}
        </Box>
      )}
    </Box>
  );
};


const AsiairPanel = () => {
  const { asiairDevices } = useContext(AstroFlowContext);

  const rigs = (asiairDevices && asiairDevices.rigs) || [];
  const summary = (asiairDevices && asiairDevices.summary) || { total: 0, active: 0, warnings: 0 };
  const warnings = summary.warnings || 0;

  // The tone drives both the icon color and the chip's background
  // tint. Warning state wins; otherwise we tint green when anything
  // is actively capturing, leave default when idle.
  const tone =
    warnings > 0 ? "warning"
    : summary.active > 0 ? "active"
    : "default";

  // Chip label: keep the existing "N ASIAirs · M active" shape, which
  // a tooltip can elaborate on. The icon shifts to a Warning glyph
  // when the tone calls for it, mirroring the swap we shipped before.
  const summaryNode = summary.total === 0
    ? <Typography variant="body2" color="text.secondary">no ASIAirs</Typography>
    : <Typography variant="body2">
        {`${summary.total} ${summary.total === 1 ? "ASIAir" : "ASIAirs"} · ${summary.active} active`}
      </Typography>;

  // AutoMode = "star + automation loop" — captures what an ASIAir
  // actually IS (an automation rig pointed at the sky) better than
  // a generic Wifi icon, and dodges the over-specific Camera glyph.
  const icon = warnings > 0 ? <Warning /> : <AutoMode />;
  const tooltipTitle =
    warnings > 0 ? `${warnings} ASIAir(s) capturing without a healthy guide loop`
    : summary.total === 0 ? "No ASIAirs discovered on the LAN"
    : "ASIAir devices";

  return (
    <DropdownChip
      icon={icon}
      summary={summaryNode}
      tone={tone}
      tooltipTitle={tooltipTitle}
      ariaLabel="ASIAir devices"
    >
        {rigs.length === 0 && (
          <MenuItem disabled>
            <Box>
              <Typography variant="body2">No ASIAirs on the LAN</Typography>
              <Typography variant="caption" color="text.secondary">
                Discovery runs over mDNS — connect an ASIAir to the same network.
              </Typography>
            </Box>
          </MenuItem>
        )}
        {rigs.length > 0 && (
          <List dense disablePadding>
            {rigs.map((rig, i) => (
              <React.Fragment key={rig.key}>
                {i > 0 && <Divider />}
                <RigRow rig={rig} />
              </React.Fragment>
            ))}
          </List>
        )}
    </DropdownChip>
  );
};

export default AsiairPanel;
