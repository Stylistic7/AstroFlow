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
import {
  ExpandMore, ExpandLess, Circle, AutoMode, Warning,
  WbSunny, CloudQueue, Cloud, FilterDrama,
} from "@mui/icons-material";
import AstroFlowContext from "./AstroFlowContext.jsx";
import DropdownChip from "./DropdownChip.jsx";
// Total guide error in arcsec per sample: sqrt(ra² + dec²). Drops
// the per-axis decomposition so the chip-level chart is one curve
// the user can read at a glance ("am I roughly steady or wandering?").
// Returns null entries for samples missing ra/dec — GuideChart skips them.
const _totalErrorSeries = (history) => (history || []).map(r => {
  if (r.ra_arcsec == null || r.dec_arcsec == null) return null;
  return Math.sqrt(r.ra_arcsec * r.ra_arcsec + r.dec_arcsec * r.dec_arcsec);
});


// Sky-quality level → chip-style tint. We deliberately use the SAME
// palette as the rig-row dot colors so the user learns one mapping.
const _SKY_COLOR = {
  good: "#81c784",      // success.light — clear
  thin: "#ffcc80",      // warning.light — transparency dropping
  heavy: "#ffab91",     // deep-orange-200 — visible cloud
  lost: "#e57373",      // error.light — guide star lost
  unknown: "#9e9e9e",   // grey.500 — no data yet
};


// Sky-quality level → icon. WbSunny is clear; the cloud variants
// escalate with level. "unknown" gets no icon so we don't show
// something misleading while the buffer is warming up.
const _SKY_ICON = {
  good: WbSunny,
  thin: CloudQueue,    // outline-only cloud — "starting to obstruct"
  heavy: Cloud,        // filled cloud
  lost: FilterDrama,   // dense overcast silhouette
  unknown: null,
};


// Pick a sparkline Y-max from a friendly tier list so the user knows
// what the chart's "ceiling" means at a glance. Most amateur guiding
// sits in 0-2″, hardcore astrophotographers hit < 0.5″, and anything
// above 5″ means something's wrong (clouds / wind / lost calibration).
const _PICK_AXIS_MAX = (maxValue) => {
  if (maxValue <= 1) return 1;
  if (maxValue <= 2) return 2;
  if (maxValue <= 5) return 5;
  if (maxValue <= 10) return 10;
  return Math.ceil(maxValue);
};


/**
 * Sparkline with a real Y-axis (left side, two ticks at 0 and axisMax)
 * plus a solid horizontal reference at 0 and a dashed reference at the
 * top. Built directly as SVG so we can position axis + reference lines
 * pixel-precisely. Used for both the chip's tiny chart and the
 * dropdown's larger one — same component, different `width` and
 * `height`. `axisLabelW` controls how much pixel space we reserve on
 * the left for the tick labels; the chip gets a slightly narrower
 * label gutter to stay compact.
 */
const GuideChart = ({
  values, width, height, color, axisMax, axisLabelW = 18,
}) => {
  const chartLeft = axisLabelW;
  const chartW = Math.max(0, width - chartLeft);
  const innerH = height;
  const norm = (v) => innerH - (v / axisMax) * innerH;

  const finite = values.map(v => Number.isFinite(v) ? v : null);
  const xStep = values.length > 1 ? chartW / (values.length - 1) : 0;
  let d = "";
  let inSeg = false;
  for (let i = 0; i < finite.length; i++) {
    if (finite[i] == null) {
      inSeg = false;
      continue;
    }
    const x = chartLeft + i * xStep;
    const y = norm(finite[i]);
    d += inSeg
      ? ` L${x.toFixed(2)},${y.toFixed(2)}`
      : ` M${x.toFixed(2)},${y.toFixed(2)}`;
    inSeg = true;
  }

  const axisColor = "#888";
  const refColor = "#555";
  const fontSize = axisLabelW >= 18 ? 9 : 8;

  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      {/* Horizontal reference at 0″ — solid baseline. */}
      <line
        x1={chartLeft} y1={innerH - 0.5}
        x2={width} y2={innerH - 0.5}
        stroke={refColor} strokeWidth={0.5}
      />
      {/* Horizontal reference at axisMax — dashed ceiling. */}
      <line
        x1={chartLeft} y1={0.5}
        x2={width} y2={0.5}
        stroke={refColor} strokeWidth={0.5}
        strokeDasharray="3,3"
      />
      {/* Y-axis. */}
      <line
        x1={chartLeft} y1={0}
        x2={chartLeft} y2={innerH}
        stroke={axisColor} strokeWidth={0.75}
      />
      {/* Tick marks at 0 and max. */}
      <line
        x1={chartLeft - 2} y1={0.5}
        x2={chartLeft} y2={0.5}
        stroke={axisColor} strokeWidth={0.75}
      />
      <line
        x1={chartLeft - 2} y1={innerH - 0.5}
        x2={chartLeft} y2={innerH - 0.5}
        stroke={axisColor} strokeWidth={0.75}
      />
      {/* Tick labels — top + bottom. */}
      <text
        x={chartLeft - 4} y={fontSize - 1}
        textAnchor="end" fontSize={fontSize} fill="#999"
      >{axisMax}″</text>
      <text
        x={chartLeft - 4} y={innerH - 1}
        textAnchor="end" fontSize={fontSize} fill="#999"
      >0″</text>
      {/* The data line. */}
      <path
        d={d} fill="none" stroke={color} strokeWidth={1.5}
        strokeLinejoin="round" strokeLinecap="round"
      />
    </svg>
  );
};


const _STATE_COLOR = {
  // green: capturing + guide healthy. The "everything's working" signal.
  green: "success.light",
  // warning (amber): capturing but guide_warning fires (clouds /
  // lost star). Capturing IS still happening; the frames are suspect.
  warning: "warning.light",
  // yellow: reachable but idle. White text — present, doing nothing.
  // The lack of color IS the signal; idle is the baseline.
  yellow: "text.primary",
  // gray: live poll didn't answer. The device may have been
  // powered down or fallen off the network. Dimmed so it visually
  // recedes from the rows that ARE reachable.
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


const RigRow = ({ rig, defaultExpanded = false }) => {
  const [expanded, setExpanded] = useState(defaultExpanded);
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
          {rig.guide_history && rig.guide_history.length > 0 && (
            <Box sx={{ mt: 1, pl: 1, borderLeft: "2px solid", borderColor: "divider" }}>
              <Typography variant="caption" color="text.secondary" component="div" sx={{ fontWeight: 600 }}>
                Guiding
              </Typography>
              {(() => {
                const latest = rig.guide_history[rig.guide_history.length - 1] || {};
                const sky = rig.sky || {};
                const series = _totalErrorSeries(rig.guide_history);
                const skyColor = _SKY_COLOR[sky.level || "unknown"];
                const seriesMax = Math.max(0, ...series.filter(v => v != null));
                const axisMax = _PICK_AXIS_MAX(seriesMax);
                return (
                  <>
                    <Box sx={{ mt: 0.5 }}>
                      <GuideChart
                        values={series}
                        width={260}
                        height={60}
                        color={skyColor}
                        axisMax={axisMax}
                        axisLabelW={22}
                      />
                    </Box>
                    <Typography variant="caption" color="text.secondary" component="div">
                      RMS: {latest.rms_arcsec != null ? `${latest.rms_arcsec.toFixed(2)}″` : "?"}
                      {latest.ra_arcsec != null && (
                        <>{" "}· RA: {latest.ra_arcsec.toFixed(2)}″</>
                      )}
                      {latest.dec_arcsec != null && (
                        <>{" "}· DEC: {latest.dec_arcsec.toFixed(2)}″</>
                      )}
                    </Typography>
                    <Typography variant="caption" color="text.secondary" component="div">
                      SNR: {latest.snr ?? "?"}
                      {" "}· star mass: {latest.star_mass ?? "?"}
                      {latest.ra_pulse_ms != null && (
                        <>{" "}· pulses: {latest.ra_pulse_ms}/{latest.dec_pulse_ms} ms</>
                      )}
                    </Typography>
                    <Typography
                      variant="caption"
                      component="div"
                      sx={{ color: skyColor, fontWeight: 500, mt: 0.5 }}
                    >
                      sky: {sky.caption || "no data"}
                      {sky.snr_ratio != null && (
                        <>{" "}({(sky.snr_ratio * 100).toFixed(0)}% SNR
                        {" / "}{(sky.star_mass_ratio * 100).toFixed(0)}% mass)</>
                      )}
                    </Typography>
                  </>
                );
              })()}
            </Box>
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
  // tint. Escalation order, most-severe first:
  //   - lost guide star (likely heavy clouds)         → error
  //   - heavy clouds / capturing-without-guide        → warning
  //   - thin clouds                                   → warning (gentler)
  //   - anything actively capturing in clear sky      → active
  //   - else                                          → default
  const reachableRigsForTone = rigs.filter(r => r.reachable);
  const _toneSkyRank = { good: 0, unknown: 0, thin: 1, heavy: 2, lost: 3 };
  const worstSkyForTone = reachableRigsForTone.reduce((worst, r) => {
    const level = r.sky?.level || "unknown";
    return (_toneSkyRank[level] || 0) > (_toneSkyRank[worst] || 0) ? level : worst;
  }, "good");
  const tone =
    worstSkyForTone === "lost" ? "error"
    : worstSkyForTone === "heavy" || warnings > 0 ? "warning"
    : worstSkyForTone === "thin" ? "warning"
    : summary.active > 0 ? "active"
    : "default";

  // Chip text counts only REACHABLE rigs — a Bonjour-stale record
  // that isn't actually answering TCP polls is effectively absent
  // for the user's purposes. The dropdown body still renders every
  // discovered rig (with its own per-row state), so diagnostic info
  // isn't lost; the chip just stays honest about what's live.
  //
  // Three modes:
  //   - 0 reachable: "0 ASIAir" — uniform with the 2+ form.
  //   - 1 reachable: name + activity, e.g. "ASIAIR_RC71 idle" or
  //        "ASIAIR_RC71 capturing 13/60". One specific device, one
  //        word for what it's doing.
  //   - 2+ reachable: aggregate, e.g. "3 ASIAir · 2 active".
  //
  // Brand name stays "ASIAir" (no plural "s") to match the product.
  const reachableRigs = rigs.filter(r => r.reachable);
  let summaryNode;
  if (reachableRigs.length === 0) {
    summaryNode = (
      <Typography variant="body2" color="text.secondary">
        0 ASIAir
      </Typography>
    );
  } else if (reachableRigs.length === 1) {
    const r = reachableRigs[0];
    // No verbose "capturing" text — the icon color does that signal
    // (green when active, white when idle). If frames are being
    // counted (stack_total > 0) we show the count tersely; otherwise
    // the chip is just label + guide chart.
    const countText = (
      r.capture_state && r.capture_state !== "idle" && r.stack_total > 0
    ) ? `${r.stack_done}/${r.stack_total}` : null;
    // When guide data is flowing, render a chart of total arcsec
    // error tinted by the sky-quality level. PHD2's AvgDist (last
    // sample's `rms_arcsec`) is the headline number; the chart's
    // SHAPE shows whether guiding is steady or wandering, the COLOR
    // shows whether the sky is letting us see clearly.
    const series = _totalErrorSeries(r.guide_history);
    const hasGuideData = series.some(v => v != null);
    const skyLevel = r.sky?.level || "unknown";
    const sparkColor = _SKY_COLOR[skyLevel];
    const latestRms = r.guide_history?.length
      ? r.guide_history[r.guide_history.length - 1].rms_arcsec
      : null;
    const seriesMax = hasGuideData
      ? Math.max(...series.filter(v => v != null))
      : 0;
    const axisMax = _PICK_AXIS_MAX(seriesMax);
    summaryNode = (
      <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
        <Typography variant="body2">
          {r.label}
        </Typography>
        {countText && (
          <Typography variant="caption" color="text.secondary">
            {countText}
          </Typography>
        )}
        {hasGuideData && (
          <>
            <GuideChart
              values={series}
              width={96}
              height={28}
              color={sparkColor}
              axisMax={axisMax}
              axisLabelW={16}
            />
            {latestRms != null && (
              <Typography variant="caption" color="text.secondary">
                {`${latestRms.toFixed(2)}″`}
              </Typography>
            )}
          </>
        )}
      </Box>
    );
  } else {
    summaryNode = (
      <Typography variant="body2">
        {`${reachableRigs.length} ASIAir · ${summary.active} active`}
      </Typography>
    );
  }

  // Chip icon priority:
  //   1. Warning (capturing but guide loop sick) — most urgent.
  //   2. Sky-quality icon if any reachable rig has a non-good /
  //      non-unknown sky reading. This is what tells the user "the
  //      weather is showing up in the data."
  //   3. AutoMode otherwise — neutral "ASIAir as automation rig"
  //      glyph; less specific than a Camera, more specific than Wifi.
  const SkyIcon = _SKY_ICON[worstSkyForTone];
  let icon;
  if (warnings > 0) {
    icon = <Warning />;
  } else if (SkyIcon && worstSkyForTone !== "good" && worstSkyForTone !== "unknown") {
    icon = <SkyIcon />;
  } else {
    icon = <AutoMode />;
  }
  const tooltipTitle =
    warnings > 0 ? `${warnings} ASIAir(s) capturing without a healthy guide loop`
    : worstSkyForTone === "thin" ? "Thin clouds — transparency dropping"
    : worstSkyForTone === "heavy" ? "Heavy clouds — guide star dim"
    : worstSkyForTone === "lost" ? "Guide star lost — likely thick clouds"
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
                <RigRow rig={rig} defaultExpanded={rigs.length === 1} />
              </React.Fragment>
            ))}
          </List>
        )}
    </DropdownChip>
  );
};

export default AsiairPanel;
