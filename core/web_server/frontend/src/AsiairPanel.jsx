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
  AutoAwesome, BlurOn, WbCloudy, FilterDrama,
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


// Sky-quality level → icon.
//   good  → AutoAwesome (sparkles) — clear stars visible, the
//           "you're doing astrophotography right now" glyph.
//   thin  → BlurOn — diffuse dotted haze. Matches the "thin clouds
//           that are quietly killing SNR" failure mode where stars
//           are still visible but transparency is dropping.
//   heavy → WbCloudy — overt cloud; visible obstruction.
//   lost  → FilterDrama — dense overcast silhouette; total
//           obscuration / guide star gone.
// "unknown" stays null — no icon while the estimator's warming up.
const _SKY_ICON = {
  good: AutoAwesome,
  thin: BlurOn,
  heavy: WbCloudy,
  lost: FilterDrama,
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
 * One-series unsigned guide chart for the chip — total arcsec error,
 * never negative. Y-axis runs 0 to axisMax, solid reference at 0,
 * dashed reference at axisMax. See GuideAxisChart for the signed
 * multi-series version used in the dropdown.
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


/**
 * Signed multi-series guide chart. Y-axis runs -axisMax to +axisMax
 * with the zero line in the middle. Used in the dropdown to show RA
 * + DEC + RMS together — RA and DEC swing positive or negative
 * (their signs tell you which way the star drifted before the
 * correction); RMS stays non-negative (PHD2's running AvgDist).
 *
 * `series` is an array of { values: number[], color: string,
 * label: string }. Each line is rendered independently; missing
 * samples (null/undefined) are skipped without breaking the path.
 */
const GuideAxisChart = ({
  series, width, height, axisMax, axisLabelW = 24,
}) => {
  const chartLeft = axisLabelW;
  const chartW = Math.max(0, width - chartLeft);
  const innerH = height;
  // Y midline = zero. Positive values draw above, negative below.
  const yZero = innerH / 2;
  const norm = (v) => yZero - (v / axisMax) * (innerH / 2);

  const buildPath = (vals) => {
    if (!vals || vals.length === 0) return "";
    const xStep = vals.length > 1 ? chartW / (vals.length - 1) : 0;
    let d = "";
    let inSeg = false;
    for (let i = 0; i < vals.length; i++) {
      const v = vals[i];
      if (v == null || !Number.isFinite(v)) {
        inSeg = false;
        continue;
      }
      const x = chartLeft + i * xStep;
      const y = norm(v);
      d += inSeg
        ? ` L${x.toFixed(2)},${y.toFixed(2)}`
        : ` M${x.toFixed(2)},${y.toFixed(2)}`;
      inSeg = true;
    }
    return d;
  };

  const axisColor = "#888";
  const refColor = "#555";
  const fontSize = 9;

  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      {/* Solid zero line — the "perfect" axis. */}
      <line
        x1={chartLeft} y1={yZero}
        x2={width} y2={yZero}
        stroke={axisColor} strokeWidth={0.75}
      />
      {/* Dashed +axisMax and -axisMax references. */}
      <line
        x1={chartLeft} y1={0.5}
        x2={width} y2={0.5}
        stroke={refColor} strokeWidth={0.5}
        strokeDasharray="3,3"
      />
      <line
        x1={chartLeft} y1={innerH - 0.5}
        x2={width} y2={innerH - 0.5}
        stroke={refColor} strokeWidth={0.5}
        strokeDasharray="3,3"
      />
      {/* Y-axis vertical. */}
      <line
        x1={chartLeft} y1={0}
        x2={chartLeft} y2={innerH}
        stroke={axisColor} strokeWidth={0.75}
      />
      {/* Three ticks: +max, 0, -max. */}
      <line x1={chartLeft - 2} y1={0.5} x2={chartLeft} y2={0.5}
        stroke={axisColor} strokeWidth={0.75} />
      <line x1={chartLeft - 2} y1={yZero} x2={chartLeft} y2={yZero}
        stroke={axisColor} strokeWidth={0.75} />
      <line x1={chartLeft - 2} y1={innerH - 0.5} x2={chartLeft} y2={innerH - 0.5}
        stroke={axisColor} strokeWidth={0.75} />
      {/* Tick labels. */}
      <text x={chartLeft - 4} y={fontSize - 1}
        textAnchor="end" fontSize={fontSize} fill="#999">
        {`+${axisMax}″`}
      </text>
      <text x={chartLeft - 4} y={yZero + 3}
        textAnchor="end" fontSize={fontSize} fill="#999">
        0
      </text>
      <text x={chartLeft - 4} y={innerH - 1}
        textAnchor="end" fontSize={fontSize} fill="#999">
        {`−${axisMax}″`}
      </text>
      {/* Series — render in given order so RMS sits on top. */}
      {series.map((s, i) => (
        <path
          key={i}
          d={buildPath(s.values)}
          fill="none"
          stroke={s.color}
          strokeWidth={s.strokeWidth || 1.25}
          strokeLinejoin="round"
          strokeLinecap="round"
          strokeDasharray={s.dashed ? "3,3" : undefined}
        />
      ))}
    </svg>
  );
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
                const skyColor = _SKY_COLOR[sky.level || "unknown"];
                // Three series: signed RA, signed DEC, unsigned RMS.
                // RA / DEC swing through zero; RMS rides above. Axis
                // tier picks the max ABSOLUTE value across all three so
                // a -3″ excursion still fits.
                const raSeries = rig.guide_history.map(r => r.ra_arcsec);
                const decSeries = rig.guide_history.map(r => r.dec_arcsec);
                const absMax = Math.max(
                  0,
                  ...raSeries.filter(v => v != null).map(Math.abs),
                  ...decSeries.filter(v => v != null).map(Math.abs),
                );
                const axisMax = _PICK_AXIS_MAX(absMax);
                return (
                  <>
                    <Box sx={{ mt: 0.5 }}>
                      <GuideAxisChart
                        width={260}
                        height={80}
                        axisMax={axisMax}
                        axisLabelW={26}
                        series={[
                          { values: raSeries, color: "#ef9a9a", label: "RA" },   // red.300
                          { values: decSeries, color: "#90caf9", label: "DEC" }, // blue.300
                        ]}
                      />
                    </Box>
                    {/* Color legend: RA + DEC only — RMS is shown as
                        a number in the readout below the chart, the
                        same way PHD2 presents it. */}
                    <Box sx={{ display: "flex", gap: 1.5, mt: 0.5, ml: 3.25 }}>
                      <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                        <Box sx={{ width: 10, height: 2, backgroundColor: "#ef9a9a" }} />
                        <Typography variant="caption" color="text.secondary">RA</Typography>
                      </Box>
                      <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                        <Box sx={{ width: 10, height: 2, backgroundColor: "#90caf9" }} />
                        <Typography variant="caption" color="text.secondary">DEC</Typography>
                      </Box>
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
  } else {
    // Single + multi-rig share the same chip layout: a "feature rig"
    // (the worst-performing one, by latest RMS) gets its full signed
    // guide chart in the chip. For 2+ rigs we prepend a "N of M"
    // badge so the user knows the chip is surfacing one of several.
    // Single-rig: the only rig IS the feature rig.
    const _latestRms = r => {
      if (!r.guide_history?.length) return null;
      const last = r.guide_history[r.guide_history.length - 1] || {};
      return last.rms_arcsec ?? null;
    };
    // Pick the worst-currently-guiding rig. If none have guide data,
    // fall back to the first reachable. "Worst" = highest latest RMS;
    // null RMS sorts last (rigs without data don't dominate the chip).
    const featureRig = reachableRigs.slice().sort((a, b) => {
      const ra = _latestRms(a), rb = _latestRms(b);
      if (ra == null && rb == null) return 0;
      if (ra == null) return 1;
      if (rb == null) return -1;
      return rb - ra;
    })[0];
    const r = featureRig;
    const countText = (
      r.capture_state && r.capture_state !== "idle" && r.stack_total > 0
    ) ? `${r.stack_done}/${r.stack_total}` : null;
    const raSeries = (r.guide_history || []).map(s => s.ra_arcsec);
    const decSeries = (r.guide_history || []).map(s => s.dec_arcsec);
    const rmsSeries = (r.guide_history || []).map(s => s.rms_arcsec);
    const hasGuideData = raSeries.some(v => v != null);
    const skyLevel = r.sky?.level || "unknown";
    const sparkColor = _SKY_COLOR[skyLevel];
    const latestRms = _latestRms(r);
    const absMax = Math.max(
      0,
      ...raSeries.filter(v => v != null).map(Math.abs),
      ...decSeries.filter(v => v != null).map(Math.abs),
      ...rmsSeries.filter(v => v != null),
    );
    const axisMax = _PICK_AXIS_MAX(absMax);
    const RigSkyIcon = _SKY_ICON[skyLevel];
    const skyTooltip =
      skyLevel === "good" ? "Sky clear — guide signal at baseline"
      : skyLevel === "thin" ? "Thin clouds — transparency dropping"
      : skyLevel === "heavy" ? "Heavy clouds — guide star dim"
      : skyLevel === "lost" ? "Guide star lost — thick clouds likely"
      : "Sky-quality estimator warming up";
    summaryNode = (
      <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
        {RigSkyIcon && (
          <Tooltip title={skyTooltip}>
            <RigSkyIcon sx={{ fontSize: 16, color: sparkColor }} />
          </Tooltip>
        )}
        <Typography variant="body2">{r.label}</Typography>
        {/* Multi-rig badge: tell the user the chip is featuring ONE
            rig (the worst) out of N. Dropdown shows them all. */}
        {reachableRigs.length > 1 && (
          <Tooltip title={`Showing the worst-guiding of ${reachableRigs.length} ASIAirs. Open the dropdown for per-rig detail.`}>
            <Typography variant="caption" color="text.disabled">
              {`(worst of ${reachableRigs.length})`}
            </Typography>
          </Tooltip>
        )}
        {countText && (
          <Typography variant="caption" color="text.secondary">
            {countText}
          </Typography>
        )}
        {hasGuideData && (
          <>
            <GuideAxisChart
              width={140}
              height={40}
              axisMax={axisMax}
              axisLabelW={20}
              series={[
                { values: raSeries, color: "#ef9a9a", label: "RA" },
                { values: decSeries, color: "#90caf9", label: "DEC" },
              ]}
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
  }

  // Leading icon: Warning if a guide loop is sick mid-capture, else
  // AutoMode (the "this chip IS the ASIAir surface" glyph). The
  // sky-quality icon lives INSIDE the summary content so we get the
  // sky reading visible regardless of conditions (sun glyph in good
  // weather, cloud glyphs in bad), without doubling up icons.
  const icon = warnings > 0 ? <Warning /> : <AutoMode />;
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
