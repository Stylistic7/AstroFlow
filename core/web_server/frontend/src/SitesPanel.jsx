// Observing Sites: every site the library's frames state (`GET /api/sites`,
// light_pollution), each with its sky-glow model drawn as the dome seen from
// below — percent brighter than the zenith by altitude and azimuth, the city
// light domes — the frames that measured it as dots, and the model's numbers;
// or why the site has no model yet.
import React, { useEffect, useState } from 'react';
import { Box, Chip, Table, TableBody, TableCell, TableRow, Tooltip, Typography } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { PanelPage, PanelSection, PanelStatus } from './PanelPage.jsx';

const ALTS = [...Array(16)].map((_, i) => 15 + 5 * i);     // the grid's rows
const AZS = [...Array(36)].map((_, j) => 10 * j);          // and columns
const STOPS = [[0, [20, 11, 46]], [0.5, [192, 57, 43]], [1, [255, 224, 102]]];

function ramp(value, vmax) {
  const t = Math.max(0, Math.min(1, value / vmax));
  const k = STOPS.findIndex(([at]) => at >= t);
  const [t0, c0] = STOPS[Math.max(0, k - 1)];
  const [t1, c1] = STOPS[Math.max(0, k)];
  const f = t1 === t0 ? 0 : (t - t0) / (t1 - t0);
  return `rgb(${c0.map((v, i) => Math.round(v + f * (c1[i] - v))).join(',')})`;
}

// Looking up: north at the top, east on the LEFT. Altitude a sits at radius R·(90−a)/75.
function point(c, R, alt, az) {
  const r = R * (90 - Math.min(90, alt)) / 75;
  const a = (az * Math.PI) / 180;
  return [c - r * Math.sin(a), c - r * Math.cos(a)];
}

function wedge(c, R, alt, az) {
  const [x0, y0] = point(c, R, alt - 2.5, az - 5);
  const [x1, y1] = point(c, R, alt - 2.5, az + 5);
  const [x2, y2] = point(c, R, alt + 2.5, az + 5);
  const [x3, y3] = point(c, R, alt + 2.5, az - 5);
  const ro = R * (90 - (alt - 2.5)) / 75;
  const ri = R * (90 - Math.min(90, alt + 2.5)) / 75;
  return `M${x0},${y0} A${ro},${ro} 0 0 0 ${x1},${y1} L${x2},${y2} A${ri},${ri} 0 0 1 ${x3},${y3} Z`;
}

function DomeMap({ grid, sigma, samples, size = 340 }) {
  const theme = useTheme();
  const c = size / 2;
  const R = c - 18;
  const values = grid.flat().filter((v) => v > 0).sort((a, b) => a - b);
  const vmax = Math.max(10, values[Math.floor(values.length * 0.95)] || 0);
  const ink = theme.palette.text.secondary;
  // The fit's own spread (bootstrap) in a direction; unsure when it is more than half the value there.
  const spread = (i, j) => (sigma && grid[i][j] !== null && sigma[i][j] !== null ? sigma[i][j] : null);
  const unsure = (i, j) => spread(i, j) !== null && spread(i, j) > Math.max(2.5, Math.abs(grid[i][j]) / 2);
  return (
    <Box>
      <svg width={size} height={size} style={{ display: 'block' }}>
        <defs>
          <pattern id="unsure" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="4" stroke={theme.palette.background.paper} strokeWidth="1.2" />
          </pattern>
        </defs>
        {ALTS.map((alt, i) => AZS.map((az, j) => {
          const v = grid[i][j];                       // null: outside the model's coverage — no frames were shot there
          const s = spread(i, j);
          const title = v === null ? 'no frames here' : `${v >= 0 ? '+' : ''}${v.toFixed(0)}% vs the zenith${s === null ? '' : ` ± ${s.toFixed(0)}`}`;
          return (
            <Tooltip key={`${i}-${j}`} title={`alt ${alt}° az ${az}°: ${title}`}>
              <path d={wedge(c, R, alt, az)} fill={v === null ? theme.palette.action.disabledBackground : ramp(v, vmax)} stroke="none" />
            </Tooltip>
          );
        }))}
        {ALTS.map((alt, i) => AZS.map((az, j) => unsure(i, j) && (
          <path key={`u${i}-${j}`} d={wedge(c, R, alt, az)} fill="url(#unsure)" stroke="none" pointerEvents="none" />
        )))}
        {[30, 60].map((alt) => (
          <circle key={alt} cx={c} cy={c} r={R * (90 - alt) / 75} fill="none" stroke={ink} strokeOpacity={0.5} strokeDasharray="3 3" />
        ))}
        <circle cx={c} cy={c} r={R} fill="none" stroke={ink} />
        {(samples || []).map(([alt, az, moonless], k) => {
          const [x, y] = point(c, R, alt, az);
          return <circle key={k} cx={x} cy={y} r={1.6} fill={moonless ? '#9be7ff' : 'none'} stroke="#9be7ff" strokeWidth={0.8} opacity={0.7} />;
        })}
        {[['N', 0], ['E', 90], ['S', 180], ['W', 270]].map(([label, az]) => {
          const [x, y] = point(c, R + 11, 15, az);
          return <text key={label} x={x} y={y + 4} fontSize={12} textAnchor="middle" fill={theme.palette.text.primary}>{label}</text>;
        })}
        {[30, 60].map((alt) => <text key={alt} x={c + 3} y={c - R * (90 - alt) / 75 - 2} fontSize={9} fill={ink}>{alt}°</text>)}
      </svg>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
        <Box sx={{ width: 120, height: 10, borderRadius: 1, background: `linear-gradient(90deg, ${ramp(0, 1)}, ${ramp(0.5, 1)}, ${ramp(1, 1)})` }} />
        <Typography variant="caption" color="text.secondary">0 → {vmax.toFixed(0)}% brighter than the zenith · grey: no frames · hatched: unsure, the fit's spread is over half the value · dots: frames (hollow: Moon up)</Typography>
      </Box>
    </Box>
  );
}

function Numbers({ site }) {
  const m = site.model;
  const rows = [
    ['Frames', `${site.frames} in ${site.runs} runs of ${site.stacks} fields · ${site.usable_frames} in astronomical night · ${site.moonlit_frames} with the Moon up`],
    ['Newest run', site.newest_run ? site.newest_run.slice(0, 10) : '—'],
  ];
  if (m) {
    rows.push(['Model', m.hash]);
    rows.push(['Fitted on', `${m.runs} moonless runs, ${m.frames} frames`]);
    const byFilter = Object.entries(site.explained_by_filter || {}).map(([f, v]) => `${f} ${(100 * v).toFixed(0)}`).join(' · ');
    rows.push(['Explains', `${(100 * m.explained).toFixed(0)}% of the tilt changes across a field's moonless frames${byFilter ? ` (${byFilter})` : ''} · ${(100 * m.explained_moonlit).toFixed(0)}% across moonlit ones`]);
    const g = site.gate;                     // the last refit against the fit's own spread, where the frames are
    if (g) {
      rows.push(['Last refit', g.previous === null ? `first model · spread ${g.noise.toFixed(2)} %/°`
        : `${g.kept ? 'kept' : 'replaced'} ${g.previous.slice(0, 8)} · change ${g.change.toFixed(2)} vs spread ${g.noise.toFixed(2)} %/° · largest ${g.local.toFixed(1)} %/° at ${g.local_significance.toFixed(0)}σ${g.amplitudes_changed ? ' · amplitudes changed' : ''}`]);
    }
    for (const [filt, a] of Object.entries(m.amplitudes || {})) {
      rows.push([`${filt} amplitudes`, `city ${a.city.toFixed(3)} · moonlight ${a.moon.toFixed(3)} counts/s${a.moon === 0 ? ' (too few moonlit runs: not Moon-corrected)' : ` · ${a.nights} nights with their own`}`]);
    }
  }
  return (
    <Table size="small" sx={{ maxWidth: 560 }}>
      <TableBody>
        {rows.map(([k, v]) => (
          <TableRow key={k}>
            <TableCell sx={{ whiteSpace: 'nowrap', color: 'text.secondary' }}>{k}</TableCell>
            <TableCell sx={{ fontVariantNumeric: 'tabular-nums' }}>{v}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

export default function SitesPanel() {
  const [sites, setSites] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const r = await fetch('/api/sites');
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
        const data = await r.json();
        if (alive) { setSites(data); setError(null); }
      } catch (e) {
        if (alive) setError(String(e));
      }
    };
    load();
    const t = setInterval(load, 60000);
    return () => { alive = false; clearInterval(t); };
  }, []);

  if (error) return <PanelStatus title="Observing Sites" error={`Observing sites unavailable: ${error}`} />;
  if (sites === null) return <PanelStatus title="Observing Sites">Loading sites…</PanelStatus>;

  return (
    <PanelPage title="Observing Sites" intro={<>
        Every place the library's frames say they were shot from, and the sky glow its own subframes
        reveal there: the city light domes as seen from below, and how strongly moonlight scatters into
        each filter. The "Siril + Site Model + OSS Mosaic" pipeline subtracts each subframe's predicted
        glow before stacking. Frames that do not state a position belong to no site and are never corrected.
      </>}>
      {sites.length === 0 && (
        <Typography color="text.secondary">No sites yet — the fold runs once the library's runs are measured.</Typography>
      )}
      {sites.map((site) => (
        <PanelSection key={site.key} title={`${site.lat.toFixed(2)}°, ${site.lon.toFixed(2)}°`}
          meta={site.model ? <Chip size="small" color="success" label="model" /> : <Chip size="small" label="no model" />}>
          <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap', alignItems: 'flex-start' }}>
            {site.grid ? <DomeMap grid={site.grid} sigma={site.sigma} samples={site.samples} /> : (
              <Typography color="text.secondary" sx={{ maxWidth: 340 }}>No model yet: {site.reason || 'not enough data'}.</Typography>
            )}
            <Numbers site={site} />
          </Box>
        </PanelSection>
      ))}
    </PanelPage>
  );
}
